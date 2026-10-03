"""Shared CV machinery for round-3 scripts: same 5x3 folds as v1/v2, OOF store in outputs/eda/v3_oof.csv."""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from common import OUT, SEED, h, load, plt, save  # noqa: F401  (re-exported)

warnings.filterwarnings("ignore")
tr, te, F = load()
X, y, Xte = tr[F].values, tr.target.values, te[F].values
X_all = np.vstack([X, Xte])
N = len(y)
FOLDS = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X))
OOF_PATH, TEST_PATH = OUT / "v3_oof.csv", OUT / "v3_test.csv"
rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def run_cv(fit_predict, Xm=X, Xm_te=None, name=None, test=True):
    """fit_predict(A, t, B) -> predictions for B. Returns (mean-over-repeats RMSE, fold RMSEs, bagged OOF)."""
    t0 = time.time()
    oof = np.zeros((3, N))
    for i, (a, b) in enumerate(FOLDS):
        oof[i // 5, b] = fit_predict(Xm[a], y[a], Xm[b])
    fold = [rmse(y[b], oof[i // 5, b]) for i, (a, b) in enumerate(FOLDS)]
    score = float(np.mean([rmse(y, o) for o in oof]))
    if name:
        store(name, oof.mean(0), fit_predict(Xm, y, Xm_te if Xm_te is not None else Xte) if test else None)
        print(f"  {name:34s} RMSE={score:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    return score, np.array(fold), oof.mean(0)


def store(name, oof, pred_te=None):
    o = pd.read_csv(OOF_PATH) if OOF_PATH.exists() else pd.DataFrame({"record_id": tr.record_id, "target": y})
    o[name] = oof
    o.to_csv(OOF_PATH, index=False)
    if pred_te is not None:
        t = pd.read_csv(TEST_PATH) if TEST_PATH.exists() else pd.DataFrame({"record_id": te.record_id})
        t[name] = pred_te
        t.to_csv(TEST_PATH, index=False)


def load_oof():
    return pd.read_csv(OOF_PATH), pd.read_csv(TEST_PATH)


def tabicl_fp(seed=SEED, device="cpu", **kw):
    from tabicl import TabICLRegressor
    kw.setdefault("norm_methods", ["none"])

    def fp(A, t, B):
        return TabICLRegressor(random_state=seed, device=device, **kw).fit(A, t).predict(B)
    return fp
