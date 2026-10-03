"""12 - TabDPT (Layer6, HF Layer6/TabDPT, open weights): a tabular foundation model pretrained on REAL datasets,
unlike TabICL/TabPFN (synthetic priors). Hypothesis: different prior -> less correlated errors -> ensemble gain.
Same 5x3 folds as v1; OOF + test predictions appended to outputs/eda/v2_oof.csv / v2_test.csv."""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from tabdpt import TabDPTRegressor

from common import OUT, SEED, h, load

warnings.filterwarnings("ignore")
tr, te, F = load()
X, y, Xte = tr[F].values, tr.target.values, te[F].values
cv = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X))

CONFIGS = {
    "TabDPT": dict(normalizer="standard"),
    "TabDPT(qnorm)": dict(normalizer="quantile-normal"),
}


def fit_predict(kw, A, t, B):
    m = TabDPTRegressor(device="cpu", compile=False, use_flash=False, verbose=False, **kw)
    m.fit(A, t)  # returns None, not self
    return m.predict(B, n_ensembles=8, seed=SEED)


oof_p, te_p = OUT / "v2_oof.csv", OUT / "v2_test.csv"
oof_df = pd.read_csv(oof_p) if oof_p.exists() else pd.DataFrame({"record_id": tr.record_id, "target": y})
te_df = pd.read_csv(te_p) if te_p.exists() else pd.DataFrame({"record_id": te.record_id})

h("TabDPT on the shared 5x3 folds")
v1 = pd.read_csv(OUT / "07_oof.csv")
for name, kw in CONFIGS.items():
    t = time.time()
    oof = np.zeros((3, len(y)))
    for i, (a, b) in enumerate(cv):
        oof[i // 5, b] = fit_predict(kw, X[a], y[a], X[b])
    score = np.mean([np.sqrt(np.mean((y - o) ** 2)) for o in oof])
    oof_df[name] = oof.mean(0)
    te_df[name] = fit_predict(kw, X, y, Xte)
    r_icl = np.corrcoef(y - oof_df[name], y - v1["TabICLv2"])[0, 1]
    r_svr = np.corrcoef(y - oof_df[name], y - v1["SVR-multiPCA"])[0, 1]
    print(f"  {name:14s} RMSE={score:.4f}  resid-corr vs TabICL={r_icl:.3f}  vs SVR={r_svr:.3f}  ({time.time() - t:.0f}s)",
          flush=True)
    oof_df.to_csv(oof_p, index=False)
    te_df.to_csv(te_p, index=False)
