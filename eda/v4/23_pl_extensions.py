"""23 - Last levers on top of v3 (TabICL-PL 4-split + SVR), random 5x3 folds (the scheme that matches test, see 22).
A. Transductive SVR: the SVR (the blend member with the most different errors) also gets the pseudo-labelled
   query rows in its training set (same 4-split rule: a row never sees its own pseudo label).
   Risk: its pseudo labels come from TabICL, so the SVR may become MORE correlated with TabICL -> less blend value.
B. Two PL rounds: refine pseudo labels with the round-1 PL output, then predict again.
2 random states for TabICL (relative comparison with the 2-seed OOF from 21)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from tabicl import TabICLRegressor

SEEDS = (0, 1)
mu, sd = X_all.mean(0), X_all.std(0)
PCAS = {k: PCA(k, random_state=SEED).fit((X_all - mu) / sd) for k in (10, 12, 14, 16, 20)}
PSD = {k: PCAS[k].transform((X_all - mu) / sd).std(0) for k in PCAS}


def bag(A, t, B):
    return np.mean([TabICLRegressor(random_state=s, norm_methods=["none"]).fit(A, t).predict(B) for s in SEEDS], 0)


def svr_multi(A, t, B):
    lt = np.log(t - 30)
    m_, s_ = lt.mean(), lt.std()
    return np.mean([np.exp(SVR(C=3, gamma=1 / k, epsilon=0.05).fit(
        p.transform((A - mu) / sd) / PSD[k], (lt - m_) / s_).predict(p.transform((B - mu) / sd) / PSD[k]) * s_ + m_) + 30
        for k, p in PCAS.items()], 0)


def cross_parts(model, A, t, B, labels, n_split=4):
    out = np.empty(len(B))
    part = np.arange(len(B)) % n_split
    for k in range(n_split):
        q, c = part == k, part != k
        out[q] = model(np.vstack([A, B[c]]), np.r_[t, labels[c]], B[q])
    return out


o = {k: np.zeros((3, N)) for k in ["SVR", "SVR-PL", "ICL-PL r1", "ICL-PL r2"]}
h("RUNNING 15 folds")
for i, (a, b) in enumerate(FOLDS):
    A, t, B = X[a], y[a], X[b]
    first = bag(A, t, B)
    r1 = cross_parts(bag, A, t, B, first)
    o["ICL-PL r1"][i // 5, b] = r1
    o["ICL-PL r2"][i // 5, b] = cross_parts(bag, A, t, B, r1)
    o["SVR"][i // 5, b] = svr_multi(A, t, B)
    o["SVR-PL"][i // 5, b] = cross_parts(svr_multi, A, t, B, first)
    print(f"  fold {i + 1}/15 done", flush=True)

h("SINGLE MODELS (RMSE averaged over repeats)")
for k, v in o.items():
    print(f"  {k:10s} RMSE={np.mean([rmse(y, r) for r in v]):.4f}")
bag_oof = {k: v.mean(0) for k, v in o.items()}
print("  residual corr SVR vs ICL-PL r1   :", round(np.corrcoef(y - bag_oof["SVR"], y - bag_oof["ICL-PL r1"])[0, 1], 3))
print("  residual corr SVR-PL vs ICL-PL r1:", round(np.corrcoef(y - bag_oof["SVR-PL"], y - bag_oof["ICL-PL r1"])[0, 1], 3))


def nested(P):
    out = []
    for s in [42, 7, 123, 2024, 99]:
        pr = np.zeros(N)
        for a, b in KFold(10, shuffle=True, random_state=s).split(P):
            w, _ = nnls(P[a], y[a])
            pr[b] = P[b] @ (w / w.sum())
        out.append(rmse(y, pr))
    return np.mean(out)


h("BLENDS (nested NNLS, 5 outer seeds)")
rows = []
for name, cols in [("v3: ICL-PL r1 + SVR", ["ICL-PL r1", "SVR"]),
                   ("ICL-PL r1 + SVR-PL", ["ICL-PL r1", "SVR-PL"]),
                   ("ICL-PL r1 + SVR + SVR-PL", ["ICL-PL r1", "SVR", "SVR-PL"]),
                   ("ICL-PL r2 + SVR", ["ICL-PL r2", "SVR"])]:
    r = nested(np.c_[tuple(bag_oof[c] for c in cols)])
    rows.append((name, r))
    print(f"  {name:28s} nested={r:.4f}")
pd.DataFrame(rows, columns=["blend", "nested"]).to_csv(OUT / "v4_23_extensions.csv", index=False)
pd.DataFrame(bag_oof).assign(record_id=pd.read_csv(OUT / "v3_oof.csv").record_id, target=y).to_csv(
    OUT / "v4_23_oof.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.barh([r[0] for r in rows], [r[1] for r in rows]); ax.invert_yaxis()
ax.set_xlim(min(r[1] for r in rows) - 0.05, max(r[1] for r in rows) + 0.03); ax.set_title("nested blend RMSE")
save(fig, "v4_23_extensions.png")
