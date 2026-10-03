"""28 - Local-context in-context learning (retrieval), a different way to use the same TabICL.
Theory (LoCalPFN-style): an in-context learner attends over the whole training set; when the target varies on a
short length scale (v2 variogram: semivariance rises fast with distance), restricting each query's context to its
k nearest labelled rows can sharpen the prediction. Risk: with only ~300 labelled rows the global context is
already small, and removing rows throws away the 'typicality' information carried by far-away points.
Per query row: context = its k nearest training rows (standardized features). k = N is the usual global model.
Single random state, TabICL norm=none, same 5x3 folds (first repeat for the sweep, all 15 folds for the best k)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tabicl import TabICLRegressor

mu, sd = X_all.mean(0), X_all.std(0)


def local_predict(A, t, B, k):
    if k >= len(A):
        return TabICLRegressor(random_state=SEED, norm_methods=["none"]).fit(A, t).predict(B)
    idx = NearestNeighbors(n_neighbors=k).fit((A - mu) / sd).kneighbors((B - mu) / sd)[1]
    out = np.empty(len(B))
    for j in range(len(B)):
        out[j] = TabICLRegressor(random_state=SEED, norm_methods=["none"]).fit(A[idx[j]], t[idx[j]]).predict(B[j:j + 1])[0]
    return out


h("SWEEP over k on repeat 1 (5 folds)")
rows, oofs = [], {}
for k in [10_000, 200, 100]:
    t0 = time.time()
    o = np.zeros(N)
    for a, b in FOLDS[:5]:
        o[b] = local_predict(X[a], y[a], X[b], k)
    oofs[k] = o
    fold = [rmse(y[b], o[b]) for a, b in FOLDS[:5]]
    rows.append((k if k < 10_000 else "all", rmse(y, o), np.round(fold, 3)))
    print(f"  k={rows[-1][0]!s:>4}  RMSE={rmse(y, o):.4f}  folds={np.round(fold, 3)}  [{time.time() - t0:.0f}s]", flush=True)

g = oofs[10_000]
h("DIVERSITY: residual correlation of local vs global, and 50:50 average")
for k in [200, 100]:
    rc = np.corrcoef(y - g, y - oofs[k])[0, 1]
    print(f"  k={k:3d}  resid-corr vs global={rc:.3f}  RMSE(0.5 global + 0.5 local)={rmse(y, 0.5 * (g + oofs[k])):.4f}"
          f"  (global alone {rmse(y, g):.4f})")

pd.DataFrame({f"k={k}": v for k, v in oofs.items()}).assign(target=y).to_csv(OUT / "v6_28_local_oof_rep1.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 4))
ks = [r[0] for r in rows]
ax.plot(range(len(ks)), [r[1] for r in rows], "o-"); ax.set_xticks(range(len(ks)), [str(k) for k in ks])
ax.set_xlabel("context size k (nearest labelled rows)"); ax.set_ylabel("RMSE, repeat 1"); ax.set_title("local-context TabICL")
save(fig, "v6_28_local_context.png")
