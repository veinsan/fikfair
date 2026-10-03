"""42 - Recover the bounded source indicators by deflation, then test whether the target is simple in them.
Re-reading 38: the most sub-Gaussian direction has mass piled near BOTH ends with a flat middle - the signature of
a bounded transform such as tanh(c*z) applied to a spread-out variable, not of two clusters. Several distinct
local minima (-1.49, -1.37, -1.29, ...) suggest several such sources.
Deflation: in the whitened 24-D space, find the min-kurtosis unit direction (many restarts), store it, project it
out, repeat until 24 directions. Then:
  - kurtosis profile of the 24 recovered directions vs a Gaussian null of the same size
  - do the bounded-looking sources share one marginal shape? (QQ of sorted standardized values)
  - CV of target models on the recovered sources: OLS, additive splines (GAM), TabICL, vs the same on raw features."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

Z = (X_all - X_all.mean(0)) / X_all.std(0)
ev, V = np.linalg.eigh(np.cov(Z.T))
Wh = Z @ V / np.sqrt(ev)
rng = np.random.default_rng(SEED)


def deflate(data, restarts=60):
    d = data.shape[1]
    B = np.eye(d)  # basis of the remaining subspace (columns)
    dirs, kurts = [], []
    for step in range(d):
        sub = data @ B
        m = sub.shape[1]
        if m == 1:
            w = np.ones(1)
        else:
            best = None
            for _ in range(restarts):
                r = minimize(lambda v: stats.kurtosis(sub @ (v / np.linalg.norm(v))), rng.standard_normal(m), method="L-BFGS-B")
                if best is None or r.fun < best.fun:
                    best = r
            w = best.x / np.linalg.norm(best.x)
        u = B @ w
        dirs.append(u)
        kurts.append(stats.kurtosis(data @ u))
        # remove u: new basis = orthonormal complement of w inside the current subspace
        Q, _ = np.linalg.qr(np.c_[w, rng.standard_normal((m, m - 1))])
        B = B @ Q[:, 1:]
    return np.array(dirs), np.array(kurts)


h("DEFLATION on real features")
D, K = deflate(Wh)
print("  kurtosis of recovered directions:", np.round(K, 2))
h("DEFLATION on a Gaussian null of the same shape")
_, Kn = deflate(rng.standard_normal(Wh.shape), restarts=20)
print("  kurtosis (null):                 ", np.round(Kn, 2))
n_bounded = int((K < Kn.min() - 0.05).sum())
print(f"  directions more sub-Gaussian than anything in the null: {n_bounded}")

S = Wh @ D.T  # recovered sources, 478 x 24
Sn = (S - S.mean(0)) / S.std(0)
h("Do the bounded-looking sources share one marginal shape? (max |difference| of sorted standardized values)")
top = np.argsort(K)[:6]
q = np.sort(Sn[:, top], axis=0)
for i in range(1, len(top)):
    print(f"  source {top[0]} vs {top[i]}: max |sorted diff| = {np.abs(q[:, 0] - q[:, i]).max():.3f}  "
          f"(or mirrored: {np.abs(q[:, 0] + q[::-1, i]).max():.3f})")

h("TARGET MODELS: recovered sources vs raw features (10-fold CV)")
cv = KFold(10, shuffle=True, random_state=SEED)
gam = lambda: make_pipeline(SplineTransformer(n_knots=6), RidgeCV(np.logspace(-2, 4, 30)))
r_lin = lambda: LinearRegression()
for name, A in [("raw features", Z[:N]), ("recovered sources (all 24)", Sn[:N]),
                (f"bounded sources only ({n_bounded})", Sn[:N, np.argsort(K)[:max(n_bounded, 1)]])]:
    print(f"  {name:32s} OLS={rmse(y, cross_val_predict(r_lin(), A, y, cv=cv)):.4f}  "
          f"GAM={rmse(y, cross_val_predict(gam(), A, y, cv=cv)):.4f}")
rho = [stats.spearmanr(Sn[:N, j], y)[0] for j in range(24)]
print("  spearman(source, y) sorted by source kurtosis:", np.round(np.array(rho)[np.argsort(K)], 2))
np.save(OUT / "v8_42_sources.npy", S)
np.save(OUT / "v8_42_dirs.npy", D)

fig, axes = plt.subplots(3, 6, figsize=(20, 9))
for a, j in zip(axes[0], np.argsort(K)[:6]):
    a.hist(Sn[:, j], bins=40); a.set_title(f"source kurt {K[j]:.2f}", fontsize=9)
for a, j in zip(axes[1], np.argsort(K)[:6]):
    a.scatter(Sn[:N, j], y, s=4); a.set_title(f"rho={rho[j]:.2f}", fontsize=9)
axes[2, 0].plot(np.sort(K), "o-", label="real"); axes[2, 0].plot(np.sort(Kn), "o-", label="Gaussian null")
axes[2, 0].legend(); axes[2, 0].set_title("deflation kurtosis profile")
for a in axes[2, 1:]:
    a.axis("off")
save(fig, "v8_42_deflation.png")
