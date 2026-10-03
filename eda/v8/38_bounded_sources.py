"""38 - Do bounded source indicators exist inside the feature space? (projection pursuit for MINIMUM kurtosis)
Pipeline: indicators -> rank -> bounded transform -> private linear projection -> standardize -> perturb -> round.
If the projection mixes k <= 24 sources with an invertible matrix, there are k directions w (in whitened space)
whose projection w.z is one of those bounded sources: light-tailed, excess kurtosis well below 0 (uniform = -1.2),
and, because ranks were taken over the same records, all sources share (nearly) the same marginal distribution.
If k > 24 sources were squeezed into 24 dimensions, every direction is a mix of many sources -> near-Gaussian
(kurtosis ~ 0) and the target cannot be reconstructed exactly from the features.
Method: gradient-free + gradient search on the unit sphere of the whitened 24-D space, 400 random starts,
deflation-free (we just record the distribution of local minima)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import stats
from scipy.optimize import minimize

from common import SEED, h, load, plt, save

tr, te, F = load()
X_all = np.vstack([tr[F].values, te[F].values])
y = tr.target.values
n = len(y)
Z = (X_all - X_all.mean(0)) / X_all.std(0)
ev, V = np.linalg.eigh(np.cov(Z.T))
Wh = Z @ V / np.sqrt(ev)  # whitened, 478 x 24


def kurt(w):
    w = w / np.linalg.norm(w)
    s = Wh @ w
    return stats.kurtosis(s)


rng = np.random.default_rng(SEED)
mins = []
for _ in range(400):
    w0 = rng.standard_normal(24)
    r = minimize(kurt, w0, method="L-BFGS-B")
    w = r.x / np.linalg.norm(r.x)
    mins.append((r.fun, w))
vals = np.array([m[0] for m in mins])
h("LOCAL MINIMA OF EXCESS KURTOSIS (whitened space)")
print("  quantiles of local minima:", np.round(np.quantile(vals, [0, 0.05, 0.25, 0.5, 0.75]), 3))
print("  reference: uniform -1.2, triangular -0.6, Gaussian 0; a 50/50 mix of 2 uniforms ~ -0.6")
# distinct directions among the best minima
best = sorted(mins, key=lambda m: m[0])
uniq = []
for v, w in best:
    if all(abs(w @ u) < 0.95 for _, u in uniq):
        uniq.append((v, w))
    if len(uniq) == 8:
        break
print("  best distinct directions (kurtosis, |corr with y|, Spearman with y):")
for v, w in uniq:
    s = Wh[:n] @ w
    print(f"    kurt={v:+.3f}  |r|={abs(np.corrcoef(s, y)[0, 1]):.3f}  rho={stats.spearmanr(s, y)[0]:+.3f}")

h("NULL: same search on 24-D data with the same covariance but GAUSSIAN rows (what pure mixing looks like)")
G = rng.standard_normal(Wh.shape)
nullmins = []
for _ in range(100):
    r = minimize(lambda w: stats.kurtosis(G @ (w / np.linalg.norm(w))), rng.standard_normal(24), method="L-BFGS-B")
    nullmins.append(r.fun)
print("  quantiles of local minima under Gaussian null:", np.round(np.quantile(nullmins, [0, 0.05, 0.5]), 3))

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].hist(vals, bins=40, alpha=0.7, label="real features"); ax[0].hist(nullmins, bins=20, alpha=0.7, label="Gaussian null")
ax[0].axvline(-1.2, c="r", ls="--", label="uniform source"); ax[0].legend(); ax[0].set_title("local minima of kurtosis")
s = Wh @ uniq[0][1]
ax[1].hist(s, bins=40); ax[1].set_title(f"most sub-Gaussian direction (kurt {uniq[0][0]:.2f})")
ax[2].scatter(s[:n], y, s=6); ax[2].set_title("target vs that direction")
save(fig, "v8_38_bounded_sources.png")
