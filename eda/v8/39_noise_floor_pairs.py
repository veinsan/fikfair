"""39 - Is a public RMSE of ~2.3 attainable from the FEATURES at all? (model-free lower bound)
If two records have (almost) identical features but different targets, ANY function f(x) must miss at least one
of them: for a pair (x, y1), (x, y2), min over f of [(y1-f)^2 + (y2-f)^2]/2 = (y1-y2)^2/4.
So over pairs of near-duplicates the irreducible MSE is  E[(y1-y2)^2] / 4  (as the pair distance -> 0), i.e. half
the semivariance. We estimate it for pairs below a distance r and extrapolate r -> 0, in two metrics, with
bootstrap intervals. Compare sqrt(floor) with the public 2.28: if the floor is clearly above it, 2.28 cannot come
from a function of these features and must use information outside them."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from common import SEED, h, load, plt, save

tr, te, F = load()
X_all = np.vstack([tr[F].values, te[F].values])
y = tr.target.values
n = len(y)
Z = (X_all - X_all.mean(0)) / X_all.std(0)
ev, V = np.linalg.eigh(np.cov(Z.T))
Wh = Z @ V / np.sqrt(ev)
rng = np.random.default_rng(SEED)

rows = []
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for a, (name, S) in zip(ax, [("standardized features", Z[:n]), ("whitened (Mahalanobis)", Wh[:n])]):
    D = squareform(pdist(S))
    iu = np.triu_indices(n, 1)
    d, dy2 = D[iu], ((y[:, None] - y[None, :]) ** 2)[iu]
    order = np.argsort(d)
    h(f"{name}: closest pairs")
    for k in [5, 10, 20, 40, 80, 160]:
        idx = order[:k]
        floor = dy2[idx].mean() / 4
        boot = [dy2[rng.choice(idx, k)].mean() / 4 for _ in range(2000)]
        rows.append((name, k, d[idx].max(), np.sqrt(floor), np.sqrt(np.percentile(boot, 5)), np.sqrt(np.percentile(boot, 95))))
        print(f"  {k:3d} closest pairs (distance <= {d[idx].max():.3f}): implied RMSE floor = {np.sqrt(floor):.2f}"
              f"  (90% CI {np.sqrt(np.percentile(boot, 5)):.2f} - {np.sqrt(np.percentile(boot, 95)):.2f})")
    print("  the 8 closest pairs: distance | y1 | y2")
    for i in order[:8]:
        a_, b_ = iu[0][i], iu[1][i]
        print(f"    {d[i]:.3f} | {y[a_]:6.2f} | {y[b_]:6.2f}")
    # extrapolation of half-semivariance to d = 0 over the 160 closest pairs
    idx = order[:160]
    c = np.polyfit(d[idx], dy2[idx] / 4, 1)
    print(f"  linear extrapolation of the floor to distance 0: RMSE ~ {np.sqrt(max(c[1], 0)):.2f}")
    a.scatter(d[order[:300]], np.sqrt(dy2[order[:300]] / 4), s=8)
    a.axhline(2.28, c="r", ls="--", label="public top-1 RMSE 2.28")
    a.set_xlabel("pair distance"); a.set_ylabel("|y1 - y2| / 2 (per-pair irreducible error)"); a.set_title(name); a.legend()
save(fig, "v8_39_noise_floor.png")
pd.DataFrame(rows, columns=["metric", "k_pairs", "max_dist", "rmse_floor", "ci5", "ci95"]).to_csv(
    Path(__file__).resolve().parents[2] / "outputs" / "eda" / "v8_39_noise_floor.csv", index=False)
