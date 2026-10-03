"""41 - 'Twin copy': for rows whose nearest labelled neighbour is extremely close (Mahalanobis metric, where the
closest pairs in 39 had near-identical targets), is copying the twin's target better than the blend?
Evaluated out-of-fold: twin = nearest TRAINING-fold row; blend = v6b OOF (per repeat). Rule: replace or mix the
blend with the twin target when the twin distance is below a threshold. Also counts how many TEST rows have such
a twin among the 382 training rows (the only place the rule could act)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, Xte, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors

mu, sd = X_all.mean(0), X_all.std(0)
Z = (X_all - mu) / sd
ev, V = np.linalg.eigh(np.cov(Z.T))
WH = lambda A: ((A - mu) / sd) @ V / np.sqrt(ev)
L = np.load(OUT / "v7_members.npz")
FM, S = L["FM"], L["SVR-plateau (v6b)"]


def nested(F, S_):
    pr = np.zeros(N)
    for a, b in KFold(10, shuffle=True, random_state=SEED).split(F):
        w, _ = nnls(np.c_[F[a], S_[a]], y[a])
        pr[b] = np.c_[F[b], S_[b]] @ (w / w.sum())
    return pr


blend = np.stack([nested(FM[r], S[r]) for r in range(3)])
dist, twin_y = np.zeros((3, N)), np.zeros((3, N))
for i, (a, b) in enumerate(FOLDS):
    d, j = NearestNeighbors(n_neighbors=1).fit(WH(X[a])).kneighbors(WH(X[b]))
    dist[i // 5, b], twin_y[i // 5, b] = d[:, 0], y[a][j[:, 0]]

h("Error of blend vs twin target, by twin distance (all repeats stacked)")
D, B, T, Y = dist.ravel(), blend.ravel(), twin_y.ravel(), np.tile(y, 3)
for lo, hi in [(0, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 99)]:
    m = (D >= lo) & (D < hi)
    if m.sum():
        print(f"  twin dist [{lo},{hi}): n={m.sum():4d}  blend RMSE={np.sqrt(np.mean((Y[m] - B[m]) ** 2)):6.2f}  "
              f"twin RMSE={np.sqrt(np.mean((Y[m] - T[m]) ** 2)):6.2f}  50/50 RMSE={np.sqrt(np.mean((Y[m] - (B[m] + T[m]) / 2) ** 2)):6.2f}")

h("Rule: use 50/50 blend+twin when twin distance < thr (per-repeat RMSE over all rows)")
base = np.mean([rmse(y, blend[r]) for r in range(3)])
print(f"  no rule: {base:.4f}")
for thr in [0.8, 0.9, 1.0, 1.1, 1.2]:
    adj = np.where(dist < thr, 0.5 * (blend + twin_y), blend)
    print(f"  thr={thr}: {np.mean([rmse(y, adj[r]) for r in range(3)]):.4f}  rows affected per repeat: {(dist < thr).sum() / 3:.0f}")

dt = NearestNeighbors(n_neighbors=1).fit(WH(X)).kneighbors(WH(Xte))[0][:, 0]
print("\n  TEST rows with a training twin closer than 0.8 / 1.0 / 1.2:", [(dt < t).sum() for t in (0.8, 1.0, 1.2)])
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(D, np.abs(Y - T), s=4, alpha=0.4, label="|y - twin y|"); ax.scatter(D, np.abs(Y - B), s=4, alpha=0.4, label="|y - blend|")
ax.set_xlim(0.6, 2.0); ax.set_xlabel("Mahalanobis distance to nearest training row"); ax.legend(); ax.set_title("twin copy vs blend")
save(fig, "v8_41_twin.png")
