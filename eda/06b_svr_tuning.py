"""06b - SVR is the best single model so far. Sweep PCA dim x C x gamma (log target) on the 5x3 folds.
Picking the best cell of a grid on the same CV is mildly optimistic, so we look for a FLAT region, not the argmin."""
import itertools

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import OUT, SEED, h, load, plt, save

tr, te, F = load()
X, y = tr[F].values, tr.target.values
Z_all = StandardScaler().fit_transform(np.vstack([X, te[F].values]))
cv = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X))
ly = np.log(y - 30)

rows = []
for k in [8, 10, 12, 14, 16, 20, 24]:
    P = PCA(k, random_state=SEED).fit(Z_all).transform(Z_all)[: len(X)]
    P = P / P.std(0)
    for C, g, eps in itertools.product([1, 3, 10, 30], [0.5, 1, 2], [0.05, 0.1]):
        gamma = g / k  # 'scale'-like: gamma relative to dimensionality
        err = []
        for tri, vai in cv:
            mu, sd = ly[tri].mean(), ly[tri].std()
            m = SVR(C=C, gamma=gamma, epsilon=eps).fit(P[tri], (ly[tri] - mu) / sd)
            p = np.exp(m.predict(P[vai]) * sd + mu) + 30
            err.append(np.mean((y[vai] - p) ** 2))
        rows.append((k, C, g, eps, np.sqrt(np.mean(err))))
res = pd.DataFrame(rows, columns=["pca_k", "C", "g", "eps", "rmse"])
res.to_csv(OUT / "06b_svr_grid.csv", index=False)

h("BEST 10 CONFIGS")
print(res.nsmallest(10, "rmse").round(4).to_string(index=False))
h("BEST RMSE PER PCA DIM (flatness check)")
print(res.groupby("pca_k").rmse.min().round(4).to_string())

piv = res[res.eps == 0.1].groupby(["pca_k", "C"]).rmse.min().unstack()
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(piv.values, cmap="viridis_r")
ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
ax.set_xlabel("C"); ax.set_ylabel("PCA k"); fig.colorbar(im, label="CV RMSE")
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="w")
ax.set_title("SVR(log y) CV RMSE, best gamma per cell")
save(fig, "06b_svr_grid.png")
