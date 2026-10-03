"""03 - Correlation structure + latent dimension.
Rules say features = private linear projection of a few climate indicators (+ perturbation + rounding).
If so, the 24 features live near a k-dim subspace (k << 24): PCA eigenvalues should drop sharply after k."""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

from common import h, load, plt, save

tr, te, F = load()
X_all = pd.concat([tr[F], te[F]], ignore_index=True)
y = tr.target

h("FEATURE-TARGET CORRELATION")
ct = pd.DataFrame({
    "pearson": tr[F].corrwith(y),
    "spearman": tr[F].corrwith(y, method="spearman"),
})
ct["|pearson|"] = ct.pearson.abs()
print(ct.sort_values("|pearson|", ascending=False).round(3).to_string())

h("FEATURE-FEATURE CORRELATION")
C = X_all.corr()
off = C.where(~np.eye(len(F), dtype=bool)).abs().stack()
print("mean |r| off-diag:", round(off.mean(), 3), "| max:", round(off.max(), 3))
print("pairs |r|>0.9:", (off > 0.9).sum() // 2, "| |r|>0.7:", (off > 0.7).sum() // 2)
print("top 10 pairs:\n", off.sort_values(ascending=False)[::2].head(10).round(3).to_string())

link = hierarchy.linkage(1 - C.abs().values[np.triu_indices(len(F), 1)], "average")
order = hierarchy.leaves_list(link)
fig, ax = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [4, 1]})
sns.heatmap(C.iloc[order, order], cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax[0],
            xticklabels=[F[i][-2:] for i in order], yticklabels=[F[i][-2:] for i in order])
ax[0].set_title("feature correlation (clustered, train+test)")
ct.pearson.iloc[order].plot.barh(ax=ax[1], color=np.where(ct.pearson.iloc[order] > 0, "tab:red", "tab:blue"))
ax[1].set_title("corr with target"); ax[1].invert_yaxis()
save(fig, "03_corr_matrix.png")

h("PCA SPECTRUM (standardized, train+test)")
Z = (X_all - X_all.mean()) / X_all.std()
ev, evec = np.linalg.eigh(np.cov(Z.values.T))
ev, evec = ev[::-1], evec[:, ::-1]
cum = ev.cumsum() / ev.sum()
for k in range(12):
    print(f"PC{k+1:2d} eig={ev[k]:7.3f}  var%={100*ev[k]/ev.sum():6.2f}  cum%={100*cum[k]:6.2f}")
print("smallest 6 eigenvalues:", np.round(ev[-6:], 4))
print("condition number:", round(ev[0] / ev[-1], 1))

# Marchenko-Pastur upper edge: eigenvalues of pure-noise correlation matrix stay below this
q = len(F) / len(X_all)
print(f"Marchenko-Pastur noise edge (1+sqrt(q))^2 = {(1 + np.sqrt(q))**2:.3f}  -> "
      f"#eig above edge = {(ev > (1 + np.sqrt(q))**2).sum()}")

# Does the target live in the top PCs?
pcs = Z.values[: len(tr)] @ evec
r_pc = [stats.pearsonr(pcs[:, k], y)[0] for k in range(len(F))]
print("\ncorr(PC_k, target):", np.round(r_pc, 3))
cv = KFold(10, shuffle=True, random_state=42)
print("\nLinear regression CV-RMSE on top-k PCs (tells how many directions carry signal):")
for k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]:
    s = -cross_val_score(LinearRegression(), pcs[:, :k], y, cv=cv, scoring="neg_root_mean_squared_error").mean()
    print(f"  k={k:2d}  RMSE={s:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(17, 4))
ax[0].bar(range(1, len(F) + 1), ev); ax[0].axhline((1 + np.sqrt(q))**2, ls="--", c="r", label="MP noise edge")
ax[0].set_yscale("log"); ax[0].set_title("PCA eigenvalues (log)"); ax[0].legend()
ax[1].plot(range(1, len(F) + 1), cum, "o-"); ax[1].set_title("cumulative explained variance")
ax[2].bar(range(1, len(F) + 1), r_pc, color="tab:green"); ax[2].set_title("corr(PC_k, target)")
save(fig, "03_pca_spectrum.png")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for i, (a, b) in enumerate([(0, 1), (0, 2), (1, 2)]):
    s = ax[i].scatter(pcs[:, a], pcs[:, b], c=y, cmap="viridis", s=12)
    te_pc = Z.values[len(tr):] @ evec
    ax[i].scatter(te_pc[:, a], te_pc[:, b], marker="x", c="red", s=12, label="test")
    ax[i].set_xlabel(f"PC{a+1}"); ax[i].set_ylabel(f"PC{b+1}")
ax[0].legend(); fig.colorbar(s, ax=ax[-1], label="target")
save(fig, "03_pc_scatter.png")
