"""15 - 'Views' of the feature space (tabular analogue of image augmentation).
Theory: the 24 features are an ARBITRARY private linear projection. TabICL is not rotation invariant (it embeds
each column separately, then attends across columns), so the same data presented in a different linear basis is
a genuinely different input. Different bases -> partly different errors -> ensemble gain (like TTA in CV).
Views (all unsupervised, fitted on train+test features, never the target):
  raw | PCA-24 scores | whitened PCA-24 | ICA-24 sources | random orthogonal rotations (x4)
Checks: CV RMSE of each view, residual correlation vs raw, and CV of raw+view averages."""
from cvlib import FOLDS, N, OUT, SEED, X, X_all, h, plt, rmse, run_cv, save, tabicl_fp, y
import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from sklearn.decomposition import FastICA

mu, sd = X_all.mean(0), X_all.std(0)
Z_all = (X_all - mu) / sd
ev, V = np.linalg.eigh(np.cov(Z_all.T))
ev, V = ev[::-1], V[:, ::-1]
n = N

VIEWS = {
    "raw": X_all,
    "pca24": Z_all @ V,
    "pca24_white": Z_all @ V / np.sqrt(ev),
    "ica24": FastICA(24, whiten="unit-variance", max_iter=20000, tol=1e-6, random_state=0).fit_transform(X_all),
}
for r in range(4):
    VIEWS[f"rot{r}"] = Z_all @ ortho_group.rvs(24, random_state=100 + r)

h("VISUAL CHECK: marginal distributions per view (first 6 columns)")
fig, axes = plt.subplots(len(VIEWS), 6, figsize=(18, 2.2 * len(VIEWS)))
for i, (vn, V_) in enumerate(VIEWS.items()):
    for j in range(6):
        axes[i, j].hist(V_[:, j], bins=30)
        axes[i, j].set_title(f"{vn} c{j}", fontsize=8)
        axes[i, j].tick_params(labelsize=6)
save(fig, "v3_15_view_marginals.png")

h("TabICLv2(norm=none) PER VIEW (5x3 KFold)")
oofs, rows = {}, []
for vn, V_ in VIEWS.items():
    s, fold, oofs[vn] = run_cv(tabicl_fp(), V_[:n], V_[n:], name=f"ICL[{vn}]")
    rows.append((vn, s))

h("RESIDUAL CORRELATION vs RAW VIEW + PAIRWISE BLEND raw/view (50:50)")
base = oofs["raw"]
for vn, o in oofs.items():
    if vn == "raw":
        continue
    rc = np.corrcoef(y - base, y - o)[0, 1]
    print(f"  {vn:12s} resid-corr={rc:.3f}  RMSE(view)={rmse(y, o):.4f}  RMSE(0.5 raw + 0.5 view)={rmse(y, (base + o) / 2):.4f}"
          f"   (raw alone {rmse(y, base):.4f})")
rots = [oofs[f"rot{r}"] for r in range(4)]
print(f"\n  average of 4 random rotations          RMSE={rmse(y, np.mean(rots, 0)):.4f}")
print(f"  raw + 4 rotations (equal weight)       RMSE={rmse(y, np.mean([base] + rots, 0)):.4f}")
allv = np.mean(list(oofs.values()), 0)
print(f"  all views averaged                      RMSE={rmse(y, allv):.4f}")

res = pd.DataFrame(rows, columns=["view", "rmse"])
res.to_csv(OUT / "v3_15_views.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].barh(res.view, res.rmse); ax[0].invert_yaxis(); ax[0].set_xlim(res.rmse.min() - 0.2, res.rmse.max() + 0.1)
ax[0].set_title("TabICL CV RMSE per view")
R = pd.DataFrame({k: y - v for k, v in oofs.items()}).corr()
im = ax[1].imshow(R.values, vmin=0.85, vmax=1, cmap="viridis"); fig.colorbar(im, ax=ax[1])
ax[1].set_xticks(range(len(R)), R.index, rotation=60, fontsize=7); ax[1].set_yticks(range(len(R)), R.index, fontsize=7)
ax[1].set_title("residual correlation between views")
save(fig, "v3_15_views.png")
