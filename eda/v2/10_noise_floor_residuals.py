"""10 - Are we 'mentok'? Two mathematical tests.
(a) Semivariogram gamma(d) = 1/2 E[(y_i - y_j)^2] over pairs at feature distance d. Its limit at d->0 (the
    'nugget') is the variance a perfect smoother could never remove -> irreducible RMSE ~ sqrt(nugget).
    The gap between that floor and our CV RMSE is what better modelling could still win.
(b) Residual learnability: fit several models on the v1 blend OOF residuals with CV. If CV R2 ~ 0 no model
    family finds leftover structure in these features.
(c) Calibration: E[y | pred] vs pred (binned by PREDICTION, the fair check; bias by TRUE bin is always
    regression-to-the-mean and says nothing about fixable miscalibration)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import OUT, SEED, h, load, plt, save

tr, te, F = load()
n = len(tr)
X_all = np.vstack([tr[F].values, te[F].values])
Z_all = (X_all - X_all.mean(0)) / X_all.std(0)
y = tr.target.values
oof = pd.read_csv(OUT / "07_oof.csv")
assert (oof.record_id.values == tr.record_id.values).all()
blend = 0.72 * oof["TabICLv2"].values + 0.28 * oof["SVR-multiPCA"].values
res = y - blend
rmse = lambda r: float(np.sqrt(np.mean(r ** 2)))
print(f"v1 blend OOF RMSE = {rmse(res):.4f}")

h("(a) SEMIVARIOGRAM + NUGGET")
spaces = {"standardized 24-D": Z_all[:n]}
P = PCA(12, random_state=SEED).fit(Z_all).transform(Z_all)
spaces["whitened PCA-12"] = (P / P.std(0))[:n]
dy2 = 0.5 * pdist(y[:, None], "sqeuclidean")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for a, (name, S) in zip(ax, spaces.items()):
    d = pdist(S)
    edges = np.quantile(d, np.linspace(0, 0.25, 26))
    mid = 0.5 * (edges[1:] + edges[:-1])
    g = np.array([dy2[(d >= lo) & (d < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    cnt = np.array([((d >= lo) & (d < hi)).sum() for lo, hi in zip(edges[:-1], edges[1:])])
    # nugget: weighted linear fit on the closest 8 bins, extrapolated to d = 0
    c = np.polyfit(mid[:8], g[:8], 1, w=np.sqrt(cnt[:8]))
    nugget = max(c[1], 0)
    print(f"{name:18s} gamma(first bin, d~{mid[0]:.2f}) = {g[0]:.1f} -> RMSE-equiv {np.sqrt(g[0]):.2f} | "
          f"nugget (d->0) = {nugget:.1f} -> irreducible RMSE ~ {np.sqrt(nugget):.2f} | sill (all pairs) = {dy2.mean():.1f}")
    a.plot(mid, g, "o-", label="empirical gamma(d)")
    xx = np.linspace(0, mid[7], 20)
    a.plot(xx, np.polyval(c, xx), "r--", label=f"extrapolation, nugget={nugget:.1f}")
    a.axhline(rmse(res) ** 2, c="k", ls=":", label=f"v1 blend MSE={rmse(res) ** 2:.1f}")
    a.axhline(dy2.mean(), c="gray", ls=":", label="sill (var y)")
    a.set_xlabel("pair distance"); a.set_ylabel("semivariance of target"); a.set_title(name); a.legend(fontsize=8)
save(fig, "v2_10_variogram.png")

h("(b) CAN ANY MODEL PREDICT THE RESIDUAL?  (10-fold CV R2 on OOF residuals)")
Xr = tr[F].values
Xr_aug = np.hstack([Xr, blend[:, None]])  # also give it the prediction itself (catches miscalibration)
cv = KFold(10, shuffle=True, random_state=SEED)
learners = {
    "Ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(np.logspace(-2, 4, 30))),
    "kNN(15)": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(15)),
    "CatBoost(shallow)": lambda: CatBoostRegressor(iterations=400, learning_rate=0.02, depth=3, l2_leaf_reg=10,
                                                    verbose=0, random_seed=SEED, allow_writing_files=False),
}
for name, make in learners.items():
    for tag, A in [("features", Xr), ("features+pred", Xr_aug)]:
        p = cross_val_predict(make(), A, res, cv=cv)
        r2 = 1 - np.mean((res - p) ** 2) / np.mean((res - res.mean()) ** 2)
        print(f"  {name:18s} on {tag:14s} CV R2 = {r2:+.4f}  -> blend RMSE if added: {rmse(res - p):.4f}")

h("(c) CALIBRATION: E[y | prediction bin]")
df = pd.DataFrame({"pred": blend, "y": y})
df["pbin"] = pd.qcut(df.pred, 10)
cal = df.groupby("pbin", observed=True).agg(pred=("pred", "mean"), y=("y", "mean"), n=("y", "size"))
cal["gap"] = cal.y - cal.pred
print(cal.round(2).to_string())

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(cal.pred, cal.y, "o-"); ax[0].plot([35, 80], [35, 80], "r--")
ax[0].set_xlabel("mean prediction in bin"); ax[0].set_ylabel("mean target in bin"); ax[0].set_title("calibration by prediction decile")
ax[1].scatter(blend, res, s=8); ax[1].axhline(0, c="r"); ax[1].set_xlabel("OOF prediction"); ax[1].set_ylabel("residual")
ax[1].set_title("residual vs prediction")
save(fig, "v2_10_calibration.png")
