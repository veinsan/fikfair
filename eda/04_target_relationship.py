"""04 - How does the target depend on features? Linear vs non-linear, interactions, target bounds.
Target is a 'constructed composite score' -> often a (clipped/rescaled) weighted sum of indicators,
so a near-linear signal in the latent space is plausible. We test it instead of assuming it."""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

from common import SEED, h, load, plt, save

tr, te, F = load()
X, y = tr[F].values, tr.target.values
cv = KFold(10, shuffle=True, random_state=SEED)
rmse = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))

h("PER-FEATURE: linear R2 vs spline R2 vs mutual information")
mi = mutual_info_regression(X, y, random_state=SEED)
rows = []
for j, c in enumerate(F):
    xj = X[:, [j]]
    lin = cross_val_predict(LinearRegression(), xj, y, cv=cv)
    spl = cross_val_predict(make_pipeline(SplineTransformer(n_knots=5), RidgeCV()), xj, y, cv=cv)
    rows.append((c, 1 - rmse(lin) ** 2 / y.var(), 1 - rmse(spl) ** 2 / y.var(), mi[j]))
per = pd.DataFrame(rows, columns=["feat", "R2_lin", "R2_spline", "MI"]).set_index("feat")
per["nonlin_gain"] = per.R2_spline - per.R2_lin
print(per.sort_values("R2_lin", ascending=False).round(3).to_string())

fig, axes = plt.subplots(4, 6, figsize=(20, 12))
for a, c in zip(axes.ravel(), F):
    a.scatter(tr[c], y, s=5, alpha=0.5)
    lw = lowess(y, tr[c], frac=0.4)
    a.plot(lw[:, 0], lw[:, 1], c="r")
    a.set_title(f"{c} R2lin={per.loc[c, 'R2_lin']:.2f}", fontsize=9)
save(fig, "04_feature_vs_target.png")

h("GLOBAL: linear vs additive-spline vs pairwise interactions (10-fold CV RMSE)")
models = {
    "OLS": make_pipeline(StandardScaler(), LinearRegression()),
    "RidgeCV": make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 3, 30))),
    "Additive splines + Ridge": make_pipeline(SplineTransformer(n_knots=5), StandardScaler(), RidgeCV(np.logspace(-2, 4, 30))),
    "Poly2 (interactions) + Ridge": make_pipeline(StandardScaler(), PolynomialFeatures(2), StandardScaler(), RidgeCV(np.logspace(-1, 5, 30))),
}
oof = {}
for n, m in models.items():
    oof[n] = cross_val_predict(m, X, y, cv=cv)
    print(f"  {n:30s} RMSE={rmse(oof[n]):.4f}")
print(f"  {'constant mean':30s} RMSE={y.std():.4f}")

h("LINEAR RESIDUAL DIAGNOSTICS")
res = y - oof["RidgeCV"]
print("residual vs fitted spearman:", round(pd.Series(res).corr(pd.Series(oof["RidgeCV"]), method="spearman"), 3))
print("mean residual by target decile:")
dec = pd.qcut(y, 10, labels=False)
print(pd.DataFrame({"dec": dec, "res": res, "y": y}).groupby("dec").agg(y_mean=("y", "mean"), res_mean=("res", "mean"), res_sd=("res", "std")).round(2).to_string())
print("OOF prediction range:", oof["RidgeCV"].min().round(2), oof["RidgeCV"].max().round(2),
      "| target range:", y.min().round(2), y.max().round(2))
print("#OOF preds > 100.44 (train max):", int((oof["RidgeCV"] > y.max()).sum()))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].scatter(oof["RidgeCV"], y, s=8); ax[0].plot([30, 105], [30, 105], "r--")
ax[0].set_xlabel("OOF pred (Ridge)"); ax[0].set_ylabel("target"); ax[0].set_title("Ridge OOF: pred vs true")
ax[1].scatter(oof["RidgeCV"], res, s=8); ax[1].axhline(0, c="r")
lw = lowess(res, oof["RidgeCV"], frac=0.3); ax[1].plot(lw[:, 0], lw[:, 1], c="orange", lw=2)
ax[1].set_title("residual vs fitted (curvature = non-linear link)")
ax[2].hist(res, bins=40); ax[2].set_title(f"residuals (sd={res.std():.2f})")
save(fig, "04_linear_residuals.png")
