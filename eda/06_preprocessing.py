"""06 - Preprocessing experiments, each one VERIFIED by CV (theory != evaluation).

Hypotheses from EDA 01-05:
  H1  features = W @ bounded(rank(indicators)) + noise  -> un-mix with ICA to recover source indicators
  H2  small-eigenvalue PCs = perturbation noise          -> PCA denoise (reconstruct from top-k)
  H3  target is right-skewed (tail to 100)               -> log / Box-Cox target may help kernels
  H4  quantile/rank-gauss helps kernels & NNs            -> QuantileTransformer
Every variant is scored with the SAME repeated KFold and 3 different model families."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, QuantileTransformer, StandardScaler
from sklearn.svm import SVR

from common import SEED, h, load, plt, save

tr, te, F = load()
X, y = tr[F].values, tr.target.values
X_all = np.vstack([X, te[F].values])
cv = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X))


def cv_rmse(model, Xm, ytf=None):
    """ytf = (forward, inverse) target transform, fitted inside CV folds only."""
    s = []
    for tri, vai in cv:
        yt = y[tri] if ytf is None else ytf[0](y[tri])
        p = clone(model).fit(Xm[tri], yt).predict(Xm[vai])
        p = p if ytf is None else ytf[1](p)
        s.append(np.sqrt(np.mean((y[vai] - p) ** 2)))
    return np.mean(s), np.std(s)


# SVR epsilon/C are in TARGET units -> standardize y inside the model, otherwise a log-target run
# silently gets epsilon=1.0 on a ~0.5-sd scale (first run: log target RMSE jumped 8.9 -> 12.0)
svr = TransformedTargetRegressor(
    make_pipeline(StandardScaler(), GridSearchCV(SVR(), {"C": [1, 3, 10], "gamma": ["scale", 0.02, 0.05], "epsilon": [0.1]}, cv=3)),
    transformer=StandardScaler())
knn = make_pipeline(StandardScaler(), KNeighborsRegressor(7, weights="distance"))
et = ExtraTreesRegressor(400, min_samples_leaf=2, max_features=0.5, random_state=SEED, n_jobs=-1)
MODELS = {"SVR": svr, "kNN": knn, "ExtraTrees": et}

h("H1/H2: FEATURE REPRESENTATION  (unsupervised transforms fitted on train+test features = transductive, no target used)")
reps = {"raw": X}
qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=SEED).fit(X_all)
reps["quantile-normal"] = qt.transform(X)
sc = StandardScaler().fit(X_all)
for k in [4, 6, 8, 12]:
    p = PCA(k, random_state=SEED).fit(sc.transform(X_all))
    reps[f"pca-denoise k={k}"] = p.inverse_transform(p.transform(sc.transform(X)))
    reps[f"pca-scores k={k}"] = p.transform(sc.transform(X))
ica_models = {}
for k in [4, 6, 8, 10, 12]:
    ica = FastICA(k, whiten="unit-variance", max_iter=5000, random_state=SEED).fit(sc.transform(X_all))
    ica_models[k] = ica
    reps[f"ica k={k}"] = ica.transform(sc.transform(X))

rows = []
for name, Xm in reps.items():
    r = {"rep": name}
    for mn, m in MODELS.items():
        mu, sd = cv_rmse(m, Xm)
        r[mn] = mu
    rows.append(r)
    print(f"  {name:22s} " + "  ".join(f"{mn}={r[mn]:.3f}" for mn in MODELS))
res = pd.DataFrame(rows).set_index("rep")
res["mean"] = res.mean(axis=1)
print("\nranking by mean RMSE over 3 model families:\n", res.sort_values("mean").round(3).to_string())

h("H1 CHECK: are ICA sources 'indicator-like'? (rank-transformed + bounded -> flat, non-gaussian)")
best_k = int(res.filter(like="ica", axis=0)["mean"].idxmin().split("=")[1])
S_all = ica_models[best_k].transform(sc.transform(X_all))
S = S_all[: len(tr)]
kurt = stats.kurtosis(S_all, axis=0)
rho = [stats.spearmanr(S[:, j], y)[0] for j in range(best_k)]
print(f"best ICA k={best_k}")
print("excess kurtosis per source (uniform=-1.2, gaussian=0):", np.round(kurt, 2))
print("spearman(source, target):", np.round(rho, 3))
print("max |spearman| raw features:", round(np.max(np.abs([stats.spearmanr(X[:, j], y)[0] for j in range(24)])), 3),
      "vs ICA sources:", round(np.max(np.abs(rho)), 3))

fig, axes = plt.subplots(2, best_k, figsize=(3 * best_k, 6))
for j in range(best_k):
    axes[0, j].hist(S_all[:, j], bins=30); axes[0, j].set_title(f"IC{j+1} kurt={kurt[j]:.2f}", fontsize=9)
    axes[1, j].scatter(S[:, j], y, s=5, alpha=0.5); axes[1, j].set_title(f"rho={rho[j]:.2f}", fontsize=9)
save(fig, "06_ica_sources.png")

h("H3: TARGET TRANSFORM (fitted inside folds)")
lo = y.min()
ytfs = {
    "none": None,
    "log(y-30)": (lambda t: np.log(t - 30), lambda p: np.exp(p) + 30),
    "sqrt(y-30)": (lambda t: np.sqrt(t - 30), lambda p: np.clip(p, 0, None) ** 2 + 30),
}
best_rep = res["mean"].idxmin()
for tn, tf in ytfs.items():
    print(f"  target={tn:12s} on rep={best_rep:18s} " + "  ".join(
        f"{mn}={cv_rmse(m, reps[best_rep], tf)[0]:.3f}" for mn, m in MODELS.items()))

fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].hist(y, bins=40); ax[0].set_title("y")
ax[1].hist(np.log(y - 30), bins=40); ax[1].set_title("log(y-30)")
ax[2].hist(np.sqrt(y - 30), bins=40); ax[2].set_title("sqrt(y-30)")
save(fig, "06_target_transforms.png")

fig, ax = plt.subplots(figsize=(10, 5))
res.drop(columns="mean").plot.barh(ax=ax); ax.set_xlim(res.values.min() - 0.5, res.values.max() + 0.3)
ax.set_title("CV RMSE by feature representation (lower = better)"); ax.invert_yaxis()
save(fig, "06_representation_rmse.png")
res.to_csv("outputs/eda/06_representation_rmse.csv")
