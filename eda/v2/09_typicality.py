"""09 - 'Typicality' hypothesis. v1 showed inverted-U shapes in most features at once (target peaks near z=0).
Math reading: if y falls as a record moves AWAY from the data centre in any direction, then y ~ -distance(x, centre).
We build unsupervised distance features (fitted on train+test FEATURES only, never the target):
  mahalanobis, euclid norm, distance inside the major subspace (PC1-4) and inside the minor subspace (PC5-24).
Then verify with CV whether adding them helps Ridge / SVR / CatBoost / TabICL (theory vs evaluation)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.nonparametric.smoothers_lowess import lowess

from common import OUT, SEED, h, load, plt, save

tr, te, F = load()
n = len(tr)
X_all = np.vstack([tr[F].values, te[F].values])
y = tr.target.values


def typicality(X_all):
    Z = (X_all - X_all.mean(0)) / X_all.std(0)
    ev, V = np.linalg.eigh(np.cov(Z.T))
    ev, V = ev[::-1], V[:, ::-1]
    P = Z @ V / np.sqrt(ev)  # whitened PC scores
    return pd.DataFrame({
        "maha": np.linalg.norm(P, axis=1),
        "maha_major": np.linalg.norm(P[:, :4], axis=1),
        "maha_minor": np.linalg.norm(P[:, 4:], axis=1),
        "euclid": np.linalg.norm(Z, axis=1),
        "max_abs_z": np.abs(Z).max(1),
    })


T = typicality(X_all)
h("TYPICALITY FEATURES vs TARGET (train rows)")
rows = []
for c in T:
    rows.append((c, stats.spearmanr(T[c][:n], y)[0], stats.pearsonr(T[c][:n], y)[0]))
print(pd.DataFrame(rows, columns=["feature", "spearman", "pearson"]).round(3).to_string(index=False))
print("reference: best raw feature |spearman| =",
      round(max(abs(stats.spearmanr(tr[c], y)[0]) for c in F), 3))

fig, axes = plt.subplots(1, len(T.columns), figsize=(4 * len(T.columns), 3.6))
for a, c in zip(axes, T):
    a.scatter(T[c][:n], y, s=6, alpha=0.5)
    lw = lowess(y, T[c][:n], frac=0.4)
    a.plot(lw[:, 0], lw[:, 1], c="r")
    a.set_title(f"{c}  rho={stats.spearmanr(T[c][:n], y)[0]:.2f}", fontsize=9)
save(fig, "v2_09_typicality_vs_target.png")

h("CV: raw features vs raw + typicality features (5x3 repeated KFold)")
cv = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(y))
X_raw = X_all[:n]
X_aug = np.hstack([X_raw, T.values[:n]])
ly = np.log(y - 30)


def svr_log():
    return make_pipeline(StandardScaler(), SVR(C=3, gamma="scale", epsilon=0.05))


def cv_rmse(make, Xm, log=False):
    e = []
    for a, b in cv:
        if log:
            mu, sd = ly[a].mean(), ly[a].std()
            p = np.exp(make().fit(Xm[a], (ly[a] - mu) / sd).predict(Xm[b]) * sd + mu) + 30
        else:
            p = make().fit(Xm[a], y[a]).predict(Xm[b])
        e.append(np.mean((y[b] - p) ** 2))
    return np.sqrt(np.mean(e))


def tabicl():
    from tabicl import TabICLRegressor
    return TabICLRegressor(random_state=SEED)


pca12 = PCA(12, random_state=SEED).fit((X_all - X_all.mean(0)) / X_all.std(0))
Xp = pca12.transform((X_all - X_all.mean(0)) / X_all.std(0))[:n]
Xp_aug = np.hstack([Xp, T.values[:n]])
models = [
    ("Ridge", lambda: make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 3, 30))), False, X_raw, X_aug),
    ("SVR(pca12,log)", svr_log, True, Xp, Xp_aug),
    ("CatBoost", lambda: CatBoostRegressor(iterations=1500, learning_rate=0.02, depth=5, l2_leaf_reg=5,
                                           verbose=0, random_seed=SEED, allow_writing_files=False), False, X_raw, X_aug),
    ("TabICLv2", tabicl, False, X_raw, X_aug),
]
res = []
for name, make, log, A, B in models:
    r0, r1 = cv_rmse(make, A, log), cv_rmse(make, B, log)
    res.append((name, r0, r1, r1 - r0))
    print(f"  {name:16s} base={r0:.4f}  +typicality={r1:.4f}  delta={r1 - r0:+.4f}", flush=True)
pd.DataFrame(res, columns=["model", "base", "plus_typicality", "delta"]).to_csv(OUT / "v2_09_typicality_cv.csv", index=False)
