"""07 - Model benchmark on IDENTICAL folds (5-fold x 3 repeats). Saves OOF + test preds for 08.
Includes pretrained tabular foundation models (in-context learners, no gradient training on our data):
  TabICLv2 (open weights, HF jingang/TabICL) and TabPFN v2 (HF Prior-Labs, ungated v2 checkpoint).
Also re-checks CV design: random KFold vs similarity-GroupKFold (leakage via near twins)."""
import sys
import time
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy.sparse.csgraph import connected_components
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVR

from common import OUT, SEED, h, load, plt, save

warnings.filterwarnings("ignore")
tr, te, F = load()
X, y, Xte = tr[F].values, tr.target.values, te[F].values
N_REP = 3
cv = list(RepeatedKFold(n_splits=5, n_repeats=N_REP, random_state=SEED).split(X))
LOG = FunctionTransformer(lambda t: np.log(t - 30), lambda p: np.exp(p) + 30, check_inverse=False)
sc_all = StandardScaler().fit(np.vstack([X, Xte]))
pca12 = PCA(12, random_state=SEED).fit(sc_all.transform(np.vstack([X, Xte])))
to_pca = FunctionTransformer(lambda a: pca12.transform(sc_all.transform(a)))


def logy(m):
    return TransformedTargetRegressor(m, transformer=LOG)


def zy(m):
    return TransformedTargetRegressor(m, transformer=StandardScaler())


def gp():
    k = ConstantKernel(1.0) * Matern(length_scale=np.ones(24) * 3, nu=2.5, length_scale_bounds=(0.1, 1e3)) + WhiteKernel(0.3)
    return zy(make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=1, random_state=SEED)))


def tabicl(**kw):
    from tabicl import TabICLRegressor
    return TabICLRegressor(random_state=SEED, **kw)


def tabpfn(**kw):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    return TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", random_state=SEED, **kw)


class MultiPCASVR:
    """06b finding: averaging SVR(log y) over PCA dims 10..20 is as good as the best single k, and robust."""
    def __init__(self, dims=(10, 12, 14, 16, 20)):
        self.dims = dims

    def _view(self, k, A):
        return self.pcas[k].transform(sc_all.transform(A)) / self.sds[k]

    def fit(self, A, t):
        Z = sc_all.transform(np.vstack([X, Xte]))
        self.pcas = {k: PCA(k, random_state=SEED).fit(Z) for k in self.dims}
        self.sds = {k: self.pcas[k].transform(Z).std(0) for k in self.dims}
        self.m = {k: logy(zy(SVR(C=3, gamma=1 / k, epsilon=0.05))).fit(self._view(k, A), t) for k in self.dims}
        return self

    def predict(self, A):
        return np.mean([self.m[k].predict(self._view(k, A)) for k in self.dims], 0)


raw_plus_pca = FunctionTransformer(lambda a: np.hstack([a, pca12.transform(sc_all.transform(a))]))

MODELS = {
    "Ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 3, 30))),
    "kNN(pca12,log)": lambda: logy(make_pipeline(to_pca, StandardScaler(), KNeighborsRegressor(7, weights="distance"))),
    "SVR(pca12,log)": lambda: logy(zy(make_pipeline(to_pca, StandardScaler(), SVR(C=3, gamma="scale", epsilon=0.1)))),
    "GP-ARD": gp,
    "GP-ARD(log)": lambda: logy(gp()),
    "ExtraTrees": lambda: ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.5, random_state=SEED, n_jobs=-1),
    "LightGBM": lambda: LGBMRegressor(n_estimators=600, learning_rate=0.02, num_leaves=8, min_child_samples=10,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1, verbose=-1, random_state=SEED),
    "CatBoost": lambda: CatBoostRegressor(iterations=1500, learning_rate=0.02, depth=5, l2_leaf_reg=5, verbose=0, random_seed=SEED),
    "TabPFNv2": tabpfn,
    "TabICLv2": tabicl,
    "TabICLv2(log)": lambda: logy(tabicl()),
    # --- 09 enhancements (run: python eda/07_model_benchmark.py SVR-multiPCA TabICLv2-32 ...) ---
    "SVR-multiPCA": MultiPCASVR,
    "TabICLv2-32": lambda: tabicl(n_estimators=32),
    "TabICLv2+PCA": lambda: make_pipeline(raw_plus_pca, tabicl()),
    "TabPFNv2-16": lambda: tabpfn(n_estimators=16),
}
only = sys.argv[1:]  # optional: run a subset, e.g. python eda/07_model_benchmark.py Ridge GP-ARD
if only:
    MODELS = {k: v for k, v in MODELS.items() if k in only}

oof_path, te_path = OUT / "07_oof.csv", OUT / "07_test_pred.csv"
oof_df = pd.read_csv(oof_path) if oof_path.exists() else pd.DataFrame({"record_id": tr.record_id, "target": y})
te_df = pd.read_csv(te_path) if te_path.exists() else pd.DataFrame({"record_id": te.record_id})

h("RANDOM 5x3 KFOLD")
summary = []
for name, make in MODELS.items():
    t = time.time()
    oof = np.zeros((N_REP, len(y)))
    fold_rmse = []
    for i, (tri, vai) in enumerate(cv):
        p = make().fit(X[tri], y[tri]).predict(X[vai])
        oof[i // 5, vai] = p
        fold_rmse.append(np.sqrt(np.mean((y[vai] - p) ** 2)))
    oof_mean = oof.mean(0)
    te_df[name] = make().fit(X, y).predict(Xte)
    oof_df[name] = oof_mean
    rep_rmse = [np.sqrt(np.mean((y - o) ** 2)) for o in oof]
    summary.append((name, np.mean(rep_rmse), np.std(fold_rmse), np.mean(np.abs(y - oof_mean)), time.time() - t))
    print(f"  {name:16s} RMSE={np.mean(rep_rmse):.4f}  fold-sd={np.std(fold_rmse):.3f}  "
          f"MAE={summary[-1][3]:.3f}  ({summary[-1][4]:.0f}s)", flush=True)
oof_df.to_csv(oof_path, index=False)
te_df.to_csv(te_path, index=False)

h("LEADERBOARD (all models run so far, RMSE of OOF averaged over repeats)")
cols = [c for c in oof_df.columns if c not in ("record_id", "target")]
lb = pd.Series({c: np.sqrt(np.mean((y - oof_df[c]) ** 2)) for c in cols}).sort_values()
print(lb.round(4).to_string())
print("const-mean:", round(y.std(ddof=0), 4))

h("CV DESIGN CHECK: similarity GroupKFold (radius 1.0 clusters kept in one fold)")
Z = sc_all.transform(X)
_, groups = connected_components(NearestNeighbors(radius=1.0).fit(Z).radius_neighbors_graph(Z), directed=False)
for name in [n for n in ["Ridge", "GP-ARD", "CatBoost", "TabICLv2"] if n in MODELS]:
    p = np.zeros(len(y))
    for tri, vai in GroupKFold(5).split(X, y, groups):
        p[vai] = MODELS[name]().fit(X[tri], y[tri]).predict(X[vai])
    print(f"  {name:12s} GroupKFold RMSE={np.sqrt(np.mean((y - p) ** 2)):.4f}  vs random KFold {lb[name]:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
lb.plot.barh(ax=ax, color="tab:blue"); ax.axvline(y.std(ddof=0), c="r", ls="--", label="predict mean")
ax.set_xlim(lb.min() - 0.7, y.std() + 0.3); ax.invert_yaxis(); ax.legend(); ax.set_title("OOF RMSE (5x3 repeated KFold)")
save(fig, "07_model_leaderboard.png")
