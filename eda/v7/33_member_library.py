"""33 - A library of cheap non-foundation members, each judged by BLEND VALUE on top of v6b.
Round 6 lesson: the foundation models (FM) fail on the same rows; the blend moves only through members with
different errors, and blend value is not the same as single-model accuracy (SVR C=30: worse alone, better in blend).
So we enumerate variations along the axes that change a kernel/local model's errors:
  target transform (log / sqrt / raw), feature view (whitened PCA 10..20 / raw 24-D / PCA-24), loss (eps-insensitive
  SVR / squared KRR), locality (kNN), smoothness prior (GP Matern-3/2), plus trees as a different family.
For each member: own RMSE, residual corr vs FM and vs the v6b SVR, and nested 3-column blend
(FM-PL-mean + SVR-plateau + member) pooled and per repeat, vs the v6b reference.
All OOFs saved per repeat to outputs/eda/v7_members.npz for 34."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, Xte, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

mu, sd = X_all.mean(0), X_all.std(0)
DIMS = (10, 12, 14, 16, 20)
PCAS = {k: PCA(k, random_state=SEED).fit((X_all - mu) / sd) for k in (*DIMS, 24)}
PSD = {k: PCAS[k].transform((X_all - mu) / sd).std(0) for k in PCAS}
view = lambda k, A: PCAS[k].transform((A - mu) / sd) / PSD[k]
CELLS = [(0.7, 10), (0.7, 30), (1.0, 10), (1.0, 30)]
TF = {"log": (lambda t: np.log(t - 30), lambda p: np.exp(p) + 30),
      "sqrt": (lambda t: np.sqrt(t - 30), lambda p: np.clip(p, 0, None) ** 2 + 30),
      "raw": (lambda t: t, lambda p: p)}


def kernel_member(est, tf="log", dims=DIMS, raw=False):
    """est(scale, C, k) -> sklearn regressor; averaged over CELLS x dims, members clipped to the target range."""
    f, inv = TF[tf]

    def fp(A, t, B):
        ft = f(t)
        m_, s_ = ft.mean(), ft.std()
        out = []
        for scale, C in CELLS:
            for k in dims:
                va, vb = ((A - mu) / sd, (B - mu) / sd) if raw else (view(k, A), view(k, B))
                p = inv(est(scale, C, va.shape[1]).fit(va, (ft - m_) / s_).predict(vb) * s_ + m_)
                out.append(np.clip(p, t.min(), t.max()))
        return np.mean(out, 0)
    return fp


svr = lambda scale, C, k: SVR(C=C, gamma=scale / k, epsilon=0.05)
krr = lambda scale, C, k: KernelRidge(kernel="rbf", gamma=scale / k, alpha=1.0 / C)


def knn_fp(k_nn):
    def fp(A, t, B):
        lt = np.log(t - 30)
        return np.mean([np.exp(KNeighborsRegressor(k_nn, weights="distance").fit(view(k, A), lt).predict(view(k, B))) + 30
                        for k in DIMS], 0)
    return fp


def gp_fp(A, t, B):
    lt = np.log(t - 30)
    m_, s_ = lt.mean(), lt.std()
    ker = ConstantKernel(1.0) * Matern(length_scale=4.0, nu=1.5) + WhiteKernel(0.3)
    g = GaussianProcessRegressor(ker, random_state=SEED).fit(view(12, A), (lt - m_) / s_)
    return np.clip(np.exp(g.predict(view(12, B)) * s_ + m_) + 30, t.min(), t.max())


def et_fp(A, t, B):
    return ExtraTreesRegressor(500, min_samples_leaf=2, max_features=0.5, random_state=SEED, n_jobs=-1).fit(A, t).predict(B)


MEMBERS = {
    "SVR-plateau (v6b)": kernel_member(svr),
    "SVR sqrt-target": kernel_member(svr, "sqrt"),
    "SVR raw-target": kernel_member(svr, "raw"),
    "SVR raw-24D view": kernel_member(svr, dims=(24,), raw=True),
    "SVR PCA-24 view": kernel_member(svr, dims=(24,)),
    "KRR log": kernel_member(krr),
    "KRR raw-target": kernel_member(krr, "raw"),
    "kNN-7 log": knn_fp(7),
    "kNN-20 log": knn_fp(20),
    "GP Matern32 log": gp_fp,
    "ExtraTrees": et_fp,
}

reps = [pd.read_csv(OUT / f"v5_25_oof_rep{r}.csv") for r in (1, 2, 3)]
FM = np.stack([0.5 * (d["ICL-PL"] + d["PFN35-PL"]).values for d in reps])

h("BUILDING MEMBER OOFs (15 folds each)")
O = {}
for name, fp in MEMBERS.items():
    o = np.zeros((3, N))
    for i, (a, b) in enumerate(FOLDS):
        o[i // 5, b] = fp(X[a], y[a], X[b])
    O[name] = o
    print(f"  {name:20s} RMSE={np.mean([rmse(y, r) for r in o]):.4f}", flush=True)
np.savez(OUT / "v7_members.npz", **{k: v for k, v in O.items()}, FM=FM)


def nested(cols, seed):
    P = np.column_stack(cols)
    pr = np.zeros(N)
    for a, b in KFold(10, shuffle=True, random_state=seed).split(P):
        w, _ = nnls(P[a], y[a])
        pr[b] = P[b] @ (w / w.sum())
    return pr


def score(members):
    pooled = np.mean([rmse(y, nested([FM.mean(0)] + [O[m].mean(0) for m in members], s)) for s in [42, 7, 123, 2024, 99]])
    per = np.mean([np.mean([rmse(y, nested([FM[r]] + [O[m][r] for m in members], s)) for s in [42, 7, 123]]) for r in range(3)])
    return pooled, per


S0 = "SVR-plateau (v6b)"
ref = score([S0])
h(f"BLEND VALUE on top of v6b (reference pooled {ref[0]:.4f} | per-repeat {ref[1]:.4f})")
rows = []
for name in MEMBERS:
    if name == S0:
        continue
    p3 = score([S0, name])
    p2 = score([name])
    rc_fm = np.corrcoef(y - O[name].mean(0), y - FM.mean(0))[0, 1]
    rc_s = np.corrcoef(y - O[name].mean(0), y - O[S0].mean(0))[0, 1]
    rows.append((name, np.mean([rmse(y, r) for r in O[name]]), rc_fm, rc_s, p3[0] - ref[0], p3[1] - ref[1], p2[0], p2[1]))
    print(f"  {name:20s} corr FM={rc_fm:.3f} SVR={rc_s:.3f} | v6b+this: pooled {p3[0]:.4f} ({p3[0] - ref[0]:+.4f}) "
          f"per-rep {p3[1]:.4f} ({p3[1] - ref[1]:+.4f}) | replaces SVR: {p2[0]:.4f} / {p2[1]:.4f}", flush=True)
res = pd.DataFrame(rows, columns=["member", "rmse", "corr_FM", "corr_SVR", "d_pooled", "d_per_rep", "alone_pooled", "alone_per_rep"])
res.to_csv(OUT / "v7_33_members.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.scatter(res.d_pooled, res.d_per_rep)
for _, r in res.iterrows():
    ax.annotate(r.member, (r.d_pooled, r.d_per_rep), fontsize=7)
ax.axhline(0, c="k"); ax.axvline(0, c="k")
ax.set_xlabel("blend delta, pooled (neg = better)"); ax.set_ylabel("blend delta, per repeat")
ax.set_title("third-member value on top of v6b (bottom-left = robust gain)")
save(fig, "v7_33_members.png")
