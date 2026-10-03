"""30 - Pick the kernel from the variogram (theory-driven model choice for the non-foundation blend member).
v2 script 10: the target semivariogram rises roughly LINEARLY from the origin. In geostatistics a linear/
exponential variogram near zero means a rough (non-differentiable) process = exponential covariance
(Matern nu=1/2, a.k.a. Laplacian kernel). The RBF kernel used by our SVR assumes an infinitely smooth function.
Test: same SVR recipe (whitened PCA views, log(y-30) target standardized in-fold, average over PCA dims
10..20), swapping the kernel: RBF vs Laplacian (L1) vs Matern-1/2 (L2 distance) vs Matern-3/2.
Reported: CV RMSE, residual corr vs the v5 foundation blend, nested blend when it replaces / joins the RBF SVR."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, Xte, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.svm import SVR

mu, sd = X_all.mean(0), X_all.std(0)
DIMS = (10, 12, 14, 16, 20)
PCAS = {k: PCA(k, random_state=SEED).fit((X_all - mu) / sd) for k in DIMS}
PSD = {k: PCAS[k].transform((X_all - mu) / sd).std(0) for k in DIMS}
view = lambda k, A: PCAS[k].transform((A - mu) / sd) / PSD[k]


def kern(name, scale):
    if name == "rbf":
        return lambda A, B: np.exp(-cdist(A, B, "sqeuclidean") / A.shape[1] * scale)
    if name == "laplacian":
        return lambda A, B: np.exp(-cdist(A, B, "cityblock") / A.shape[1] * scale)
    if name == "matern12":
        return lambda A, B: np.exp(-cdist(A, B) / np.sqrt(A.shape[1]) * scale)
    if name == "matern32":
        def f(A, B):
            r = np.sqrt(3) * cdist(A, B) / np.sqrt(A.shape[1]) * scale
            return (1 + r) * np.exp(-r)
        return f


def svr_fp(kname, scale, C):
    K = kern(kname, scale)

    def fp(A, t, B):
        lt = np.log(t - 30)
        m_, s_ = lt.mean(), lt.std()
        out = []
        for k in DIMS:
            va, vb = view(k, A), view(k, B)
            m = SVR(kernel="precomputed", C=C, epsilon=0.05).fit(K(va, va), (lt - m_) / s_)
            out.append(np.exp(m.predict(K(vb, va)) * s_ + m_) + 30)
        return np.mean(out, 0)
    return fp


def cv(fp):
    oof = np.zeros((3, N))
    for i, (a, b) in enumerate(FOLDS):
        oof[i // 5, b] = fp(X[a], y[a], X[b])
    return float(np.mean([rmse(y, o) for o in oof])), oof.mean(0)


h("KERNEL x LENGTH-SCALE x C  (RMSE averaged over the 3 repeats)")
rows, oofs = [], {}
for kname in ["rbf", "laplacian", "matern12", "matern32"]:
    for scale in [0.5, 1.0, 2.0]:
        for C in [1, 3, 10]:
            s, o = cv(svr_fp(kname, scale, C))
            rows.append((kname, scale, C, s))
            oofs[(kname, scale, C)] = o
res = pd.DataFrame(rows, columns=["kernel", "scale", "C", "rmse"])
res.to_csv(OUT / "v6_30_kernel_grid.csv", index=False)
print(res.pivot_table(index=["kernel", "scale"], columns="C", values="rmse").round(4).to_string())
best = res.loc[res.groupby("kernel").rmse.idxmin()]
print("\nbest per kernel:\n", best.round(4).to_string(index=False))
print("flatness: share of grid cells within 0.05 of the kernel's best:",
      res.groupby("kernel").apply(lambda d: (d.rmse < d.rmse.min() + 0.05).mean()).round(2).to_dict())

h("IN THE BLEND (v5 foundation blend = mean(ICL-PL, PFN35-PL), repeat-averaged OOF)")
reps = [pd.read_csv(OUT / f"v5_25_oof_rep{r}.csv") for r in (1, 2, 3)]
avg = sum(d[["ICL-PL", "PFN35-PL", "SVR"]] for d in reps) / 3
FM = (0.5 * (avg["ICL-PL"] + avg["PFN35-PL"])).values


def nested(P):
    out = []
    for s in [42, 7, 123, 2024, 99]:
        pr = np.zeros(N)
        for a, b in KFold(10, shuffle=True, random_state=s).split(P):
            w, _ = nnls(P[a], y[a])
            pr[b] = P[b] @ (w / w.sum())
        out.append(rmse(y, pr))
    return float(np.mean(out))


ref = nested(np.c_[FM, avg["SVR"].values])
print(f"  v5 reference (FM + RBF SVR): {ref:.4f}")
brow = []
for _, r in best.iterrows():
    o = oofs[(r.kernel, r.scale, r.C)]
    rc = np.corrcoef(y - o, y - FM)[0, 1]
    b1 = nested(np.c_[FM, o])
    brow.append((r.kernel, r.rmse, rc, b1, b1 - ref))
    print(f"  {r.kernel:10s} best RMSE={r.rmse:.4f}  resid-corr vs FM={rc:.3f}  FM + this kernel={b1:.4f} ({b1 - ref:+.4f})")
pd.DataFrame(brow, columns=["kernel", "rmse", "resid_corr", "blend", "delta"]).to_csv(OUT / "v6_30_kernel_blend.csv", index=False)
pd.DataFrame({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in oofs.items()}).assign(target=y).to_csv(OUT / "v6_30_kernel_oof.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
for kname, d in res.groupby("kernel"):
    g = d.groupby("scale").rmse.min()
    ax.plot(g.index, g.values, "o-", label=kname)
ax.set_xscale("log"); ax.set_xlabel("inverse length-scale multiplier"); ax.set_ylabel("best CV RMSE over C")
ax.legend(); ax.set_title("kernel choice for the SVR member")
save(fig, "v6_30_kernel_choice.png")
