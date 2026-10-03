"""43 - Gaussian processes for a near-deterministic target.
39: near-duplicate pairs differ by only ~2 in target -> y is close to a function of x; our best CV ~7.25 means that
function is complex for 300 samples. A GP with ARD learns one length-scale per input direction by marginal
likelihood, i.e. it discovers which directions matter and how smoothly y varies along each.
37 showed the target signal is spread over small-variance PCs, which an ARD GP on RAW standardized features
handles badly (a minor PC is a combination of all raw columns). So we compare input bases:
  raw standardized | whitened PCs | unwhitened PCs | rank (uniform) features
and kernels: ARD Matern-5/2, ARD RBF, additive (sum of 1-D RBFs) + ARD RBF.
Target: log(y-30), standardized in-fold; predictions back-transformed with the GP predictive variance
(E[y] = exp(mu + var/2) + 30). Repeat-1 folds (5) for the sweep."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern, WhiteKernel

mu, sd = X_all.mean(0), X_all.std(0)
Z = (X_all - mu) / sd
ev, V = np.linalg.eigh(np.cov(Z.T))
U = (stats.rankdata(X_all, axis=0) - 0.5) / len(X_all)
BASES = {
    "raw": Z,
    "whitened PCs": Z @ V / np.sqrt(ev),
    "PCs (unwhitened)": Z @ V,
    "ranks": (U - 0.5) / U.std(0),
}
KERNELS = {
    "ARD Matern52": lambda: C(1.0) * Matern(length_scale=np.full(24, 3.0), nu=2.5, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(0.1, (1e-5, 1)),
    "ARD RBF": lambda: C(1.0) * RBF(length_scale=np.full(24, 3.0), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(0.1, (1e-5, 1)),
}


def gp_fp(basis, kern):
    B = BASES[basis]

    def fp(a, b):
        lt = np.log(y[a] - 30)
        m_, s_ = lt.mean(), lt.std()
        g = GaussianProcessRegressor(KERNELS[kern](), n_restarts_optimizer=2, random_state=SEED, normalize_y=False)
        g.fit(B[a], (lt - m_) / s_)
        pm, ps = g.predict(B[b], return_std=True)
        return np.exp(pm * s_ + m_ + 0.5 * (ps * s_) ** 2) + 30, g
    return fp


h("GP sweep, repeat 1 (5 folds)")
rows, oofs = [], {}
for basis in BASES:
    for kern in KERNELS:
        o = np.zeros(N)
        noise = []
        for a, b in FOLDS[:5]:
            o[b], g = gp_fp(basis, kern)(a, b)
            noise.append(g.kernel_.k2.noise_level)
        oofs[(basis, kern)] = o
        rows.append((basis, kern, rmse(y, o), np.mean(noise)))
        ls = g.kernel_.k1.k2.length_scale
        print(f"  {basis:18s} {kern:13s} RMSE={rmse(y, o):.4f}  learned noise var (std. log-y)={np.mean(noise):.3f}  "
              f"shortest length-scales: {np.round(np.sort(ls)[:4], 2)}", flush=True)
res = pd.DataFrame(rows, columns=["basis", "kernel", "rmse", "noise"])
res.to_csv(OUT / "v8_43_gp.csv", index=False)
print("\n  references, same folds: TabICL plain 7.525, SVR-plateau ~7.9, v6b blend ~7.2")
pd.DataFrame({f"{k[0]}|{k[1]}": v for k, v in oofs.items()}).assign(target=y).to_csv(OUT / "v8_43_gp_oof_rep1.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
for kern, d in res.groupby("kernel"):
    ax.plot(d.basis, d.rmse, "o-", label=kern)
ax.axhline(7.525, c="k", ls="--", label="TabICL"); ax.legend(); ax.set_ylabel("CV RMSE, repeat 1"); ax.set_title("GP: input basis x kernel")
save(fig, "v8_43_gp.png")
