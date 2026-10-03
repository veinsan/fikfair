"""31 - Refine the SVR member for BLEND value (not single-model RMSE).
30 found: along C at length-scale 1.0 the v5 blend improves monotonically (C=1 7.342, C=3 7.296, C=10 7.261)
while the SVR's own RMSE barely moves -> a less regularised SVR makes more 'independent' errors.
Here: finer grid in C, length-scale and epsilon; we look for a PLATEAU (neighbouring cells agree), and re-check
the winner with the blend computed per repeat (single-fit, the honest view for the final model)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.svm import SVR

mu, sd = X_all.mean(0), X_all.std(0)
DIMS = (10, 12, 14, 16, 20)
PCAS = {k: PCA(k, random_state=SEED).fit((X_all - mu) / sd) for k in DIMS}
PSD = {k: PCAS[k].transform((X_all - mu) / sd).std(0) for k in DIMS}
reps = [pd.read_csv(OUT / f"v5_25_oof_rep{r}.csv") for r in (1, 2, 3)]
FM_rep = [0.5 * (d["ICL-PL"] + d["PFN35-PL"]).values for d in reps]
FM = np.mean(FM_rep, 0)


def svr_oof(scale, C, eps):
    oof = np.zeros((3, N))
    for i, (a, b) in enumerate(FOLDS):
        lt = np.log(y[a] - 30)
        m_, s_ = lt.mean(), lt.std()
        ps = []
        for k in DIMS:
            va = PCAS[k].transform((X[a] - mu) / sd) / PSD[k]
            vb = PCAS[k].transform((X[b] - mu) / sd) / PSD[k]
            ps.append(np.exp(SVR(C=C, gamma=scale / k, epsilon=eps).fit(va, (lt - m_) / s_).predict(vb) * s_ + m_) + 30)
        oof[i // 5, b] = np.mean(ps, 0)
    return oof


def nested(F, S):
    out = []
    for s in [42, 7, 123, 2024, 99]:
        pr = np.zeros(N)
        for a, b in KFold(10, shuffle=True, random_state=s).split(F):
            w, _ = nnls(np.c_[F[a], S[a]], y[a])
            pr[b] = np.c_[F[b], S[b]] @ (w / w.sum())
        out.append(rmse(y, pr))
    return float(np.mean(out))


h("GRID: blend = nested NNLS of FM-PL-mean + SVR   (pooled | mean over single repeats)")
rows = []
for scale in [0.7, 1.0, 1.4]:
    for C in [3, 10, 30, 100]:
        for eps in [0.05, 0.2]:
            o = svr_oof(scale, C, eps)
            pooled = nested(FM, o.mean(0))
            per = np.mean([nested(FM_rep[r], o[r]) for r in range(3)])
            rows.append((scale, C, eps, np.mean([rmse(y, x) for x in o]), pooled, per))
res = pd.DataFrame(rows, columns=["scale", "C", "eps", "svr_rmse", "blend_pooled", "blend_per_repeat"])
res.to_csv(OUT / "v6_31_svr_refine.csv", index=False)
print(res.round(4).to_string(index=False))
ref = res[(res.scale == 1.0) & (res.C == 3) & (res.eps == 0.05)].iloc[0]
print(f"\nv5 setting (scale 1, C 3, eps 0.05): pooled {ref.blend_pooled:.4f} | per-repeat {ref.blend_per_repeat:.4f}")
b = res.loc[res.blend_per_repeat.idxmin()]
print(f"best by per-repeat blend: scale {b.scale} C {b.C} eps {b.eps}: pooled {b.blend_pooled:.4f} | per-repeat {b.blend_per_repeat:.4f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for a, col in zip(ax, ["blend_pooled", "blend_per_repeat"]):
    piv = res[res.eps == 0.05].pivot(index="scale", columns="C", values=col)
    im = a.imshow(piv.values, cmap="viridis_r"); fig.colorbar(im, ax=a)
    a.set_xticks(range(len(piv.columns)), piv.columns); a.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            a.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    a.set_xlabel("C"); a.set_ylabel("length-scale multiplier"); a.set_title(col + " (eps=0.05)")
save(fig, "v6_31_svr_refine.png")
