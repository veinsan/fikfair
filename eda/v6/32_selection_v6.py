"""32 - v6 candidate: FM-PL-mean + plateau-averaged SVR (C in {10,30} x length-scale in {0.7,1.0}, eps 0.05).
Protocol vs v5 (same FM blend + SVR C=3) and v3 (ICL-PL + SVR C=3): nested NNLS pooled and per repeat,
paired bootstrap on per-repeat stacked rows, public-29 / private-67 win probabilities."""
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
PLATEAU = [(0.7, 10), (0.7, 30), (1.0, 10), (1.0, 30)]
reps = [pd.read_csv(OUT / f"v5_25_oof_rep{r}.csv") for r in (1, 2, 3)]


def svr_oof(cells):
    oof = np.zeros((3, N))
    for i, (a, b) in enumerate(FOLDS):
        lt = np.log(y[a] - 30)
        m_, s_ = lt.mean(), lt.std()
        ps = []
        for scale, C in cells:
            for k in DIMS:
                va = PCAS[k].transform((X[a] - mu) / sd) / PSD[k]
                vb = PCAS[k].transform((X[b] - mu) / sd) / PSD[k]
                ps.append(np.exp(SVR(C=C, gamma=scale / k, epsilon=0.05).fit(va, (lt - m_) / s_).predict(vb) * s_ + m_) + 30)
        oof[i // 5, b] = np.mean(ps, 0)
    return oof


def nested(F, S, seed):
    pr = np.zeros(N)
    for a, b in KFold(10, shuffle=True, random_state=seed).split(F):
        w, _ = nnls(np.c_[F[a], S[a]], y[a])
        pr[b] = np.c_[F[b], S[b]] @ (w / w.sum())
    return pr


S_old = np.stack([d["SVR"].values for d in reps])
S_new = svr_oof(PLATEAU)
ICL = np.stack([d["ICL-PL"].values for d in reps])
FM = np.stack([0.5 * (d["ICL-PL"] + d["PFN35-PL"]).values for d in reps])
print(f"SVR RMSE  old (C=3): {np.mean([rmse(y, s) for s in S_old]):.4f}   plateau avg: {np.mean([rmse(y, s) for s in S_new]):.4f}")
print(f"resid corr vs FM  old: {np.corrcoef(y - S_old.mean(0), y - FM.mean(0))[0, 1]:.3f}   plateau: "
      f"{np.corrcoef(y - S_new.mean(0), y - FM.mean(0))[0, 1]:.3f}")

CANDS = {"v3: ICL-PL + SVR(C=3)": (ICL, S_old), "v5: FM-mean + SVR(C=3)": (FM, S_old),
         "v6: FM-mean + SVR(plateau)": (FM, S_new), "ICL-PL + SVR(plateau)": (ICL, S_new)}
seeds = [42, 7, 123, 2024, 99]
h("NESTED BLEND RMSE")
rows, stacked = [], {}
for name, (F, S) in CANDS.items():
    pooled = np.mean([rmse(y, nested(F.mean(0), S.mean(0), s)) for s in seeds])
    per = [np.mean([rmse(y, nested(F[r], S[r], s)) for s in seeds]) for r in range(3)]
    stacked[name] = np.concatenate([nested(F[r], S[r], SEED) for r in range(3)])
    w, _ = nnls(np.c_[F.mean(0), S.mean(0)], y)
    rows.append((name, pooled, np.mean(per), *per, w[1] / w.sum()))
res = pd.DataFrame(rows, columns=["candidate", "pooled", "per_repeat_mean", "rep1", "rep2", "rep3", "w_svr"])
print(res.round(4).to_string(index=False))

h("PAIRED TESTS vs v5 (per-repeat stacked, 3 x 382 rows)")
Y = np.tile(y, 3)
rng = np.random.default_rng(SEED)
base = stacked["v5: FM-mean + SVR(C=3)"]
for name in ["v6: FM-mean + SVR(plateau)", "v3: ICL-PL + SVR(C=3)"]:
    c = stacked[name]
    d = (Y - c) ** 2 - (Y - base) ** 2
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    win = lambda k: np.mean([np.mean(((Y - c) ** 2)[i]) < np.mean(((Y - base) ** 2)[i])
                             for i in (rng.choice(len(Y), k, replace=False) for _ in range(4000))])
    print(f"  {name:28s} P(better than v5)={np.mean(boot < 0):.3f}  P(win) public-29={win(29):.3f}  private-67={win(67):.3f}")
res.to_csv(OUT / "v6_32_selection.csv", index=False)
pd.DataFrame({"svr_plateau_rep%d" % (r + 1): S_new[r] for r in range(3)}).assign(target=y).to_csv(OUT / "v6_32_svr_oof.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(res))
ax.bar(x - 0.2, res.pooled, 0.4, label="pooled"); ax.bar(x + 0.2, res.per_repeat_mean, 0.4, label="mean of single repeats")
ax.set_xticks(x, res.candidate, rotation=15, fontsize=7); ax.set_ylim(7.2, 7.5); ax.legend(); ax.set_title("nested blend RMSE")
save(fig, "v6_32_selection.png")
