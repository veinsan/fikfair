"""26 - Pool the three repeats of 25 (15 folds) and decide v5 vs v3 with the same protocol as rounds 2-4.
Per repeat: fixed-weight blends (no fitting). Pooled: repeat-averaged OOF -> nested NNLS (5 outer seeds), paired
bootstrap P(v5 better than v3), public-29 / private-67 win probabilities."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import N, OUT, SEED, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold

reps = [pd.read_csv(OUT / f"v5_25_oof_rep{r}.csv") for r in (1, 2, 3)]
for r in reps:
    assert (r.target.values == y).all()
M = ["ICL-PL", "PFN35-PL", "SVR"]

h("PER REPEAT, fixed weights")
w_v3 = {"ICL-PL": 0.77, "SVR": 0.23}
w_v5 = {"ICL-PL": 0.385, "PFN35-PL": 0.385, "SVR": 0.23}
blend = lambda df, w: sum(df[k] * v for k, v in w.items())
rows = []
for i, df in enumerate(reps, 1):
    a, b = rmse(y, blend(df, w_v3)), rmse(y, blend(df, w_v5))
    rows.append((i, a, b, b - a))
    print(f"  repeat {i}: v3={a:.4f}  v5={b:.4f}  delta={b - a:+.4f}")
per = pd.DataFrame(rows, columns=["repeat", "v3", "v5", "delta"])
print(f"  mean over repeats: v3={per.v3.mean():.4f}  v5={per.v5.mean():.4f}  delta={per.delta.mean():+.4f}"
      f"  (sd of delta {per.delta.std():.4f})")

h("POOLED (repeat-averaged OOF): nested NNLS, bootstrap, LB simulation")
avg = sum(df[M] for df in reps) / 3


def nested(cols, seed):
    P = avg[cols].values
    out = np.zeros(N)
    for a, b in KFold(10, shuffle=True, random_state=seed).split(P):
        w, _ = nnls(P[a], y[a])
        out[b] = P[b] @ (w / w.sum())
    return out


cands = {"v3: ICL-PL + SVR": ["ICL-PL", "SVR"], "v5: ICL-PL + PFN35-PL + SVR": M, "PFN35-PL + SVR": ["PFN35-PL", "SVR"]}
base = nested(cands["v3: ICL-PL + SVR"], SEED)
rng = np.random.default_rng(SEED)
res = []
for name, cols in cands.items():
    runs = [rmse(y, nested(cols, s)) for s in [42, 7, 123, 2024, 99]]
    c = nested(cols, SEED)
    d = (y - c) ** 2 - (y - base) ** 2
    boot = np.array([d[rng.integers(0, N, N)].mean() for _ in range(4000)])
    win = lambda k: np.mean([((y - c) ** 2)[i].mean() < ((y - base) ** 2)[i].mean()
                             for i in (rng.choice(N, k, replace=False) for _ in range(4000))])
    P = avg[cols].values
    w, _ = nnls(P, y)
    res.append((name, np.mean(runs), np.std(runs), (boot < 0).mean(), win(29), win(67),
                " ".join(f"{k}:{v:.2f}" for k, v in zip(cols, w / w.sum()))))
res = pd.DataFrame(res, columns=["candidate", "nested", "sd", "P(better)boot", "P(win)pub29", "P(win)prv67", "weights"])
pd.set_option("display.width", 220)
print(res.round(4).to_string(index=False))
res.to_csv(OUT / "v5_26_selection.csv", index=False)
print("  residual corr ICL-PL vs PFN35-PL (pooled):", round(np.corrcoef(y - avg["ICL-PL"], y - avg["PFN35-PL"])[0, 1], 3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(per.repeat, per.v3, "o-", label="v3"); ax[0].plot(per.repeat, per.v5, "o-", label="v5")
ax[0].set_xticks([1, 2, 3]); ax[0].set_xlabel("repeat"); ax[0].set_ylabel("RMSE"); ax[0].legend(); ax[0].set_title("fixed-weight blends")
ax[1].barh(res.candidate, res.nested, xerr=res.sd); ax[1].invert_yaxis()
ax[1].set_xlim(res.nested.min() - 0.05, res.nested.max() + 0.03); ax[1].set_title("pooled nested NNLS")
save(fig, "v5_26_selection.png")
