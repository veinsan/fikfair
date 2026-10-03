"""35 - Where did the v6 gain come from, and can a 'tail expert' extend it?
Facts: 26 training rows with y > 70 (7%) carry ~40% of the squared error. v6 gained by loosening the SVR (C 3 -> 10-30).
A. Decompose the v5 -> v6b squared-error change by TRUE target range (diagnostic only) and by PREDICTED range
   (actionable: what the model can know at test time).
B. How separable is the tail? AUC of the blend's own prediction for y > 70, and recall at the top-k predictions.
C. Tail-weighted SVR members: sample_weight = 1 + lam * ((y - mean)/sd)^2 pushes the fit toward extreme targets.
   Theory: a member that shrinks less at the extremes complements foundation models that regress to the mean.
   Risk: it over-predicts ordinary rows. Judged by blend value on top of v6b, pooled and per repeat."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.svm import SVR

L = np.load(OUT / "v7_members.npz")
FM, S6 = L["FM"], L["SVR-plateau (v6b)"]
reps = [pd.read_csv(OUT / f"v5_25_oof_rep{r}.csv") for r in (1, 2, 3)]
S5 = np.stack([d["SVR"].values for d in reps])


def nested(cols, seed=SEED):
    P = np.column_stack(cols)
    pr = np.zeros(N)
    for a, b in KFold(10, shuffle=True, random_state=seed).split(P):
        w, _ = nnls(P[a], y[a])
        pr[b] = P[b] @ (w / w.sum())
    return pr


b5 = np.stack([nested([FM[r], S5[r]]) for r in range(3)])
b6 = np.stack([nested([FM[r], S6[r]]) for r in range(3)])
print(f"per-repeat RMSE  v5={np.mean([rmse(y, b) for b in b5]):.4f}  v6b={np.mean([rmse(y, b) for b in b6]):.4f}")

h("A. squared-error change v5 -> v6b (negative = v6b better), per repeat stacked")
Y = np.tile(y, 3)
d = ((Y - b6.ravel()) ** 2 - (Y - b5.ravel()) ** 2)
for label, key, edges in [("TRUE target", Y, [30, 40, 50, 60, 70, 101]), ("PREDICTED (v5)", b5.ravel(), [30, 40, 50, 60, 70, 101])]:
    g = pd.DataFrame({"bin": pd.cut(key, edges), "d": d}).groupby("bin", observed=True).d.agg(["sum", "count", "mean"])
    g["share_of_gain"] = g["sum"] / d.sum()
    print(f"\n  by {label}:\n" + g.round(3).to_string())

h("B. tail separability (y > 70) from the v6b blend prediction")
tail = (y > 70).astype(int)
p6 = b6.mean(0)
print(f"  AUC of blend prediction for y>70: {roc_auc_score(tail, p6):.3f}   (n tail = {tail.sum()})")
for k in [10, 26, 50]:
    top = np.argsort(-p6)[:k]
    print(f"  top-{k} predictions contain {tail[top].sum()} of {tail.sum()} tail rows; their mean target {y[top].mean():.1f}, mean pred {p6[top].mean():.1f}")
print(f"  tail rows: mean target {y[tail == 1].mean():.1f}, mean prediction {p6[tail == 1].mean():.1f}")

h("C. tail-weighted SVR members (blend value on top of v6b)")
mu, sd = X_all.mean(0), X_all.std(0)
DIMS = (10, 12, 14, 16, 20)
PCAS = {k: PCA(k, random_state=SEED).fit((X_all - mu) / sd) for k in DIMS}
PSD = {k: PCAS[k].transform((X_all - mu) / sd).std(0) for k in DIMS}
CELLS = [(0.7, 10), (0.7, 30), (1.0, 10), (1.0, 30)]


def wsvr_oof(lam, upper_only):
    o = np.zeros((3, N))
    for i, (a, b) in enumerate(FOLDS):
        t = y[a]
        lt = np.log(t - 30)
        m_, s_ = lt.mean(), lt.std()
        z = (t - t.mean()) / t.std()
        w = 1 + lam * (np.clip(z, 0, None) if upper_only else z) ** 2
        ps = []
        for scale, C in CELLS:
            for k in DIMS:
                va = PCAS[k].transform((X[a] - mu) / sd) / PSD[k]
                vb = PCAS[k].transform((X[b] - mu) / sd) / PSD[k]
                p = np.exp(SVR(C=C, gamma=scale / k, epsilon=0.05).fit(va, (lt - m_) / s_, sample_weight=w).predict(vb) * s_ + m_) + 30
                ps.append(np.clip(p, t.min(), t.max()))
        o[i // 5, b] = np.mean(ps, 0)
    return o


def score(cols_fn):
    pooled = np.mean([rmse(y, nested(cols_fn(None), s)) for s in [42, 7, 123, 2024, 99]])
    per = np.mean([np.mean([rmse(y, nested(cols_fn(r), s)) for s in [42, 7, 123]]) for r in range(3)])
    return pooled, per


pick = lambda A, r: A.mean(0) if r is None else A[r]
ref = score(lambda r: [pick(FM, r), pick(S6, r)])
print(f"  v6b reference: pooled {ref[0]:.4f} | per-repeat {ref[1]:.4f}")
rows = []
for upper in [False, True]:
    for lam in [0.5, 1.0, 2.0, 4.0]:
        W = wsvr_oof(lam, upper)
        rep = score(lambda r: [pick(FM, r), pick(W, r)])
        add = score(lambda r: [pick(FM, r), pick(S6, r), pick(W, r)])
        own = np.mean([rmse(y, x) for x in W])
        rows.append(("upper" if upper else "both", lam, own, rep[0] - ref[0], rep[1] - ref[1], add[0] - ref[0], add[1] - ref[1]))
        print(f"  weights={rows[-1][0]:5s} lam={lam:<3}  own RMSE={own:.3f}  replaces SVR: pooled {rep[0] - ref[0]:+.4f} per-rep {rep[1] - ref[1]:+.4f}"
              f" | added as 3rd: pooled {add[0] - ref[0]:+.4f} per-rep {add[1] - ref[1]:+.4f}", flush=True)
res = pd.DataFrame(rows, columns=["weights", "lam", "own_rmse", "rep_pooled", "rep_per", "add_pooled", "add_per"])
res.to_csv(OUT / "v7_35_tail_svr.csv", index=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
g = pd.DataFrame({"bin": pd.cut(Y, [30, 40, 50, 60, 70, 101]), "d": d}).groupby("bin", observed=True).d.sum()
ax[0].bar([str(i) for i in g.index], g.values); ax[0].axhline(0, c="k"); ax[0].set_title("squared-error change v5 -> v6b by true target range")
for wname, dd in res.groupby("weights"):
    ax[1].plot(dd.lam, dd.rep_per, "o-", label=f"{wname}: replaces SVR (per-repeat)")
ax[1].axhline(0, c="k"); ax[1].set_xlabel("lambda"); ax[1].set_ylabel("blend delta vs v6b"); ax[1].legend(fontsize=8)
ax[1].set_title("tail-weighted SVR")
save(fig, "v7_35_tail.png")
