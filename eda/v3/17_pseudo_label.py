"""17 - Transductive pseudo-labelling, dissected. 16 found ICL[pseudo-label] 7.512 vs 7.601 (single seed).
Before trusting it, rule out the boring explanations:
  control-1  the 2nd pass used another random_state -> plain TabICL with that seed
  control-2  all v2 comparisons are 4-seed bagged -> bag everything over seeds 0..3
Variants (a query row NEVER sees its own pseudo label; true labels of query rows are never used):
  halves     : pseudo-labelled half of the query set joins the context to predict the other half
  4-split    : 3/4 of the query set joins the context (more pseudo rows)
  blend-PL   : pseudo labels come from the v2 blend (0.725 TabICL-bag + 0.275 SVR) instead of TabICL alone
  2 rounds   : repeat the procedure once more with the refined labels
Why could it work? The unlabeled rows densify the context around each query (variogram: error is limited by
neighbour spacing ~1.15) and the pseudo labels act as a smoothness prior. Checked below per distance bin."""
from cvlib import FOLDS, N, OUT, SEED, X, X_all, Xte, h, plt, rmse, run_cv, save, store, y
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVR
from tabicl import TabICLRegressor

SEEDS = (0, 1, 2, 3)
mu, sd = X_all.mean(0), X_all.std(0)
Z_all = (X_all - mu) / sd
PCAS = {k: PCA(k, random_state=SEED).fit(Z_all) for k in (10, 12, 14, 16, 20)}
PSD = {k: PCAS[k].transform(Z_all).std(0) for k in PCAS}


def icl_bag(A, t, B, seeds=SEEDS):
    return np.mean([TabICLRegressor(random_state=s, norm_methods=["none"]).fit(A, t).predict(B) for s in seeds], 0)


def svr_multi(A, t, B):
    lt = np.log(t - 30)
    m_, s_ = lt.mean(), lt.std()
    out = []
    for k, p in PCAS.items():
        va, vb = p.transform((A - mu) / sd) / PSD[k], p.transform((B - mu) / sd) / PSD[k]
        out.append(np.exp(SVR(C=3, gamma=1 / k, epsilon=0.05).fit(va, (lt - m_) / s_).predict(vb) * s_ + m_) + 30)
    return np.mean(out, 0)


def pseudo(first_pass, n_split=2, rounds=1):
    def fp(A, t, B):
        p = first_pass(A, t, B)
        for _ in range(rounds):
            out = np.empty(len(B))
            part = np.arange(len(B)) % n_split
            for k in range(n_split):
                q, c = part == k, part != k
                out[q] = icl_bag(np.vstack([A, B[c]]), np.r_[t, p[c]], B[q])
            p = out
        return p
    return fp


blend_first = lambda A, t, B: 0.725 * icl_bag(A, t, B) + 0.275 * svr_multi(A, t, B)

h("CONTROLS")
run_cv(lambda A, t, B: TabICLRegressor(random_state=SEED + 1, norm_methods=["none"]).fit(A, t).predict(B),
       name="ICL[seed 43] (control-1)", test=False)
s_plain, f_plain, o_plain = run_cv(icl_bag, name="ICL-bag4 (v2 model)")
s_svr, f_svr, o_svr = run_cv(svr_multi, name="SVR-multiPCA")

h("PSEUDO-LABEL VARIANTS (all 4-seed bagged)")
res = {}
for name, fp in [("PL halves", pseudo(icl_bag, 2)),
                 ("PL 4-split", pseudo(icl_bag, 4)),
                 ("blend-PL halves", pseudo(blend_first, 2)),
                 ("blend-PL halves 2 rounds", pseudo(blend_first, 2, rounds=2))]:
    res[name] = run_cv(fp, name=f"ICL-bag4[{name}]")

h("PAIRED PER-FOLD COMPARISON vs ICL-bag4 (negative = better)")
for name, (s, f, o) in res.items():
    d = f - f_plain
    print(f"  {name:26s} RMSE={s:.4f}  mean diff={d.mean():+.4f}  better folds={(d < 0).sum()}/15  worst={d.max():+.3f}")

h("IN THE BLEND: 0.725/0.275-style NNLS with SVR, nested 10-fold, 5 seeds")
from scipy.optimize import nnls
from sklearn.model_selection import KFold


def nested(P):
    r = []
    for sd_ in [42, 7, 123, 2024, 99]:
        out = np.zeros(N)
        for a, b in KFold(10, shuffle=True, random_state=sd_).split(P):
            w, _ = nnls(P[a], y[a])
            out[b] = P[b] @ (w / w.sum())
        r.append(rmse(y, out))
    return np.mean(r), np.std(r)


print(f"  v2 : ICL-bag4 + SVR           nested={nested(np.c_[o_plain, o_svr])[0]:.4f}")
for name, (s, f, o) in res.items():
    m_, s_ = nested(np.c_[o, o_svr])
    print(f"  {name:26s} + SVR  nested={m_:.4f} (sd {s_:.4f})")

h("WHERE DOES IT HELP? improvement by distance to nearest TRAINING row")
best = min(res, key=lambda k: res[k][0])
o_best = res[best][2]
d_nn = np.zeros(N)
for a, b in FOLDS[:5]:
    d_nn[b] = NearestNeighbors(n_neighbors=1).fit(Z_all[a]).kneighbors(Z_all[b])[0][:, 0]
gain = (y - o_plain) ** 2 - (y - o_best) ** 2
df = pd.DataFrame({"d": d_nn, "gain": gain})
df["bin"] = pd.qcut(df.d, 4)
print(df.groupby("bin", observed=True).gain.agg(["mean", "count"]).round(3).to_string())
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].scatter(o_plain, o_best, s=8); ax[0].plot([30, 90], [30, 90], "r--")
ax[0].set_xlabel("ICL-bag4 OOF"); ax[0].set_ylabel(f"{best} OOF"); ax[0].set_title("prediction change")
g = df.groupby("bin", observed=True).gain.mean()
ax[1].bar(range(len(g)), g.values); ax[1].set_xticks(range(len(g)), [str(i) for i in g.index], fontsize=7)
ax[1].set_title("MSE reduction by distance to nearest training row"); ax[1].axhline(0, c="k")
save(fig, "v3_17_pseudo_label.png")
