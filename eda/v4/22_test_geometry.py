"""22 - Which validation scheme looks like the TEST set? (decides whether pseudo-labelling can be trusted)
21 showed a hard trade-off: PL gains ~0.10 under random KFold but loses ~0.12 under similarity-GroupKFold, and no
uncertainty filter wins both. PL fails when the query pool (a) is far from labelled data and (b) is clustered, so
first-pass errors are correlated and get reinforced. Both are measurable from FEATURES only:
  d_lab  = distance from each query row to its nearest labelled row
  d_pool = distance from each query row to its nearest OTHER query row
We compare the distributions for: test vs train (the real situation), random-fold val vs train, group-fold val vs
train, using the 2-sample KS statistic (0 = identical). Then we evaluate hedged blends
alpha * PL + (1 - alpha) * plain, each blended with the SVR by nested NNLS, under both schemes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, Xte, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVR

mu, sd = X_all.mean(0), X_all.std(0)
Z, Zte = (X - mu) / sd, (Xte - mu) / sd
_, groups = connected_components(NearestNeighbors(radius=1.0).fit(Z).radius_neighbors_graph(Z), directed=False)
GROUP_FOLDS = list(GroupKFold(5).split(X, y, groups))


def geometry(Zlab, Zq):
    d_lab = NearestNeighbors(n_neighbors=1).fit(Zlab).kneighbors(Zq)[0][:, 0]
    d_pool = NearestNeighbors(n_neighbors=2).fit(Zq).kneighbors(Zq)[0][:, 1]
    return d_lab, d_pool


h("GEOMETRY: test vs the two validation schemes")
test_lab, test_pool = geometry(Z, Zte)
geo = {"test": (test_lab, test_pool)}
for name, folds in [("random folds", FOLDS[:5]), ("group folds", GROUP_FOLDS)]:
    L, P = zip(*[geometry(Z[a], Z[b]) for a, b in folds])
    geo[name] = (np.concatenate(L), np.concatenate(P))
rows = []
for name, (dl, dp) in geo.items():
    ks_l = stats.ks_2samp(dl, test_lab).statistic if name != "test" else 0.0
    ks_p = stats.ks_2samp(dp, test_pool).statistic if name != "test" else 0.0
    rows.append((name, np.median(dl), np.median(dp), ks_l, ks_p))
    print(f"  {name:13s} median d_lab={np.median(dl):.3f}  median d_pool={np.median(dp):.3f}  "
          f"KS vs test: d_lab={ks_l:.3f}  d_pool={ks_p:.3f}")
pd.DataFrame(rows, columns=["set", "median_d_lab", "median_d_pool", "KS_d_lab", "KS_d_pool"]).to_csv(
    OUT / "v4_22_geometry.csv", index=False)
print("  note: a query pool of 96 (test) vs ~77 (fold) rows -> d_pool is naturally a bit smaller for test")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for name, (dl, dp) in geo.items():
    ax[0].hist(dl, bins=30, density=True, histtype="step", lw=2, label=name)
    ax[1].hist(dp, bins=30, density=True, histtype="step", lw=2, label=name)
ax[0].set_title("distance to nearest LABELLED row"); ax[1].set_title("distance to nearest other QUERY row")
ax[0].legend(); ax[1].legend()
save(fig, "v4_22_geometry.png")

h("HEDGED BLENDS under both schemes (2-seed TabICL OOF from 21, SVR recomputed per scheme)")
o21 = pd.read_csv(OUT / "v4_21_oof.csv")
PCAS = {k: PCA(k, random_state=SEED).fit((X_all - mu) / sd) for k in (10, 12, 14, 16, 20)}
PSD = {k: PCAS[k].transform((X_all - mu) / sd).std(0) for k in PCAS}


def svr_multi(A, t, B):
    lt = np.log(t - 30)
    m_, s_ = lt.mean(), lt.std()
    return np.mean([np.exp(SVR(C=3, gamma=1 / k, epsilon=0.05).fit(
        p.transform((A - mu) / sd) / PSD[k], (lt - m_) / s_).predict(p.transform((B - mu) / sd) / PSD[k]) * s_ + m_) + 30
        for k, p in PCAS.items()], 0)


def oof_of(fp, folds):
    n_rep = len(folds) // 5
    o = np.zeros((n_rep, N))
    for i, (a, b) in enumerate(folds):
        o[i // 5, b] = fp(X[a], y[a], X[b])
    return o.mean(0)


def nested(P):
    out = []
    for s in [42, 7, 123, 2024, 99]:
        pr = np.zeros(N)
        for a, b in KFold(10, shuffle=True, random_state=s).split(P):
            w, _ = nnls(P[a], y[a])
            pr[b] = P[b] @ (w / w.sum())
        out.append(rmse(y, pr))
    return np.mean(out)


svr = {"random 5x3": oof_of(svr_multi, FOLDS), "group 5": oof_of(svr_multi, GROUP_FOLDS)}
res = []
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    r = {"alpha(PL)": alpha}
    for sch in ["random 5x3", "group 5"]:
        icl = alpha * o21[f"{sch}|PL keep=1.0"] + (1 - alpha) * o21[f"{sch}|plain"]
        r[f"ICL {sch}"] = rmse(y, icl)
        r[f"blend+SVR {sch}"] = nested(np.c_[icl, svr[sch]])
    res.append(r)
res = pd.DataFrame(res)
for sch in ["random 5x3", "group 5"]:
    res[f"gain {sch}"] = res.loc[0, f"blend+SVR {sch}"] - res[f"blend+SVR {sch}"]
print(res.round(4).to_string(index=False))
res.to_csv(OUT / "v4_22_hedge.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
for sch in ["random 5x3", "group 5"]:
    ax.plot(res["alpha(PL)"], res[f"gain {sch}"], "o-", label=sch)
ax.axhline(0, c="k"); ax.set_xlabel("weight on pseudo-labelled TabICL (alpha)"); ax.set_ylabel("blend RMSE gain vs alpha=0")
ax.legend(); ax.set_title("hedging PL under both validation schemes")
save(fig, "v4_22_hedge.png")
