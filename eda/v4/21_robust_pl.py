"""21 - Make pseudo-labelling robust. v3 finding: PL wins under random KFold (7.52 vs 7.61) but LOSES under the
pessimistic similarity-GroupKFold (7.93 vs 7.87).
Hypothesis: confirmation bias. When a whole cluster is far from labelled data, its first-pass labels are wrong in
the same direction, and feeding them back as context reinforces the error.
Theory-driven fix: only pseudo-label rows the model is CONFIDENT about. TabICL returns predictive quantiles, so the
inter-quartile range (IQR) is a per-row uncertainty. Keep the most certain fraction of the query pool as context.
A variant is accepted only if it beats plain TabICL under BOTH validation schemes.
2 random states per model (relative comparison); the chosen one is re-checked with 4 in the notebook."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from tabicl import TabICLRegressor

SEEDS = (0, 1)
Z = (X - X_all.mean(0)) / X_all.std(0)
_, groups = connected_components(NearestNeighbors(radius=1.0).fit(Z).radius_neighbors_graph(Z), directed=False)
SCHEMES = {"random 5x3": FOLDS, "group 5": list(GroupKFold(5).split(X, y, groups))}


def bag(A, t, B, with_iqr=False):
    outs = [TabICLRegressor(random_state=s, norm_methods=["none"]).fit(A, t)
            .predict(B, output_type=["mean", "quantiles"], alphas=[0.25, 0.75]) for s in SEEDS]
    mean = np.mean([o["mean"] for o in outs], 0)
    iqr = np.mean([np.asarray(o["quantiles"])[:, 1] - np.asarray(o["quantiles"])[:, 0] for o in outs], 0)
    return (mean, iqr) if with_iqr else mean


def pl(keep=1.0, n_split=4):
    def fp(A, t, B):
        first, iqr = bag(A, t, B, with_iqr=True)
        confident = iqr <= np.quantile(iqr, keep) if keep < 1 else np.ones(len(B), bool)
        out = np.empty(len(B))
        part = np.arange(len(B)) % n_split
        for k in range(n_split):
            q, c = part == k, (part != k) & confident
            out[q] = bag(np.vstack([A, B[c]]), np.r_[t, first[c]], B[q])
        return out
    return fp


CONFIGS = {"plain": bag, "PL keep=1.0": pl(1.0), "PL keep=0.75": pl(0.75), "PL keep=0.5": pl(0.5)}

rows, oofs = [], {}
for sch, folds in SCHEMES.items():
    h(f"SCHEME: {sch}")
    for name, fp in CONFIGS.items():
        n_rep = len(folds) // 5
        oof = np.zeros((n_rep, N))
        for i, (a, b) in enumerate(folds):
            oof[i // 5, b] = fp(X[a], y[a], X[b])
        score = np.mean([rmse(y, o) for o in oof])
        oofs[(sch, name)] = oof.mean(0)
        rows.append((sch, name, score))
        print(f"  {name:14s} RMSE={score:.4f}", flush=True)

res = pd.DataFrame(rows, columns=["scheme", "config", "rmse"]).pivot(index="config", columns="scheme", values="rmse")
res["gain random"] = res.loc["plain", "random 5x3"] - res["random 5x3"]
res["gain group"] = res.loc["plain", "group 5"] - res["group 5"]
h("SUMMARY (gain = plain - config; a robust variant is positive in BOTH columns)")
print(res.round(4).to_string())
res.to_csv(OUT / "v4_21_robust_pl.csv")
pd.DataFrame({f"{s}|{c}": v for (s, c), v in oofs.items()}).assign(record_id=pd.read_csv(OUT / "v3_oof.csv").record_id,
                                                                    target=y).to_csv(OUT / "v4_21_oof.csv", index=False)

h("UNCERTAINTY SANITY CHECK: is IQR larger where the first-pass error is larger? (random scheme, plain)")
err = np.abs(y - oofs[("random 5x3", "plain")])
a, b = FOLDS[0]
_, iqr_b = bag(X[a], y[a], X[b], with_iqr=True)
print(f"  spearman(IQR, |error|) on fold 0 = {pd.Series(iqr_b).corr(pd.Series(err[b]), method='spearman'):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
res[["gain random", "gain group"]].drop("plain").plot.bar(ax=ax[0]); ax[0].axhline(0, c="k")
ax[0].set_title("RMSE gain over plain TabICL (positive = better)")
ax[1].scatter(iqr_b, err[b], s=10); ax[1].set_xlabel("predictive IQR"); ax[1].set_ylabel("|OOF error|")
ax[1].set_title("uncertainty vs error (fold 0)")
save(fig, "v4_21_robust_pl.png")
