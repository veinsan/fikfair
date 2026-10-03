"""34 - Context subagging for in-context learners.
Observation (all rounds): repeat-averaged OOF is ~0.16-0.19 RMSE better than single-repeat OOF (TabICL-PL 7.33 vs
7.52). A repeat-average = mean of 3 models that each saw a DIFFERENT 80% of the data -> the variance due to which
rows are in the context is large (seed variance is only ~0.02).
Theory (subagging, Buhlmann & Yu): averaging a learner over random subsamples removes that variance, at the cost
of bias from smaller contexts (learning curve: fewer labelled rows = worse).
Test INSIDE each training fold, so nothing leaks: single model on 100% of the fold vs the average of M models on
random fractions f of the fold. TabICL (norm=none), seed varies with the subsample, 15 folds."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from tabicl import TabICLRegressor

M = 8
CONFIGS = {"single 100%": None, "subag f=0.9": 0.9, "subag f=0.8": 0.8, "subag f=0.65": 0.65}


def fit_pred(A, t, B, seed):
    return TabICLRegressor(random_state=seed, norm_methods=["none"]).fit(A, t).predict(B)


oof = {c: np.zeros((3, N)) for c in CONFIGS}
oof["seed-bag 100% x8"] = np.zeros((3, N))
curve = {f: np.zeros((M, 3, N)) for f in (0.9, 0.8, 0.65)}
h(f"15 folds, M={M}")
for i, (a, b) in enumerate(FOLDS):
    rng = np.random.default_rng(SEED + i)
    oof["single 100%"][i // 5, b] = fit_pred(X[a], y[a], X[b], 0)
    oof["seed-bag 100% x8"][i // 5, b] = np.mean([fit_pred(X[a], y[a], X[b], s) for s in range(M)], 0)
    for f in (0.9, 0.8, 0.65):
        ps = []
        for m in range(M):
            sub = rng.choice(a, int(f * len(a)), replace=False)
            ps.append(fit_pred(X[sub], y[sub], X[b], m))
            curve[f][m, i // 5, b] = ps[-1]
        oof[f"subag f={f}"][i // 5, b] = np.mean(ps, 0)
    print(f"  fold {i + 1}/15", flush=True)

h("RESULT (RMSE averaged over the 3 repeats = single-fit view, the honest one for the final model)")
rows = []
base = np.array([rmse(y[b], oof["single 100%"][i // 5, b]) for i, (a, b) in enumerate(FOLDS)])
for c, o in oof.items():
    per = np.mean([rmse(y, r) for r in o])
    fold = np.array([rmse(y[b], o[i // 5, b]) for i, (a, b) in enumerate(FOLDS)])
    rows.append((c, per, rmse(y, o.mean(0)), (fold < base).sum()))
    print(f"  {c:18s} per-repeat RMSE={per:.4f} ({per - rows[0][1]:+.4f})  pooled={rmse(y, o.mean(0)):.4f}  "
          f"folds better than single: {(fold < base).sum()}/15")
pd.DataFrame(rows, columns=["config", "per_repeat", "pooled", "folds_better"]).to_csv(OUT / "v7_34_subagging.csv", index=False)
np.savez(OUT / "v7_34_oof.npz", **{k.replace(" ", "_"): v for k, v in oof.items()})

h("HOW MANY SUBSAMPLES ARE NEEDED? (per-repeat RMSE of the first m members)")
fig, ax = plt.subplots(figsize=(7, 4))
for f, arr in curve.items():
    vals = [np.mean([rmse(y, arr[:m].mean(0)[r]) for r in range(3)]) for m in range(1, M + 1)]
    print(f"  f={f}: " + " ".join(f"m={m}:{v:.3f}" for m, v in zip(range(1, M + 1), vals)))
    ax.plot(range(1, M + 1), vals, "o-", label=f"subsample fraction {f}")
ax.axhline(rows[0][1], c="k", ls="--", label="single model, 100% context")
ax.set_xlabel("number of subsampled contexts averaged"); ax.set_ylabel("per-repeat RMSE"); ax.legend()
ax.set_title("context subagging, TabICL")
save(fig, "v7_34_subagging.png")
