"""08 - Where do models fail, which models complement each other, and how much can the public LB be trusted.
Uses OOF from 07. All ensemble weights are estimated with NESTED CV over OOF rows (no optimistic bias)."""
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import nnls
from sklearn.model_selection import KFold

from common import DATA, OUT, ROOT, SEED, h, load, plt, save

tr, te, F = load()
oof = pd.read_csv(OUT / "07_oof.csv")
tp = pd.read_csv(OUT / "07_test_pred.csv")
y = oof.target.values
M = [c for c in oof.columns if c not in ("record_id", "target")]
rmse = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))

h("SINGLE MODELS")
single = pd.Series({m: rmse(oof[m]) for m in M}).sort_values()
print(single.round(4).to_string())

h("RESIDUAL CORRELATION (low = models make different mistakes = ensemble gain)")
R = pd.DataFrame({m: y - oof[m] for m in M})
print(R.corr().round(2).to_string())
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(R.corr(), annot=True, fmt=".2f", cmap="viridis", ax=ax); ax.set_title("OOF residual correlation")
save(fig, "08_residual_corr.png")

h("ENSEMBLES (nested 10-fold over OOF rows)")


def nnls_w(P, t):
    w, _ = nnls(P, t)
    return w / w.sum()


P = oof[M].values
outer = KFold(10, shuffle=True, random_state=SEED)
nested = np.zeros(len(y))
for a, b in outer.split(P):
    nested[b] = P[b] @ nnls_w(P[a], y[a])
w_full = pd.Series(nnls_w(P, y), index=M)
print("NNLS weights (full OOF):\n", w_full[w_full > 0.001].round(3).to_string())
print(f"NNLS nested RMSE = {rmse(nested):.4f}   (in-sample {rmse(P @ w_full.values):.4f})")
for k in [2, 3, 4, 5]:
    top = single.index[:k]
    print(f"mean of top-{k} {list(top)}: RMSE={rmse(oof[top].mean(1)):.4f}")

h("TAIL FIX? linear stretch  y ~ a + b*blend  (b>1 = undo shrinkage toward the mean), nested")
b_full = np.polyfit(nested, y, 1)
print(f"OLS of y on nested blend: slope={b_full[0]:.3f} intercept={b_full[1]:.2f}")
stretched = np.zeros(len(y))
for a, b in outer.split(P):
    w = nnls_w(P[a], y[a])
    inner = np.zeros(len(a))
    for a2, b2 in KFold(5, shuffle=True, random_state=SEED).split(a):
        inner[b2] = P[a][b2] @ nnls_w(P[a][a2], y[a][a2])
    coef = np.polyfit(inner, y[a], 1)
    stretched[b] = np.polyval(coef, P[b] @ w)
print(f"nested RMSE: blend={rmse(nested):.4f}  blend+stretch={rmse(stretched):.4f}")

h("ERROR ANALYSIS of best blend")
blend = P @ w_full.values
res = y - blend
df = pd.DataFrame({"y": y, "pred": blend, "res": res})
df["bin"] = pd.cut(df.y, [30, 40, 50, 60, 70, 80, 101])
g = df.groupby("bin", observed=True).agg(n=("y", "size"), bias=("res", "mean"), rmse=("res", lambda r: np.sqrt(np.mean(r ** 2))))
g["share_of_SSE"] = df.groupby("bin", observed=True).res.apply(lambda r: (r ** 2).sum()) / (res ** 2).sum()
print(g.round(3).to_string())
top5 = np.argsort(-res ** 2)[: int(0.05 * len(y))]
print(f"worst 5% rows ({len(top5)}) carry {100 * (res[top5] ** 2).sum() / (res ** 2).sum():.1f}% of squared error;"
      f" their mean target = {y[top5].mean():.1f}")
print(f"RMSE without worst 5%: {np.sqrt(np.mean(np.delete(res, top5) ** 2)):.3f}")

h("PUBLIC LB NOISE SIMULATION (public ~30% of 96 = ~29 rows)")
rng = np.random.default_rng(SEED)
pub = [np.sqrt(np.mean(rng.choice(res, 29, replace=False) ** 2)) for _ in range(5000)]
prv = [np.sqrt(np.mean(rng.choice(res, 67, replace=False) ** 2)) for _ in range(5000)]
print(f"true OOF RMSE={rmse(blend):.3f} | public-29 RMSE 90% interval = [{np.percentile(pub, 5):.2f}, {np.percentile(pub, 95):.2f}]"
      f" | private-67 90% interval = [{np.percentile(prv, 5):.2f}, {np.percentile(prv, 95):.2f}]")
print("-> a public-LB gain smaller than this spread is noise; select by CV, use LB only as a sanity check.")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].scatter(blend, y, s=8); ax[0].plot([30, 101], [30, 101], "r--"); ax[0].set_xlabel("OOF blend"); ax[0].set_ylabel("target")
ax[0].set_title(f"blend OOF RMSE={rmse(blend):.2f}")
g.rmse.plot.bar(ax=ax[1], color="tab:orange"); ax[1].set_title("RMSE by target bin")
ax[2].hist(pub, bins=50, alpha=0.6, label="public (29 rows)"); ax[2].hist(prv, bins=50, alpha=0.6, label="private (67 rows)")
ax[2].axvline(rmse(blend), c="k"); ax[2].legend(); ax[2].set_title("LB RMSE sampling spread")
save(fig, "08_error_analysis.png")

h("WRITE SUBMISSION (local sanity file; the Kaggle notebook produces the official one)")
tmpl = pd.read_csv(DATA / "submission_template.csv")
pred = tp.set_index("record_id")[M].values @ w_full.values
sub = tmpl[["record_id"]].merge(pd.DataFrame({"record_id": tp.record_id, "prediction": pred}), on="record_id", how="left")
assert len(sub) == len(tmpl) and sub.prediction.notna().all() and sub.record_id.is_unique
sub.to_csv(ROOT / "outputs" / "submission_local_blend.csv", index=False)
print(sub.prediction.describe().round(2).to_string())
print("train OOF blend mean:", blend.mean().round(2), "| test pred mean:", pred.mean().round(2))
