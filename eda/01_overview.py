"""01 - Overview: shape, integrity, rounding precision, duplicates, ID leakage check, target distribution."""
import numpy as np
from scipy import stats

from common import h, load, plt, save

tr, te, F = load()

h("SHAPE / DTYPES / MISSING")
print("train", tr.shape, "test", te.shape)
print("dtypes:", tr[F].dtypes.value_counts().to_dict())
print("missing train:", int(tr.isna().sum().sum()), "test:", int(te.isna().sum().sum()))
print("record_id unique train/test:", tr.record_id.is_unique, te.record_id.is_unique,
      "| overlap:", len(set(tr.record_id) & set(te.record_id)))

h("ROUNDING PRECISION (decimals per feature)")
def decimals(s):
    return s.astype(str).str.split(".").str[1].str.len().fillna(0).max()
print({c: int(decimals(tr[c])) for c in F[:6]}, "... target:", int(decimals(tr.target)))
nuniq = tr[F].nunique()
print("unique values per feature (min/median/max):", nuniq.min(), int(nuniq.median()), nuniq.max())

h("FEATURE SUMMARY (train)")
d = tr[F].describe().T
d["skew"] = tr[F].skew()
d["kurt"] = tr[F].kurt()
print(d.round(3).to_string())

h("DUPLICATES")
print("exact dup rows (features) train:", tr.duplicated(F).sum(),
      "| test:", te.duplicated(F).sum(),
      "| train-test:", len(tr[F].round(6).merge(te[F].round(6), how="inner")))
print("dup targets in train:", tr.target.duplicated().sum())

h("RECORD_ID vs TARGET (should be meaningless per rules)")
tr_id = tr.record_id.str[1:].astype(int)
te_id = te.record_id.str[1:].astype(int)
print(f"train id range {tr_id.min()}-{tr_id.max()}, test id range {te_id.min()}-{te_id.max()}")
print("union covers 1..478 exactly:", sorted(set(tr_id) | set(te_id)) == list(range(1, 479)))
r, p = stats.spearmanr(tr_id, tr.target)
print(f"spearman(id, target) = {r:.3f} (p={p:.3g})")

h("TARGET DISTRIBUTION")
y = tr.target
print(y.describe().round(3).to_string())
print(f"skew={y.skew():.3f} kurt={y.kurt():.3f}")
print("shapiro p:", f"{stats.shapiro(y).pvalue:.3g}")
print("share of target > 95:", round((y > 95).mean(), 3), "| < 50:", round((y < 50).mean(), 3))
print("constant-mean baseline RMSE (train, in-sample):", round(y.std(ddof=0), 4))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(y, bins=40, color="tab:blue", edgecolor="w"); ax[0].set_title("target histogram")
stats.probplot(y, plot=ax[1]); ax[1].set_title("target QQ-plot (normal)")
ax[2].scatter(tr_id, y, s=8); ax[2].set_title(f"record_id vs target (rho={r:.2f})")
ax[2].set_xlabel("record_id number")
save(fig, "01_target.png")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].bar(range(len(F)), d["std"]); ax[0].set_title("feature std (train)")
ax[1].bar(range(len(F)), d["skew"], color="tab:orange"); ax[1].set_title("feature skew (train)")
for a in ax:
    a.set_xticks(range(len(F)), [c[-2:] for c in F])
save(fig, "01_feature_std_skew.png")
