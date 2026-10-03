"""29 - Does the ORDER of record_id carry structure? (leakage audit, not a feature hunt)
v1 only checked corr(id, target) = 0.05. A different leak would be serial structure: if consecutive ids were the
same source entity (e.g. one region over several years), rows with adjacent ids would have similar features and
targets, and random K-fold would leak through id-neighbours. Rules say the id has no hidden meaning; we verify.
Tests (all 478 rows ordered by id number, permutation null with 2000 shuffles):
  A. lag-1..5 autocorrelation of the target along id (train rows only, in id order)
  B. mean feature distance between id-adjacent rows vs random pairs
  C. is the train/test assignment random along id? (runs test)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from common import OUT, SEED, h, load, plt, save

tr, te, F = load()
allr = pd.concat([tr.assign(is_test=0), te.assign(is_test=1, target=np.nan)], ignore_index=True)
allr["idn"] = allr.record_id.str[1:].astype(int)
allr = allr.sort_values("idn").reset_index(drop=True)
Z = (allr[F] - allr[F].mean()) / allr[F].std()
rng = np.random.default_rng(SEED)

h("A. target autocorrelation along id order (train rows)")
t = allr.loc[allr.is_test == 0, "target"].values
ac = lambda v, k: np.corrcoef(v[:-k], v[k:])[0, 1]
null = np.array([[ac(rng.permutation(t), k) for k in range(1, 6)] for _ in range(2000)])
for k in range(1, 6):
    a = ac(t, k)
    print(f"  lag {k}: autocorr={a:+.3f}  null 95% band=[{np.percentile(null[:, k - 1], 2.5):+.3f}, "
          f"{np.percentile(null[:, k - 1], 97.5):+.3f}]  p={np.mean(np.abs(null[:, k - 1]) >= abs(a)):.3f}")

h("B. feature distance between id-adjacent rows vs random pairs (all 478 rows)")
d_adj = np.linalg.norm(Z.values[1:] - Z.values[:-1], axis=1)
null_d = np.array([np.linalg.norm(Z.values[p][1:] - Z.values[p][:-1], axis=1).mean()
                   for p in (rng.permutation(len(Z)) for _ in range(2000))])
print(f"  mean adjacent distance={d_adj.mean():.3f}  null mean={null_d.mean():.3f}  "
      f"null 95% band=[{np.percentile(null_d, 2.5):.3f}, {np.percentile(null_d, 97.5):.3f}]  "
      f"p(one-sided smaller)={np.mean(null_d <= d_adj.mean()):.3f}")
print(f"  share of adjacent pairs closer than 1.0: {np.mean(d_adj < 1.0):.3f}")

h("C. train/test assignment along id (Wald-Wolfowitz runs test)")
s = allr.is_test.values
runs = 1 + np.sum(s[1:] != s[:-1])
n1, n0 = s.sum(), len(s) - s.sum()
mu = 2 * n1 * n0 / (n1 + n0) + 1
var = (mu - 1) * (mu - 2) / (n1 + n0 - 1)
print(f"  runs={runs}  expected={mu:.1f}  z={(runs - mu) / np.sqrt(var):+.2f}  (|z|<2 -> random assignment)")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].bar(range(1, 6), [ac(t, k) for k in range(1, 6)])
ax[0].fill_between(range(1, 6), np.percentile(null, 2.5, 0), np.percentile(null, 97.5, 0), alpha=0.3, color="gray")
ax[0].set_title("target autocorrelation along id (gray = null band)"); ax[0].set_xlabel("lag")
ax[1].hist(null_d, bins=40, color="gray", alpha=0.6, label="shuffled ids"); ax[1].axvline(d_adj.mean(), c="r", label="actual")
ax[1].set_title("mean distance between id-adjacent rows"); ax[1].legend()
ax[2].scatter(allr.idn[allr.is_test == 0], t, s=6); ax[2].set_title("target vs id number"); ax[2].set_xlabel("id")
save(fig, "v6_29_id_structure.png")
