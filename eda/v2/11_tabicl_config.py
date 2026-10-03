"""11 - TabICLv2 internal preprocessing knobs + seed bagging.
Only lever left after 10 (residuals unlearnable) is VARIANCE of the best learner. Knobs tested:
  norm_methods (default ['none','power']), outlier_threshold (default 4.0: |z|>4 is clipped; our features reach ~4),
  n_estimators, and averaging several random_state (different feature permutations / ensemble views)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from tabicl import TabICLRegressor

from common import OUT, SEED, h, load, plt, save

tr, te, F = load()
X, y = tr[F].values, tr.target.values
cv = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X))

CONFIGS = {
    "default": {},
    "norm=none": {"norm_methods": ["none"]},
    "norm=power": {"norm_methods": ["power"]},
    "norm=quantile": {"norm_methods": ["quantile"]},
    "norm=none+power+quantile": {"norm_methods": ["none", "power", "quantile"]},
    "norm=robust": {"norm_methods": ["robust"]},
    "outlier=3": {"outlier_threshold": 3.0},
    "outlier=6": {"outlier_threshold": 6.0},
    "outlier=100(off)": {"outlier_threshold": 100.0},
    "shuffle=random": {"feat_shuffle_method": "random"},
    "n_est=16": {"n_estimators": 16},
}


def run(kw, seeds=(SEED,)):
    oof = np.zeros((3, len(y)))
    for i, (a, b) in enumerate(cv):
        oof[i // 5, b] = np.mean([TabICLRegressor(random_state=s, **kw).fit(X[a], y[a]).predict(X[b]) for s in seeds], 0)
    return np.mean([np.sqrt(np.mean((y - o) ** 2)) for o in oof]), oof.mean(0)


h("TabICLv2 CONFIG SWEEP (5x3 KFold, RMSE averaged over repeats)")
rows, oofs = [], {}
for name, kw in CONFIGS.items():
    t = time.time()
    r, oofs[name] = run(kw)
    rows.append((name, r))
    print(f"  {name:26s} RMSE={r:.4f}  ({time.time() - t:.0f}s)", flush=True)

h("SEED BAGGING (default config, 4 random_states averaged)")
# a single random_state can be lucky: compare the two best configs apple-to-apple, both bagged over 4 seeds
for name, kw in [("default x4 seeds", {}), ("norm=none x4 seeds", {"norm_methods": ["none"]}),
                 ("shuffle=random x4 seeds", {"feat_shuffle_method": "random"})]:
    r, oofs[name] = run(kw, seeds=(0, 1, 2, 3))
    rows.append((name, r))
    print(f"  {name:26s} RMSE={r:.4f}", flush=True)

res = pd.DataFrame(rows, columns=["config", "rmse"]).sort_values("rmse")
res.to_csv(OUT / "v2_11_tabicl_config.csv", index=False)
pd.DataFrame(oofs).assign(record_id=tr.record_id, target=y).to_csv(OUT / "v2_11_tabicl_oof.csv", index=False)
print("\n", res.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(res.config, res.rmse); ax.invert_yaxis()
ax.set_xlim(res.rmse.min() - 0.2, res.rmse.max() + 0.1)
ax.axvline(res.set_index("config").loc["default", "rmse"], c="r", ls="--", label="default")
ax.legend(); ax.set_title("TabICLv2 CV RMSE by configuration")
save(fig, "v2_11_tabicl_config.png")
