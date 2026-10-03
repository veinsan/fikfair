"""13 - Submission selection protocol (anti public-LB overfitting).
For every candidate blend vs the v1 blend (TabICLv2 + SVR-multiPCA):
  1. nested NNLS blend RMSE under 5 outer seeds (mean +- sd)
  2. paired bootstrap of per-row squared-error difference vs v1 -> P(candidate better)
  3. LB simulation: P(candidate beats v1) on random 29-row (public-like) and 67-row (private-like) subsets
  4. adversarially weighted RMSE: validation rows that look like test rows count more
A candidate replaces v1 only if it wins on 1 AND the bootstrap/private probability is convincing."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from common import OUT, SEED, h, load, plt, save

tr, te, F = load()
y = tr.target.values
oof = pd.read_csv(OUT / "07_oof.csv")
for extra in ["v2_oof.csv", "v2_11_tabicl_oof.csv"]:
    p = OUT / extra
    if p.exists():
        e = pd.read_csv(p)
        assert (e.record_id.values == oof.record_id.values).all()
        cols = [c for c in e.columns if c not in ("record_id", "target")]
        oof = oof.join(e[cols].rename(columns=lambda c: c if extra == "v2_oof.csv" else f"TabICL[{c}]"))
rmse = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))


def nnls_w(P, t):
    w, _ = nnls(P, t)
    return w / w.sum()


def nested(cols, seed):
    P = oof[cols].values
    out = np.zeros(len(y))
    for a, b in KFold(10, shuffle=True, random_state=seed).split(P):
        out[b] = P[b] @ nnls_w(P[a], y[a])
    return out


# adversarial weights: P(test | x) estimated on train+test features, OOF
Xa = np.vstack([tr[F].values, te[F].values])
lab = np.r_[np.zeros(len(tr)), np.ones(len(te))]
pt = cross_val_predict(LogisticRegression(C=0.3, max_iter=3000), Xa, lab,
                       cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[: len(tr), 1]
aw = pt / (1 - pt)
aw /= aw.mean()
print(f"adversarial weights: min={aw.min():.2f} max={aw.max():.2f} (1.0 = no shift)")

CANDIDATES = {
    "v1: TabICL+SVR": ["TabICLv2", "SVR-multiPCA"],
    # single-seed TabICL variants (can be seed-lucky)
    "TabICL[norm=none]+SVR": ["TabICL[norm=none]", "SVR-multiPCA"],
    # apple-to-apple: every TabICL bagged over 4 random_states
    "TabICL[x4]+SVR": ["TabICL[default x4 seeds]", "SVR-multiPCA"],
    "TabICL[norm=none x4]+SVR": ["TabICL[norm=none x4 seeds]", "SVR-multiPCA"],
    "TabICL[shuffle=random x4]+SVR": ["TabICL[shuffle=random x4 seeds]", "SVR-multiPCA"],
    # other model families added to v1
    "TabICL+SVR+TabPFN": ["TabICLv2", "SVR-multiPCA", "TabPFNv2-16"],
    "TabICL+SVR+GP": ["TabICLv2", "SVR-multiPCA", "GP-ARD(log)"],
    "TabICL+SVR+TabDPT": ["TabICLv2", "SVR-multiPCA", "TabDPT"],
    "TabICL[x4]+SVR+TabDPT": ["TabICL[default x4 seeds]", "SVR-multiPCA", "TabDPT"],
    "all FMs+SVR+GP": ["TabICLv2", "TabPFNv2-16", "TabDPT", "SVR-multiPCA", "GP-ARD(log)"],
}
CANDIDATES = {k: v for k, v in CANDIDATES.items() if all(c in oof for c in v)}

h("CANDIDATES")
base = nested(CANDIDATES["v1: TabICL+SVR"], SEED)
rng = np.random.default_rng(SEED)
rows = []
for name, cols in CANDIDATES.items():
    runs = [rmse(nested(cols, s)) for s in [42, 7, 123, 2024, 99]]
    cand = nested(cols, SEED)
    d = (y - cand) ** 2 - (y - base) ** 2
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    sim = lambda k: np.mean([np.sqrt(np.mean(((y - cand) ** 2)[i])) < np.sqrt(np.mean(((y - base) ** 2)[i]))
                             for i in (rng.choice(len(y), k, replace=False) for _ in range(4000))])
    w = nnls_w(oof[cols].values, y)
    rows.append((name, np.mean(runs), np.std(runs), np.sqrt(np.mean(aw * (y - cand) ** 2)),
                 (boot < 0).mean(), sim(29), sim(67), " ".join(f"{c}:{x:.2f}" for c, x in zip(cols, w))))
res = pd.DataFrame(rows, columns=["candidate", "nested_rmse", "sd_seeds", "adv_weighted_rmse",
                                  "P(better)_boot", "P(win)_public29", "P(win)_private67", "weights"])
pd.set_option("display.width", 250)
print(res.round(4).to_string(index=False))
res.to_csv(OUT / "v2_13_selection.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.barh(res.candidate, res.nested_rmse, xerr=res.sd_seeds); ax.invert_yaxis()
ax.set_xlim(res.nested_rmse.min() - 0.15, res.nested_rmse.max() + 0.1)
ax.axvline(res.nested_rmse.iloc[0], c="r", ls="--", label="v1")
ax.legend(); ax.set_title("nested blend RMSE (5 outer seeds)")
save(fig, "v2_13_selection.png")
