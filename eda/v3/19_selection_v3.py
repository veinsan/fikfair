"""19 - Selection protocol, round 3: every candidate vs the CURRENT submission (v2 = TabICL-bag4 + SVR).
Nested NNLS RMSE over 5 outer seeds, paired bootstrap P(better than v2), LB simulation on 29 (public-like)
and 67 (private-like) random rows, and how different the test predictions are (bounds the public-score move:
|RMSE(a) - RMSE(b)| <= RMS(a - b) on the same rows)."""
from cvlib import N, OUT, SEED, h, load_oof, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold

oof, tst = load_oof()


def w_nnls(P, t):
    w, _ = nnls(P, t)
    return w / w.sum()


def nested(cols, seed):
    P = oof[cols].values
    out = np.zeros(N)
    for a, b in KFold(10, shuffle=True, random_state=seed).split(P):
        out[b] = P[b] @ w_nnls(P[a], y[a])
    return out


CANDS = {
    "v2: ICL-bag4 + SVR": ["ICL-bag4 (v2 model)", "SVR-multiPCA"],
    "PL halves + SVR": ["ICL-bag4[PL halves]", "SVR-multiPCA"],
    "v3: PL 4-split + SVR": ["ICL-bag4[PL 4-split]", "SVR-multiPCA"],
    "PL 8-split + SVR": ["ICL-bag4[PL 8-split]", "SVR-multiPCA"],
    "blend-PL 2 rounds + SVR": ["ICL-bag4[blend-PL halves 2 rounds]", "SVR-multiPCA"],
    "PL 4-split + ICL-bag4 + SVR": ["ICL-bag4[PL 4-split]", "ICL-bag4 (v2 model)", "SVR-multiPCA"],
}
base = nested(CANDS["v2: ICL-bag4 + SVR"], SEED)
v2_test = tst[CANDS["v2: ICL-bag4 + SVR"]].values @ w_nnls(oof[CANDS["v2: ICL-bag4 + SVR"]].values, y)
rng = np.random.default_rng(SEED)
rows = []
for name, cols in CANDS.items():
    runs = [rmse(y, nested(cols, s)) for s in [42, 7, 123, 2024, 99]]
    c = nested(cols, SEED)
    se_c, se_b = (y - c) ** 2, (y - base) ** 2
    boot = np.array([(se_c - se_b)[rng.integers(0, N, N)].mean() for _ in range(4000)])
    win = lambda k: np.mean([se_c[i].mean() < se_b[i].mean() for i in (rng.choice(N, k, replace=False) for _ in range(4000))])
    w = w_nnls(oof[cols].values, y)
    pred = tst[cols].values @ w if all(k in tst for k in cols) else None
    diff = rmse(pred, v2_test) if pred is not None else np.nan
    rows.append((name, np.mean(runs), np.std(runs), (boot < 0).mean(), win(29), win(67), diff,
                 " ".join(f"{k.split('[')[-1][:12]}:{x:.2f}" for k, x in zip(cols, w))))
res = pd.DataFrame(rows, columns=["candidate", "nested", "sd", "P(better)boot", "P(win)pub29", "P(win)prv67",
                                  "RMS test diff vs v2", "weights"])
pd.set_option("display.width", 250)
h("CANDIDATES vs v2")
print(res.round(4).to_string(index=False))
res.to_csv(OUT / "v3_19_selection.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 4))
ax.barh(res.candidate, res.nested, xerr=res.sd); ax.invert_yaxis()
ax.set_xlim(res.nested.min() - 0.1, res.nested.max() + 0.05); ax.axvline(res.nested.iloc[0], c="r", ls="--", label="v2")
ax.legend(); ax.set_title("nested blend RMSE, round 3 candidates")
save(fig, "v3_19_selection.png")
