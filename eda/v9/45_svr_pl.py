"""45 - Pseudo-labelling for the SVR member (plateau settings of v6b).
The FM members gained ~0.09 from transductive pseudo-labelling (query rows join the context with first-pass
labels; a row never sees its own label). The SVR - now the member that moves the blend - was only tried with C=3.
First-pass labels come from the FM OOF of the SAME fold (models trained on that fold's training rows only), so
nothing leaks. Variants: pseudo labels from FM-mean, from the v6b-style blend; 4-split; pseudo rows down-weighted.
Judged by the nested blend with FM-mean, pooled and per repeat, vs v6b."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.svm import SVR

mu, sd = X_all.mean(0), X_all.std(0)
DIMS = (10, 12, 14, 16, 20)
PCAS = {k: PCA(k, random_state=SEED).fit((X_all - mu) / sd) for k in DIMS}
PSD = {k: PCAS[k].transform((X_all - mu) / sd).std(0) for k in DIMS}
CELLS = [(0.7, 10), (0.7, 30), (1.0, 10), (1.0, 30)]
L = np.load(OUT / "v7_members.npz")
FM, S6 = L["FM"], L["SVR-plateau (v6b)"]


def svr_plateau(A, t, B, w=None):
    lt = np.log(t - 30)
    m_, s_ = lt.mean(), lt.std()
    out = []
    for scale, C in CELLS:
        for k in DIMS:
            va = PCAS[k].transform((A - mu) / sd) / PSD[k]
            vb = PCAS[k].transform((B - mu) / sd) / PSD[k]
            p = np.exp(SVR(C=C, gamma=scale / k, epsilon=0.05).fit(va, (lt - m_) / s_, sample_weight=w).predict(vb) * s_ + m_) + 30
            out.append(np.clip(p, t.min(), t.max()))
    return np.mean(out, 0)


def nested_w(F, S_):
    w, _ = nnls(np.c_[F, S_], y)
    return w / w.sum()


def svr_pl(first, pw, n_split=4):
    o = np.zeros((3, N))
    for i, (a, b) in enumerate(FOLDS):
        r = i // 5
        lab = first[r][b]
        part = np.arange(len(b)) % n_split
        pred = np.empty(len(b))
        for k in range(n_split):
            q, c = part == k, part != k
            A = np.vstack([X[a], X[b][c]])
            t = np.r_[y[a], lab[c]]
            w = np.r_[np.ones(len(a)), np.full(c.sum(), pw)]
            pred[q] = svr_plateau(A, t, X[b][q], w)
        o[r, b] = pred
    return o


def blend_scores(F, S_):
    def nested(Fx, Sx, seed):
        pr = np.zeros(N)
        for a, b in KFold(10, shuffle=True, random_state=seed).split(Fx):
            w, _ = nnls(np.c_[Fx[a], Sx[a]], y[a])
            pr[b] = np.c_[Fx[b], Sx[b]] @ (w / w.sum())
        return pr
    pooled = np.mean([rmse(y, nested(F.mean(0), S_.mean(0), s)) for s in [42, 7, 123, 2024, 99]])
    per = np.mean([np.mean([rmse(y, nested(F[r], S_[r], s)) for s in [42, 7, 123]]) for r in range(3)])
    return pooled, per


ref = blend_scores(FM, S6)
h(f"v6b reference: pooled {ref[0]:.4f} | per-repeat {ref[1]:.4f}")
w6 = nested_w(FM.mean(0), S6.mean(0))
v6b_first = np.stack([w6[0] * FM[r] + w6[1] * S6[r] for r in range(3)])
rows = []
for src_name, first in [("FM-mean labels", FM), ("v6b-blend labels", v6b_first)]:
    for pw in [1.0, 0.5]:
        o = svr_pl(first, pw)
        sc = blend_scores(FM, o)
        own = np.mean([rmse(y, x) for x in o])
        rc = np.corrcoef(y - o.mean(0), y - FM.mean(0))[0, 1]
        rows.append((src_name, pw, own, rc, sc[0] - ref[0], sc[1] - ref[1]))
        print(f"  {src_name:17s} pseudo weight {pw}: SVR-PL own RMSE={own:.4f} resid corr vs FM={rc:.3f}  "
              f"blend delta pooled {sc[0] - ref[0]:+.4f}  per-repeat {sc[1] - ref[1]:+.4f}", flush=True)
        np.save(OUT / f"v9_45_svrpl_{src_name.split()[0]}_{pw}.npy", o)
pd.DataFrame(rows, columns=["labels", "pseudo_weight", "own_rmse", "corr_FM", "d_pooled", "d_per"]).to_csv(OUT / "v9_45_svr_pl.csv", index=False)
print(f"  (plain SVR-plateau own RMSE {np.mean([rmse(y, x) for x in S6]):.4f}, corr vs FM {np.corrcoef(y - S6.mean(0), y - FM.mean(0))[0, 1]:.3f})")

fig, ax = plt.subplots(figsize=(7, 4))
r = pd.DataFrame(rows, columns=["labels", "pw", "own", "rc", "dp", "dr"])
for lab, d in r.groupby("labels"):
    ax.plot(d.pw, d.dr, "o-", label=f"{lab} (per-repeat)")
    ax.plot(d.pw, d.dp, "s--", label=f"{lab} (pooled)")
ax.axhline(0, c="k"); ax.set_xlabel("weight of pseudo-labelled rows"); ax.set_ylabel("blend delta vs v6b"); ax.legend(fontsize=7)
ax.set_title("SVR with pseudo-labelled context")
save(fig, "v9_45_svr_pl.png")
