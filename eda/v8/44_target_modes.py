"""44 - Is the bimodal target driven by a hidden binary 'regime'?
The target has two modes (~37 and ~57) and a floor at 32.6. If a hidden binary variable selects the mode (two
groups scored differently), the error splits into (a) predicting the wrong mode and (b) error within a mode.
  A. Gaussian-mixture decomposition of y: modes, weights, within-mode spread (the error a perfect mode
     classifier + perfect within-mode mean would leave)
  B. how well the features predict the mode (AUC of the v6b blend and of TabICL classifier), and how much of the
     blend's squared error comes from rows whose prediction sits between the modes
  C. two-stage model: P(high mode) from a classifier x mode-specific regressors, vs the blend (OOF)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from tabicl import TabICLClassifier, TabICLRegressor

h("A. mixture decomposition of the target")
for k in [2, 3]:
    g = GaussianMixture(k, random_state=SEED, n_init=10).fit(y[:, None])
    o = np.argsort(g.means_.ravel())
    print(f"  k={k}: means={np.round(g.means_.ravel()[o], 2)} sds={np.round(np.sqrt(g.covariances_.ravel()[o]), 2)} "
          f"weights={np.round(g.weights_[o], 3)}  BIC={g.bic(y[:, None]):.1f}")
g2 = GaussianMixture(2, random_state=SEED, n_init=10).fit(y[:, None])
hi = np.argmax(g2.means_.ravel())
post = g2.predict_proba(y[:, None])[:, hi]
mode = (post > 0.5).astype(int)
within = np.sqrt(np.mean([(y[mode == m] - y[mode == m].mean()) ** 2 for m in (0, 1)][0].tolist() + [(y[mode == 1] - y[mode == 1].mean()) ** 2][0].tolist()))
print(f"  threshold between modes ~ {y[mode == 0].max():.1f} / {y[mode == 1].min():.1f}; rows in high mode: {mode.sum()}")
print(f"  RMSE of 'perfect mode label + mode mean' predictor: {within:.3f}  (vs constant mean {y.std():.3f})")

L = np.load(OUT / "v7_members.npz")
FM, S = L["FM"], L["SVR-plateau (v6b)"]


def nested(F, S_):
    pr = np.zeros(N)
    for a, b in KFold(10, shuffle=True, random_state=SEED).split(F):
        w, _ = nnls(np.c_[F[a], S_[a]], y[a])
        pr[b] = np.c_[F[b], S_[b]] @ (w / w.sum())
    return pr


blend = np.stack([nested(FM[r], S[r]) for r in range(3)]).mean(0)
h("B. can the features tell the mode?")
print(f"  AUC of v6b blend prediction for the high mode: {roc_auc_score(mode, blend):.3f}")
pc = np.zeros(N)
for a, b in FOLDS[:5]:
    pc[b] = TabICLClassifier(random_state=SEED).fit(X[a], mode[a]).predict_proba(X[b])[:, 1]
print(f"  AUC of TabICL classifier (repeat 1): {roc_auc_score(mode, pc):.3f}")
mid = (blend > 44) & (blend < 52)
se = (y - blend) ** 2
print(f"  rows predicted between the modes (44-52): {mid.sum()} rows carry {se[mid].sum() / se.sum():.1%} of the squared error")

h("C. two-stage: P(high) x E[y | high] + (1-P) x E[y | low], OOF on repeat 1")
two = np.zeros(N)
for a, b in FOLDS[:5]:
    p = TabICLClassifier(random_state=SEED).fit(X[a], mode[a]).predict_proba(X[b])[:, 1]
    r_hi = TabICLRegressor(random_state=SEED, norm_methods=["none"]).fit(X[a][mode[a] == 1], y[a][mode[a] == 1]).predict(X[b])
    r_lo = TabICLRegressor(random_state=SEED, norm_methods=["none"]).fit(X[a][mode[a] == 0], y[a][mode[a] == 0]).predict(X[b])
    two[b] = p * r_hi + (1 - p) * r_lo
plain = np.zeros(N)
for a, b in FOLDS[:5]:
    plain[b] = TabICLRegressor(random_state=SEED, norm_methods=["none"]).fit(X[a], y[a]).predict(X[b])
print(f"  plain TabICL repeat-1 RMSE: {rmse(y, plain):.4f}   two-stage: {rmse(y, two):.4f}   mean of both: {rmse(y, (plain + two) / 2):.4f}")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].hist(y, bins=50); ax[0].set_title("target with 2-mode mixture")
xx = np.linspace(30, 101, 300)
ax[0].plot(xx, np.exp(g2.score_samples(xx[:, None])) * len(y) * 1.4, "r")
ax[1].scatter(blend, y, c=mode, s=8, cmap="coolwarm"); ax[1].plot([30, 90], [30, 90], "k--"); ax[1].set_xlabel("v6b OOF"); ax[1].set_title("colour = target mode")
ax[2].hist(pc[mode == 0], bins=30, alpha=0.6, label="low mode"); ax[2].hist(pc[mode == 1], bins=30, alpha=0.6, label="high mode"); ax[2].legend(); ax[2].set_title("classifier P(high mode)")
save(fig, "v8_44_modes.png")
