"""02 - Train vs test distribution shift: per-feature KS test, overlaid densities, adversarial validation."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

from common import SEED, h, load, plt, save

tr, te, F = load()

h("PER-FEATURE KS TEST train vs test")
rows = []
for c in F:
    ks = stats.ks_2samp(tr[c], te[c])
    rows.append((c, tr[c].mean(), te[c].mean(), tr[c].std(), te[c].std(), ks.statistic, ks.pvalue))
ks = pd.DataFrame(rows, columns=["feat", "mu_tr", "mu_te", "sd_tr", "sd_te", "KS", "p"]).set_index("feat")
print(ks.round(4).to_string())
print("features with p<0.05:", (ks.p < 0.05).sum(), "/", len(F), "(expected by chance ~", round(0.05 * len(F), 1), ")")

fig, axes = plt.subplots(4, 6, figsize=(20, 12))
for a, c in zip(axes.ravel(), F):
    a.hist(tr[c], bins=30, density=True, alpha=0.5, label="train")
    a.hist(te[c], bins=30, density=True, alpha=0.5, label="test")
    a.set_title(f"{c}  KS p={ks.loc[c, 'p']:.2f}", fontsize=9)
axes[0, 0].legend()
save(fig, "02_feature_hist_train_vs_test.png")

h("ADVERSARIAL VALIDATION (can a model tell train from test?)")
X = pd.concat([tr[F], te[F]], ignore_index=True).values
is_test = np.r_[np.zeros(len(tr)), np.ones(len(te))]
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
for name, m in [("logreg", LogisticRegression(max_iter=2000)),
                ("rf", RandomForestClassifier(500, min_samples_leaf=3, random_state=SEED, n_jobs=-1))]:
    p = cross_val_predict(m, X, is_test, cv=cv, method="predict_proba")[:, 1]
    print(f"{name:7s} AUC = {roc_auc_score(is_test, p):.4f}   (0.5 = identical distributions)")
    if name == "rf":
        p_tr = p[: len(tr)]

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(p_tr, bins=30, alpha=0.6, label="train rows")
ax.hist(p[len(tr):], bins=30, alpha=0.6, label="test rows")
ax.set_title("adversarial P(test) (RF, OOF)"); ax.legend()
save(fig, "02_adversarial.png")

r, pv = stats.spearmanr(p_tr, tr.target)
print(f"spearman(P(test-like), target) on train = {r:.3f} (p={pv:.3g})")
