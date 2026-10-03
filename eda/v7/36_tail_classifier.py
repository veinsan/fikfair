"""36 - Can a dedicated CLASSIFIER find the high tail better than the regression blend?
35: blend prediction ranks y>70 with AUC 0.872; tail rows are predicted 64.7 on average vs true 80.5.
If a classifier trained on the binary question reaches a clearly higher AUC, its probability is new information
for the blend. Thresholds 65 / 70 / 75; TabICL classifier and logistic/SVC baselines; same 15 folds.
Then the honest test: add the out-of-fold probability as a stacking feature (nested ridge on [blend, p])."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tabicl import TabICLClassifier

L = np.load(OUT / "v7_members.npz")
FM, S6 = L["FM"], L["SVR-plateau (v6b)"]


def nested(cols, seed=SEED):
    P = np.column_stack(cols)
    pr = np.zeros(N)
    for a, b in KFold(10, shuffle=True, random_state=seed).split(P):
        w, _ = nnls(P[a], y[a])
        pr[b] = P[b] @ (w / w.sum())
    return pr


blend = np.stack([nested([FM[r], S6[r]]) for r in range(3)])
CLF = {
    "TabICL-clf": lambda: TabICLClassifier(random_state=SEED),
    "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)),
    "SVC-rbf": lambda: make_pipeline(StandardScaler(), SVC(C=3, probability=True, random_state=SEED)),
}
h("AUC for the tail question (per repeat mean)")
P, rows = {}, []
for thr in [65, 70, 75]:
    lab = (y > thr).astype(int)
    ref = np.mean([roc_auc_score(lab, b) for b in blend])
    print(f"  y > {thr} (n={lab.sum()}): regression blend AUC = {ref:.3f}")
    for name, make in CLF.items():
        p = np.zeros((3, N))
        for i, (a, b) in enumerate(FOLDS):
            p[i // 5, b] = make().fit(X[a], lab[a]).predict_proba(X[b])[:, 1]
        auc = np.mean([roc_auc_score(lab, q) for q in p])
        P[(thr, name)] = p
        rows.append((thr, name, auc, auc - ref))
        print(f"      {name:12s} AUC = {auc:.3f} ({auc - ref:+.3f})", flush=True)
pd.DataFrame(rows, columns=["threshold", "classifier", "auc", "delta_vs_blend"]).to_csv(OUT / "v7_36_tail_clf.csv", index=False)

h("STACKING TEST: nested linear regression on [blend, tail probability] vs blend alone (per repeat)")
base = np.mean([rmse(y, b) for b in blend])
print(f"  blend alone: {base:.4f}")
for (thr, name), p in P.items():
    out = []
    for r in range(3):
        F = np.c_[blend[r], p[r]]
        pr = np.zeros(N)
        for a, b in KFold(10, shuffle=True, random_state=SEED).split(F):
            pr[b] = LinearRegression().fit(F[a], y[a]).predict(F[b])
        out.append(rmse(y, pr))
    print(f"  + P(y>{thr}) from {name:12s}: {np.mean(out):.4f} ({np.mean(out) - base:+.4f})")

fig, ax = plt.subplots(figsize=(8, 4))
res = pd.DataFrame(rows, columns=["threshold", "classifier", "auc", "delta"])
for name, d in res.groupby("classifier"):
    ax.plot(d.threshold, d.delta, "o-", label=name)
ax.axhline(0, c="k"); ax.set_xlabel("tail threshold"); ax.set_ylabel("AUC minus regression blend AUC"); ax.legend()
ax.set_title("dedicated tail classifiers vs the regression blend")
save(fig, "v7_36_tail_clf.png")
