"""37 - Forensics: is the target (nearly) DETERMINISTIC given the features, through some structure we missed?
Context: three teams reach public RMSE 2.3-2.7 while everyone else is >= 7.7. Our variogram suggested an
irreducible RMSE ~3 at most, so a legitimate 2.3 needs a structure that makes y almost a function of x.
Clues already in hand: the SMALLEST-variance principal components correlate with y (PC24, eigenvalue 0.003,
r=0.17), and adding the last 4 PCs cuts linear CV RMSE by 0.33. Under 'private linear projection' with an
ill-conditioned matrix, information about y can sit in tiny-variance directions that look like noise in raw
feature space. Tests:
  F1  significance of corr(PC_k, y) for every PC, permutation null
  F2  linear / additive models in the WHITENED PC basis (every direction equal weight) vs raw basis
  F3  linear model on per-feature uniform ranks (undo a marginal rank transform)
  F4  1-NN target agreement under different metrics (raw standardized vs Mahalanobis/whitened)
  F5  rounding forensics: decimals, digit distribution, any digit/target relation"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from common import OUT, SEED, h, load, plt, save

tr, te, F = load()
X, y = tr[F].values, tr.target.values
X_all = np.vstack([X, te[F].values])
n = len(y)
mu, sd = X_all.mean(0), X_all.std(0)
Z_all = (X_all - mu) / sd
ev, V = np.linalg.eigh(np.cov(Z_all.T))
ev, V = ev[::-1], V[:, ::-1]
P_all = Z_all @ V
W_all = P_all / np.sqrt(ev)
P, W = P_all[:n], W_all[:n]
cv = KFold(10, shuffle=True, random_state=SEED)
rmse = lambda p: float(np.sqrt(np.mean((y - p) ** 2)))

h("F1. corr(PC_k, y) with permutation p-values (k=1 largest variance ... 24 smallest)")
rng = np.random.default_rng(SEED)
r = np.array([stats.pearsonr(P[:, k], y)[0] for k in range(24)])
null = np.array([[abs(np.corrcoef(P[:, k], rng.permutation(y))[0, 1]) for k in range(24)] for _ in range(1000)])
pv = (null >= np.abs(r)).mean(0)
for k in range(24):
    flag = "  <-- p<0.01" if pv[k] < 0.01 else ""
    print(f"  PC{k + 1:2d} eig={ev[k]:8.4f}  r={r[k]:+.3f}  p={pv[k]:.3f}{flag}")
print(f"  share of y-variance explained linearly by PC1-4 vs PC5-24: {np.sum(r[:4] ** 2):.3f} vs {np.sum(r[4:] ** 2):.3f}")

h("F2. linear and additive models: raw basis vs whitened PC basis")
models = {
    "OLS raw": (LinearRegression(), X),
    "OLS whitened PCs": (LinearRegression(), W),
    "Ridge whitened PCs": (RidgeCV(np.logspace(-3, 3, 30)), W),
    "additive splines raw": (make_pipeline(SplineTransformer(n_knots=5), RidgeCV(np.logspace(-2, 4, 30))), X),
    "additive splines whitened PCs": (make_pipeline(SplineTransformer(n_knots=5), RidgeCV(np.logspace(-2, 4, 30))), W),
    "additive splines whitened PC5-24": (make_pipeline(SplineTransformer(n_knots=5), RidgeCV(np.logspace(-2, 4, 30))), W[:, 4:]),
}
for name, (m, A) in models.items():
    print(f"  {name:34s} CV RMSE={rmse(cross_val_predict(m, A, y, cv=cv)):.4f}")

h("F3. linear model on per-feature uniform ranks (undo a marginal rank / bounded transform)")
U = (stats.rankdata(X_all, axis=0) - 0.5) / len(X_all)
for name, A in [("ranks", U[:n]), ("normal scores", stats.norm.ppf(U[:n])), ("logit ranks", np.log(U[:n] / (1 - U[:n])))]:
    print(f"  OLS on {name:14s} CV RMSE={rmse(cross_val_predict(LinearRegression(), A, y, cv=cv)):.4f}")

h("F4. 1-NN / 5-NN target agreement under different metrics (leave-one-out)")
for name, A in [("standardized raw", Z_all[:n]), ("whitened PCs (Mahalanobis)", W),
                ("whitened PC5-24", W[:, 4:]), ("whitened PC1-4", W[:, :4]), ("PC scores unwhitened", P)]:
    for k in [1, 5]:
        p = cross_val_predict(KNeighborsRegressor(k), A, y, cv=KFold(n))
        print(f"  {name:28s} k={k}  LOO RMSE={rmse(p):.4f}")

h("F5. rounding / digit forensics")
s = tr[F].astype(str)
dec = s.apply(lambda c: c.str.split(".").str[1].str.len().fillna(0))
print("  decimals per value (counts over all train cells):", pd.Series(dec.values.ravel()).value_counts().sort_index().to_dict())
last = s.apply(lambda c: c.str[-1].astype(int))
print("  last-digit distribution:", np.bincount(last.values.ravel(), minlength=10))
rho = [stats.spearmanr(last[c], y)[0] for c in F]
print("  max |spearman(last digit of feature, y)| over features:", round(max(abs(x) for x in rho), 3))
ty = pd.Series(y).astype(str).str.split(".").str[1].str.len()
print("  target decimals:", ty.value_counts().to_dict())

fig, ax = plt.subplots(1, 2, figsize=(14, 4))
ax[0].bar(range(1, 25), np.abs(r), color=["tab:red" if p < 0.01 else "tab:blue" for p in pv])
ax[0].set_xlabel("PC (1 = largest variance)"); ax[0].set_ylabel("|corr with y|"); ax[0].set_title("target signal per principal component (red: p<0.01)")
ax2 = ax[0].twinx(); ax2.semilogy(range(1, 25), ev, "k.--", alpha=0.5); ax2.set_ylabel("eigenvalue (log)")
ax[1].scatter(W[:, 23], y, s=6); ax[1].set_xlabel("whitened PC24 (smallest variance)"); ax[1].set_ylabel("target")
ax[1].set_title("target vs the smallest-variance direction")
save(fig, "v8_37_forensics.png")
