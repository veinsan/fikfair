"""08 - Can the organisers' pipeline (rank -> bounded -> private linear projection -> standardize -> perturb ->
round) be inverted? Three tests, each with a theoretical expectation:
  A. standardization scope: mean 0 / sd 1 on train only, or on train+test?
  B. projection rank: if k < 24 indicators were projected, 24-k eigenvalues sit on a flat noise floor.
  C. source recovery: rank+bounded sources are flat (sub-Gaussian, kurtosis<0). Full 24-comp ICA should find them.
  D. sufficient dimension reduction (SIR = monotone effects, SAVE = symmetric/U effects): how many directions
     carry the target, and does modelling in that subspace beat PCA (fitted INSIDE folds - uses y)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy import stats
from sklearn.decomposition import FastICA
from sklearn.model_selection import RepeatedKFold
from sklearn.svm import SVR

from common import SEED, h, load, plt, save
from sdr import SDR, sir_save

tr, te, F = load()
n = len(tr)
X_all = np.vstack([tr[F].values, te[F].values])
y = tr.target.values

h("A. STANDARDIZATION SCOPE")
print("train-only  mean range:", tr[F].mean().min().round(3), tr[F].mean().max().round(3),
      "| sd range:", tr[F].std(ddof=0).min().round(3), tr[F].std(ddof=0).max().round(3))
print("train+test  mean range:", X_all.mean(0).min().round(4), X_all.mean(0).max().round(4),
      "| sd range:", X_all.std(0).min().round(4), X_all.std(0).max().round(4))

h("B. FULL EIGEN SPECTRUM (standardized, train+test)")
ev = np.linalg.eigvalsh(np.cov(X_all.T))[::-1]
print("eigenvalues:", np.round(ev, 4))
print("log10 ratios between consecutive eigenvalues (flat floor -> ~0):", np.round(np.diff(np.log10(ev)), 2))

h("C. FULL 24-COMPONENT ICA (3 contrast functions)")
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for a, fun in zip(ax, ["logcosh", "exp", "cube"]):
    S = FastICA(24, whiten="unit-variance", fun=fun, max_iter=20000, tol=1e-6, random_state=0).fit_transform(X_all)
    ku = np.sort(stats.kurtosis(S, axis=0))
    rho = np.abs([stats.spearmanr(S[:n, j], y)[0] for j in range(24)])
    print(f"{fun:8s} #sources with kurtosis<0: {(ku < 0).sum():2d}/24 | median kurtosis {np.median(ku):.2f} | "
          f"max |spearman(source,y)| {rho.max():.3f}")
    a.bar(range(24), ku); a.axhline(-1.2, c="r", ls="--", label="uniform (bounded) source")
    a.set_title(f"ICA-{fun}: sorted source kurtosis"); a.legend(fontsize=8)
save(fig, "v2_08_ica24_kurtosis.png")

h("D1. SIR / SAVE EIGENVALUES vs PERMUTATION NULL")
e_sir, e_save = sir_save(tr[F].values, y, 10, "sir")[0], sir_save(tr[F].values, y, 10, "save")[0]
rng = np.random.default_rng(SEED)
null_sir = np.array([sir_save(tr[F].values, rng.permutation(y), 10, "sir")[0][:6] for _ in range(200)])
null_save = np.array([sir_save(tr[F].values, rng.permutation(y), 10, "save")[0][:6] for _ in range(200)])
print("SIR  eig :", np.round(e_sir[:6], 3), "\n     null95:", np.round(np.percentile(null_sir, 95, 0), 3))
print("SAVE eig :", np.round(e_save[:6], 3), "\n     null95:", np.round(np.percentile(null_save, 95, 0), 3))
print("significant directions  SIR:", int((e_sir[:6] > np.percentile(null_sir, 95, 0)).sum()),
      "| SAVE:", int((e_save[:6] > np.percentile(null_save, 95, 0)).sum()))

h("D2. SVR(log y) IN THE SDR SUBSPACE vs PCA-12 (SDR fitted inside each fold)")
cv = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(y))
X = tr[F].values
ly = np.log(y - 30)


def cv_sdr(ds, dv):
    e = []
    for a, b in cv:
        s = SDR(ds, dv).fit(X[a], y[a])
        A, B = s.transform(X[a]), s.transform(X[b])
        mu, sd = ly[a].mean(), ly[a].std()
        m = SVR(C=3, gamma=1 / A.shape[1], epsilon=0.05).fit(A, (ly[a] - mu) / sd)
        e.append(np.mean((y[b] - (np.exp(m.predict(B) * sd + mu) + 30)) ** 2))
    return np.sqrt(np.mean(e))


grid = [(2, 0), (3, 0), (4, 0), (0, 4), (3, 3), (4, 4), (6, 6)]
scores = [cv_sdr(*g) for g in grid]
for g, s in zip(grid, scores):
    print(f"  SIR={g[0]} SAVE={g[1]}  RMSE={s:.4f}")
print("  reference: SVR(log) on PCA-12 = 8.13 (06b)")

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].semilogy(range(1, 25), ev, "o-"); ax[0].set_title("full eigen spectrum (no flat floor = full rank)")
ax[1].plot(range(1, 7), e_sir[:6], "o-", label="SIR"); ax[1].plot(range(1, 7), np.percentile(null_sir, 95, 0), "k--", label="null 95%")
ax[1].set_title("SIR eigenvalues"); ax[1].legend()
ax[2].bar([f"{a}/{b}" for a, b in grid], scores); ax[2].axhline(8.13, c="r", ls="--", label="SVR on PCA-12")
ax[2].set_ylim(7.5, max(scores) + 0.3); ax[2].set_title("SVR RMSE in SDR subspace (SIR/SAVE dims)"); ax[2].legend()
save(fig, "v2_08_sdr.png")
