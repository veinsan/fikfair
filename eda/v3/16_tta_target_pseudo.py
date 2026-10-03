"""16 - Three more 'preprocessing' hypotheses for TabICLv2 (norm=none), each checked on the same 5x3 folds.

A. Test-time augmentation (TTA). The organisers added 'controlled perturbation' to the features. Averaging the
   prediction over K jittered copies of each query row (Gaussian noise of scale sigma in standardized units)
   smooths that noise. Theory: helps if sigma ~ perturbation scale, hurts if sigma blurs real signal.
B. Target transform done RIGHT. v1 found TabICL on log(y-30) worse (7.79 vs 7.63). Suspected cause: back-
   transforming the MEAN of log y gives exp(E[log y]) < E[y] (Jensen) -> systematic under-prediction.
   Fix: ask TabICL for its predictive QUANTILES in transformed space, invert each quantile (monotone map keeps
   quantiles valid), then average -> a proper estimate of E[y]. Tested for log and rank-gauss targets.
C. Transductive pseudo-labelling: add validation rows with their first-pass predictions to the context and
   predict again (mimics using the 96 unlabeled test rows)."""
from cvlib import FOLDS, N, OUT, SEED, X, X_all, h, plt, rmse, run_cv, save, tabicl_fp, y
import numpy as np
import pandas as pd
from scipy import stats
from tabicl import TabICLRegressor

mu, sd = X_all.mean(0), X_all.std(0)
ALPHAS = np.linspace(0.005, 0.995, 199)


def icl(seed=SEED):
    return TabICLRegressor(random_state=seed, norm_methods=["none"])


h("A. TTA JITTER (K=16 copies per query row)")
rng = np.random.default_rng(SEED)
tta_rows = []
for sigma in [0.0, 0.03, 0.06, 0.1, 0.2, 0.3]:
    def fp(A, t, B, s=sigma):
        m = icl().fit(A, t)
        if s == 0:
            return m.predict(B)
        K = 16
        Bj = np.repeat(B, K, 0) + rng.normal(0, s, (len(B) * K, B.shape[1])) * sd
        return m.predict(Bj).reshape(len(B), K).mean(1)
    score, fold, _ = run_cv(fp, name=f"ICL[tta s={sigma}]", test=(sigma == 0.1))
    tta_rows.append((sigma, score))
base_s = tta_rows[0][1]
for s_, r in tta_rows:
    print(f"    sigma={s_:<5} RMSE={r:.4f}  delta={r - base_s:+.4f}")

h("VISUAL CHECK of the jitter: nearest-neighbour distance vs sigma")
from sklearn.neighbors import NearestNeighbors
Z = (X - mu) / sd
d1 = NearestNeighbors(n_neighbors=2).fit(Z).kneighbors(Z)[0][:, 1]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].hist(d1, bins=40, alpha=0.7, label="1-NN distance (train)")
for s_ in [0.06, 0.1, 0.2, 0.3]:
    ax[0].axvline(s_ * np.sqrt(24), ls="--", label=f"jitter norm, sigma={s_}")
ax[0].legend(fontsize=7); ax[0].set_title("jitter size vs neighbour spacing")
ax[1].plot([t[0] for t in tta_rows], [t[1] for t in tta_rows], "o-"); ax[1].set_xlabel("sigma"); ax[1].set_ylabel("CV RMSE")
ax[1].set_title("TTA jitter: CV RMSE vs sigma")
save(fig, "v3_16_tta.png")

h("B. TARGET TRANSFORMS, back-transformed via predictive quantiles")


def fp_transform(fwd, inv, via_quantiles):
    def fp(A, t, B):
        m = icl().fit(A, fwd(t, t))
        if via_quantiles:
            q = m.predict(B, output_type="quantiles", alphas=ALPHAS)
            return inv(np.asarray(q), t).mean(1)
        return inv(m.predict(B), t)
    return fp


log_f = lambda v, t: np.log(v - 30)
log_i = lambda p, t: np.exp(p) + 30


def rg_f(v, t):
    return stats.norm.ppf((stats.rankdata(t) - 0.5) / len(t))


def rg_i(p, t):
    grid = stats.norm.ppf((np.arange(1, len(t) + 1) - 0.5) / len(t))
    return np.interp(p, grid, np.sort(t))


run_cv(lambda A, t, B: icl().fit(A, t).predict(B), name="ICL[y, mean]", test=False)
run_cv(lambda A, t, B: icl().fit(A, t).predict(B, output_type="quantiles", alphas=ALPHAS).mean(1),
       name="ICL[y, mean of quantiles]", test=False)
run_cv(fp_transform(log_f, log_i, False), name="ICL[log y, exp(mean)]  (v1 way)", test=False)
run_cv(fp_transform(log_f, log_i, True), name="ICL[log y, via quantiles]", test=True)
run_cv(fp_transform(rg_f, rg_i, True), name="ICL[rank-gauss y, via quantiles]", test=True)

h("C. TRANSDUCTIVE PSEUDO-LABELLING")


def fp_pseudo(A, t, B):
    # a query row must never see its own pseudo label (TabICL would just copy it) -> cross-halves:
    # pseudo-labelled half 1 joins the context to predict half 2, and vice versa
    p0 = icl().fit(A, t).predict(B)
    out = np.empty(len(B))
    idx = np.arange(len(B)) % 2
    for k in (0, 1):
        other = idx != k
        out[idx == k] = icl(SEED + 1).fit(np.vstack([A, B[other]]), np.r_[t, p0[other]]).predict(B[idx == k])
    return out


run_cv(fp_pseudo, name="ICL[pseudo-label]", test=False)
