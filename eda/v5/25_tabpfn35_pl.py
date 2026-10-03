"""25 - TabPFN-3.5 (newest Prior Labs checkpoint, Sept 2026) with pseudo-labelling, paired against TabICL-PL.
24 found TabPFN-3.5 plain = 7.661 (closest to TabICL's 7.61) but residual corr 0.98 with TabICL-PL, so plain adds
nothing to the v3 blend. Question: does PL lift it past TabICL-PL, or does averaging the two PL models help
(two different pretrained priors behaving like a strong seed bag)?
CPU budget: first repeat only (5 folds), every model on identical folds, paired per-fold differences."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, OUT, SEED, X, X_all, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from tabicl import TabICLRegressor
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

REP = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # which repeat of the 5x3 split
FOLDS5 = FOLDS[5 * REP:5 * REP + 5]
mu, sd = X_all.mean(0), X_all.std(0)
PCAS = {k: PCA(k, random_state=SEED).fit((X_all - mu) / sd) for k in (10, 12, 14, 16, 20)}
PSD = {k: PCAS[k].transform((X_all - mu) / sd).std(0) for k in PCAS}


def icl(A, t, B):
    return np.mean([TabICLRegressor(random_state=s, norm_methods=["none"]).fit(A, t).predict(B) for s in range(4)], 0)


def pfn35(A, t, B):
    return TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, device="cpu", n_estimators=8,
                                                      random_state=SEED).fit(A, t).predict(B)


def svr_multi(A, t, B):
    lt = np.log(t - 30)
    m_, s_ = lt.mean(), lt.std()
    return np.mean([np.exp(SVR(C=3, gamma=1 / k, epsilon=0.05).fit(
        p.transform((A - mu) / sd) / PSD[k], (lt - m_) / s_).predict(p.transform((B - mu) / sd) / PSD[k]) * s_ + m_) + 30
        for k, p in PCAS.items()], 0)


def pl(model, first, A, t, B, n_split=4):
    out = np.empty(len(B))
    part = np.arange(len(B)) % n_split
    for k in range(n_split):
        q, c = part == k, part != k
        out[q] = model(np.vstack([A, B[c]]), np.r_[t, first[c]], B[q])
    return out


cols = ["ICL", "ICL-PL", "PFN35", "PFN35-PL", "PFN35-PL(icl labels)", "SVR"]
P = {c: np.zeros(len(y)) for c in cols}
seen = np.zeros(len(y), bool)
h(f"5 FOLDS (repeat {REP + 1})")
for i, (a, b) in enumerate(FOLDS5):
    A, t, B = X[a], y[a], X[b]
    P["ICL"][b] = icl(A, t, B)
    P["ICL-PL"][b] = pl(icl, P["ICL"][b], A, t, B)
    P["PFN35"][b] = pfn35(A, t, B)
    P["PFN35-PL"][b] = pl(pfn35, P["PFN35"][b], A, t, B)
    P["PFN35-PL(icl labels)"][b] = pl(pfn35, P["ICL"][b], A, t, B)
    P["SVR"][b] = svr_multi(A, t, B)
    seen[b] = True
    print(f"  fold {i + 1}/5  " + "  ".join(f"{c}={rmse(y[b], P[c][b]):.3f}" for c in cols), flush=True)

h(f"RESULTS on the 382 rows of repeat {REP + 1}")
fold_err = {c: np.array([rmse(y[b], P[c][b]) for a, b in FOLDS5]) for c in cols}
for c in cols:
    d = fold_err[c] - fold_err["ICL-PL"]
    print(f"  {c:22s} RMSE={rmse(y, P[c]):.4f}  vs ICL-PL: mean fold diff={d.mean():+.4f} better folds={(d < 0).sum()}/5")
print("  residual corr ICL-PL vs PFN35-PL:", round(np.corrcoef(y - P["ICL-PL"], y - P["PFN35-PL"])[0, 1], 3))

h("SIMPLE BLENDS (fixed weights, no fitting -> no optimism)")
blends = {
    "v3: 0.77 ICL-PL + 0.23 SVR": 0.77 * P["ICL-PL"] + 0.23 * P["SVR"],
    "0.5 ICL-PL + 0.5 PFN35-PL": 0.5 * P["ICL-PL"] + 0.5 * P["PFN35-PL"],
    "0.385/0.385 ICL-PL/PFN35-PL + 0.23 SVR": 0.385 * P["ICL-PL"] + 0.385 * P["PFN35-PL"] + 0.23 * P["SVR"],
    "0.385/0.385 ICL-PL/PFN35-PL(icl) + 0.23 SVR": 0.385 * P["ICL-PL"] + 0.385 * P["PFN35-PL(icl labels)"] + 0.23 * P["SVR"],
}
rows = []
for name, p in blends.items():
    fe = np.array([rmse(y[b], p[b]) for a, b in FOLDS5])
    base = np.array([rmse(y[b], blends["v3: 0.77 ICL-PL + 0.23 SVR"][b]) for a, b in FOLDS5])
    rows.append((name, rmse(y, p), (fe < base).sum()))
    print(f"  {name:46s} RMSE={rmse(y, p):.4f}  folds better than v3: {(fe < base).sum()}/5")
pd.DataFrame(P).assign(record_id=pd.read_csv(OUT / "v3_oof.csv").record_id, target=y).to_csv(OUT / f"v5_25_oof_rep{REP + 1}.csv", index=False)
pd.DataFrame(rows, columns=["blend", "rmse", "folds_better"]).to_csv(OUT / f"v5_25_blends_rep{REP + 1}.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
for c in ["ICL-PL", "PFN35", "PFN35-PL", "PFN35-PL(icl labels)"]:
    ax.plot(range(1, 6), fold_err[c], "o-", label=c)
ax.set_xlabel("fold"); ax.set_ylabel("RMSE"); ax.legend(); ax.set_title("per-fold RMSE, repeat 1")
save(fig, f"v5_25_pfn35_pl_rep{REP + 1}.png")
