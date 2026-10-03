"""27 - Search for a strong model whose ERRORS differ from the foundation models.
Rounds 3-5 lesson: every in-context foundation model fails on the same rows (residual corr 0.95-0.98), so adding
more of them cannot help. The only member that ever added ensemble value was the SVR (corr 0.84-0.86).
Theory: models trained from scratch with a different inductive bias (MLPs with numerical embeddings, parameter-
efficient ensembles, retrieval nets, learned-metric kernel machines) should decorrelate from in-context learners.
Candidates (pytabkit, TabArena's strongest non-foundation models): RealMLP-TD, TabM, RealTabR, MLP-PLR,
FT-Transformer, ResNet, xRFM. Each: 5x3 CV RMSE, residual corr vs the v5 foundation blend, and nested blend
delta when added to v5 (FM-PL-mean + SVR, nested 7.296).
Usage: python eda/v6/27_nn_diversity.py [names...]"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, Xte, h, plt, rmse, save, y
import numpy as np
import pandas as pd
import pytabkit
from scipy.optimize import nnls
from sklearn.model_selection import KFold

OOF_P, TE_P = OUT / "v6_oof.csv", OUT / "v6_test.csv"
reps = [pd.read_csv(OUT / f"v5_25_oof_rep{r}.csv") for r in (1, 2, 3)]
avg = sum(d[["ICL-PL", "PFN35-PL", "SVR"]] for d in reps) / 3
FM, SVR_ = (0.5 * (avg["ICL-PL"] + avg["PFN35-PL"])).values, avg["SVR"].values
ids = reps[0].record_id


def ptk(cls, **kw):
    return lambda seed: getattr(pytabkit, cls)(device="cpu", random_state=seed, n_cv=1, n_refit=0, verbosity=0, **kw)


ZOO = {
    "RealMLP-TD": ptk("RealMLP_TD_Regressor"),
    "RealMLP-TD-S": ptk("RealMLP_TD_S_Regressor"),
    "TabM-D": ptk("TabM_D_Regressor"),
    "RealTabR-D": ptk("RealTabR_D_Regressor"),
    "MLP-PLR-D": ptk("MLP_PLR_D_Regressor"),
    "FTT-D": ptk("FTT_D_Regressor"),
    "ResNet-D": ptk("Resnet_RTDL_D_Regressor"),
    "xRFM-D": ptk("XRFM_D_Regressor"),
}


def nested(P):
    out = []
    for s in [42, 7, 123, 2024, 99]:
        pr = np.zeros(N)
        for a, b in KFold(10, shuffle=True, random_state=s).split(P):
            w, _ = nnls(P[a], y[a])
            pr[b] = P[b] @ (w / w.sum())
        out.append(rmse(y, pr))
    return float(np.mean(out))


def store(name, oof, pred):
    o = pd.read_csv(OOF_P) if OOF_P.exists() else pd.DataFrame({"record_id": ids, "target": y})
    t = pd.read_csv(TE_P) if TE_P.exists() else pd.DataFrame({"record_id": pd.read_csv(OUT / "v3_test.csv").record_id})
    o[name], t[name] = oof, pred
    o.to_csv(OOF_P, index=False)
    t.to_csv(TE_P, index=False)


base = nested(np.c_[FM, SVR_])
h(f"CANDIDATES (v5 reference: FM-PL-mean + SVR nested = {base:.4f}; SVR alone 8.11, resid corr vs FM "
  f"{np.corrcoef(y - SVR_, y - FM)[0, 1]:.3f})")
rows = []
for name in (sys.argv[1:] or list(ZOO)):
    t0 = time.time()
    try:
        oof = np.zeros((3, N))
        for i, (a, b) in enumerate(FOLDS):
            oof[i // 5, b] = np.asarray(ZOO[name](SEED + i).fit(X[a], y[a]).predict(X[b])).ravel()
        pred = np.asarray(ZOO[name](SEED).fit(X, y).predict(Xte)).ravel()
    except Exception as e:
        print(f"  {name:14s} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
        continue
    score = float(np.mean([rmse(y, o) for o in oof]))
    o = oof.mean(0)
    store(name, o, pred)
    rc = np.corrcoef(y - o, y - FM)[0, 1]
    bl = nested(np.c_[FM, SVR_, o])
    bl2 = nested(np.c_[FM, o])
    rows.append((name, score, rmse(y, o), rc, bl, bl - base, bl2, time.time() - t0))
    print(f"  {name:14s} RMSE={score:.4f} (bagged OOF {rmse(y, o):.4f})  resid-corr vs FM={rc:.3f}  "
          f"v5+this={bl:.4f} ({bl - base:+.4f})  FM+this(no SVR)={bl2:.4f}  [{time.time() - t0:.0f}s]", flush=True)

res = pd.DataFrame(rows, columns=["model", "rmse", "rmse_bagged", "resid_corr_vs_FM", "v5_plus", "delta", "fm_plus", "sec"])
p = OUT / "v6_27_nn.csv"
if p.exists():
    res = pd.concat([pd.read_csv(p), res]).drop_duplicates("model", keep="last")
res.to_csv(p, index=False)
if len(res):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(res.resid_corr_vs_FM, res.rmse)
    for _, r in res.iterrows():
        ax.annotate(f"{r.model} ({r.delta:+.3f})", (r.resid_corr_vs_FM, r.rmse), fontsize=8)
    ax.scatter([np.corrcoef(y - SVR_, y - FM)[0, 1]], [8.11], c="r"); ax.annotate("SVR (in v5)", (np.corrcoef(y - SVR_, y - FM)[0, 1], 8.11), fontsize=8)
    ax.set_xlabel("residual corr vs FM blend (lower = more diverse)"); ax.set_ylabel("CV RMSE (lower = more accurate)")
    ax.set_title("accuracy vs diversity; label = blend delta when added to v5")
    save(fig, "v6_27_nn_diversity.png")
