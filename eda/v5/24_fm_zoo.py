"""24 - Foundation-model zoo with and without pseudo-labelling (PL), same 5x3 folds as v1-v4.
Goal: find a model that either beats TabICL-PL or makes DIFFERENT errors (ensemble value). Target to beat:
v3 blend (TabICL-PL + SVR) nested RMSE 7.316.
Models: TabPFN v2 (ungated), TabDPT (ungated) and the gated Prior Labs checkpoints (need a token in
~/.cache/tabpfn/auth_token or TABPFN_TOKEN): v2.5 default / small-samples / real / real-variant / low-skew, v2.6, v3.
Each model is scored plain and with the v3 PL scheme (4-split, a query row never sees its own pseudo label).
Usage: python eda/v5/24_fm_zoo.py [model names...]   (default: all)"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, Xte, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.model_selection import KFold

OOF_P, TE_P = OUT / "v5_oof.csv", OUT / "v5_test.csv"


def tabpfn(ckpt=None, n_est=8, version="v2"):
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    if ckpt is None:
        return lambda: TabPFNRegressor.create_default_for_version(ModelVersion(version), device="cpu",
                                                                  n_estimators=n_est, random_state=SEED)
    return lambda: TabPFNRegressor(model_path=ckpt, device="cpu", n_estimators=n_est, random_state=SEED)


class TabDPT:
    def fit(self, A, t):
        from tabdpt import TabDPTRegressor
        self.m = TabDPTRegressor(device="cpu", compile=False, use_flash=False, verbose=False)
        self.m.fit(A, t)
        return self

    def predict(self, B):
        return self.m.predict(B, n_ensembles=8, seed=SEED)


ZOO = {
    "TabPFN-v2": tabpfn(version="v2"),
    "TabDPT": TabDPT,
    "TabPFN-2.5": tabpfn("tabpfn-v2.5-regressor-v2.5_default.ckpt"),
    "TabPFN-2.5-small": tabpfn("tabpfn-v2.5-regressor-v2.5_small-samples.ckpt"),
    "RealTabPFN-2.5": tabpfn("tabpfn-v2.5-regressor-v2.5_real.ckpt"),
    "RealTabPFN-2.5-var": tabpfn("tabpfn-v2.5-regressor-v2.5_real-variant.ckpt"),
    "TabPFN-2.5-lowskew": tabpfn("tabpfn-v2.5-regressor-v2.5_low-skew.ckpt"),
    "TabPFN-2.6": tabpfn("tabpfn-v2.6-regressor-v2.6_default.ckpt"),
    "TabPFN-3": tabpfn("tabpfn-v3-regressor-v3_default.ckpt"),
    "TabPFN-3.5": tabpfn(version="v3.5"),
    "TabPFN-3.5-fast": tabpfn(version="v3.5-fast"),
}
GATED = {k for k in ZOO if k not in ("TabPFN-v2", "TabDPT")}


def has_token():
    from tabpfn.browser_auth import get_cached_token
    return get_cached_token() is not None


def pl_predict(make, A, t, B, n_split=4):
    first = make().fit(A, t).predict(B)
    out = np.empty(len(B))
    part = np.arange(len(B)) % n_split
    for k in range(n_split):
        q, c = part == k, part != k
        out[q] = make().fit(np.vstack([A, B[c]]), np.r_[t, first[c]]).predict(B[q])
    return out


def cv(fp):
    oof = np.zeros((3, N))
    for i, (a, b) in enumerate(FOLDS):
        oof[i // 5, b] = fp(X[a], y[a], X[b])
    return float(np.mean([rmse(y, o) for o in oof])), oof.mean(0)


def store(name, oof, pred):
    o = pd.read_csv(OOF_P) if OOF_P.exists() else pd.DataFrame({"record_id": pd.read_csv(OUT / "v3_oof.csv").record_id,
                                                                 "target": y})
    t = pd.read_csv(TE_P) if TE_P.exists() else pd.DataFrame({"record_id": pd.read_csv(OUT / "v3_test.csv").record_id})
    o[name], t[name] = oof, pred
    o.to_csv(OOF_P, index=False)
    t.to_csv(TE_P, index=False)


def nested(P):
    out = []
    for s in [42, 7, 123, 2024, 99]:
        pr = np.zeros(N)
        for a, b in KFold(10, shuffle=True, random_state=s).split(P):
            w, _ = nnls(P[a], y[a])
            pr[b] = P[b] @ (w / w.sum())
        out.append(rmse(y, pr))
    return float(np.mean(out))


want = sys.argv[1:] or list(ZOO)
MODES = os.environ.get("ZOO_MODES", "plain,PL").split(",")  # e.g. ZOO_MODES=plain for a fast first pass
token = has_token()
print("Prior Labs token available:", token)
v3 = pd.read_csv(OUT / "v3_oof.csv")
icl_pl, svr = v3["ICL-bag4[PL 4-split]"].values, v3["SVR-multiPCA"].values
base = nested(np.c_[icl_pl, svr])
h(f"ZOO (reference: v3 blend ICL-PL + SVR nested = {base:.4f})")
rows = []
for name in want:
    if name in GATED and not token:
        print(f"  {name:20s} SKIPPED (gated checkpoint, no token)")
        continue
    make = ZOO[name]
    for mode in MODES:
        t0 = time.time()
        fp = (lambda A, t, B: make().fit(A, t).predict(B)) if mode == "plain" else (lambda A, t, B: pl_predict(make, A, t, B))
        try:
            score, oof = cv(fp)
        except Exception as e:  # e.g. licence not accepted, download failure
            print(f"  {name:20s} {mode:5s} FAILED: {type(e).__name__}: {str(e)[:120]}")
            break
        pred = fp(X, y, Xte)
        key = f"{name}[{mode}]"
        store(key, oof, pred)
        rc = np.corrcoef(y - oof, y - icl_pl)[0, 1]
        bl = nested(np.c_[icl_pl, svr, oof])
        rows.append((key, score, rc, bl, bl - base, time.time() - t0))
        print(f"  {key:26s} RMSE={score:.4f}  resid-corr vs ICL-PL={rc:.3f}  "
              f"blend(ICL-PL+SVR+this)={bl:.4f} ({bl - base:+.4f})  [{time.time() - t0:.0f}s]", flush=True)

res = pd.DataFrame(rows, columns=["model", "rmse", "resid_corr_vs_ICLPL", "blend_nested", "delta_vs_v3", "sec"])
path = OUT / "v5_24_zoo.csv"
if path.exists():
    res = pd.concat([pd.read_csv(path), res]).drop_duplicates("model", keep="last")
res.to_csv(path, index=False)
if len(res):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    r = res.sort_values("rmse")
    ax[0].barh(r.model, r.rmse); ax[0].invert_yaxis(); ax[0].axvline(7.522, c="r", ls="--", label="TabICL-PL (v3)")
    ax[0].set_xlim(r.rmse.min() - 0.3, r.rmse.max() + 0.1); ax[0].legend(); ax[0].set_title("single-model CV RMSE")
    ax[1].scatter(r.resid_corr_vs_ICLPL, r.delta_vs_v3)
    for _, row in r.iterrows():
        ax[1].annotate(row.model, (row.resid_corr_vs_ICLPL, row.delta_vs_v3), fontsize=7)
    ax[1].axhline(0, c="k"); ax[1].set_xlabel("residual corr vs TabICL-PL"); ax[1].set_ylabel("blend delta vs v3 (neg = better)")
    ax[1].set_title("accuracy vs diversity")
    save(fig, "v5_24_zoo.png")
