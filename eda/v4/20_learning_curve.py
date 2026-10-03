"""20 - Learning curve: does the pseudo-label (PL) gain shrink as labelled data grows?
Public check: v3 gained +0.015 over v2 on public, CV predicted +0.083 (still inside the 29-row noise band).
Theory: PL helps because labelled data is sparse (v2 variogram). In CV the labelled context is 306 rows, at test
time 382 -> the real test gain could be SMALLER than CV says. We measure plain vs PL TabICL at several labelled
sizes n with the pseudo/labelled ratio held at the test value (0.75*q/n = 72/382 = 0.19, so q = n/4 query rows),
fit RMSE(n) = a + b * n^-c, and extrapolate both curves and the gain to n = 382.
All 15 folds; 2 random states per model to keep CPU time reasonable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, OUT, SEED, X, h, plt, rmse, save, y
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tabicl import TabICLRegressor

SEEDS = (0, 1)
SIZES = [120, 180, 240, 306]


def bag(A, t, B):
    return np.mean([TabICLRegressor(random_state=s, norm_methods=["none"]).fit(A, t).predict(B) for s in SEEDS], 0)


def pl(A, t, B, n_split=4):
    first = bag(A, t, B)
    out = np.empty(len(B))
    part = np.arange(len(B)) % n_split
    for k in range(n_split):
        q, c = part == k, part != k
        out[q] = bag(np.vstack([A, B[c]]), np.r_[t, first[c]], B[q])
    return out


rng = np.random.default_rng(SEED)
rows = []
h("PLAIN vs PL at matched pseudo/labelled ratio (per-fold RMSE on the query subset)")
for n in SIZES:
    q = n // 4
    e0, e1 = [], []
    for a, b in FOLDS:
        a_n = rng.choice(a, n, replace=False) if n < len(a) else a
        b_q = rng.choice(b, min(q, len(b)), replace=False)
        p0 = bag(X[a_n], y[a_n], X[b_q])
        p1 = pl(X[a_n], y[a_n], X[b_q])
        e0.append(np.mean((y[b_q] - p0) ** 2))
        e1.append(np.mean((y[b_q] - p1) ** 2))
    r0, r1 = np.sqrt(np.mean(e0)), np.sqrt(np.mean(e1))
    d = np.sqrt(e0) - np.sqrt(e1)
    rows.append((n, q, r0, r1, r0 - r1, (d > 0).mean()))
    print(f"  n={n:3d} q={q:2d}  plain={r0:.4f}  PL={r1:.4f}  gain={r0 - r1:+.4f}  folds PL better={(d > 0).mean():.2f}",
          flush=True)

res = pd.DataFrame(rows, columns=["n", "q", "plain", "pl", "gain", "frac_folds_better"])
res.to_csv(OUT / "v4_20_learning_curve.csv", index=False)

h("POWER-LAW FIT  RMSE(n) = a + b * n^-c  and extrapolation to n = 382")
f = lambda n, a, b, c: a + b * np.power(n, -c)
fits = {}
for col in ["plain", "pl"]:
    try:
        p, _ = curve_fit(f, res.n, res[col], p0=[6, 50, 0.5], maxfev=20000)
    except RuntimeError:
        p = None
    fits[col] = p
    if p is not None:
        print(f"  {col:5s}: a={p[0]:.3f} b={p[1]:.2f} c={p[2]:.3f}  -> RMSE(306)={f(306, *p):.4f}  RMSE(382)={f(382, *p):.4f}")
lin = np.polyfit(np.log(res.n), res.gain, 1)
print(f"  gain vs log(n) slope = {lin[0]:+.4f} per log-unit -> gain extrapolated to n=382: "
      f"{np.polyval(lin, np.log(382)):+.4f} (CV at n=306: {res.gain.iloc[-1]:+.4f})")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
nn = np.linspace(100, 420, 100)
for col, c in [("plain", "tab:blue"), ("pl", "tab:orange")]:
    ax[0].plot(res.n, res[col], "o", c=c, label=col)
    if fits[col] is not None:
        ax[0].plot(nn, f(nn, *fits[col]), "--", c=c)
ax[0].axvline(382, c="k", ls=":", label="test-time n=382"); ax[0].set_xlabel("labelled context size n")
ax[0].set_ylabel("RMSE"); ax[0].legend(); ax[0].set_title("learning curve")
ax[1].plot(res.n, res.gain, "o-"); ax[1].plot(nn, np.polyval(lin, np.log(nn)), "r--")
ax[1].axvline(382, c="k", ls=":"); ax[1].axhline(0, c="k"); ax[1].set_title("PL gain vs n"); ax[1].set_xlabel("n")
save(fig, "v4_20_learning_curve.png")
