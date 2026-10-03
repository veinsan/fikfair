"""40 - Structured inversion network: fit the hypothesised data-generating process directly.
Hypothesis: x = A * bounded(rank(indicators)) (+ small perturbation), and the target is a composite of the
indicators, y = sum_i a_i * h(s_i) with s = V x + c (V = inverse projection) and h = a monotone 'un-bounding' map.
That is a ONE-hidden-layer network with K units, a shared monotone activation and a linear read-out. Generic
models (TabICL, RealMLP) are not constrained to this form; if the hypothesis is right, the constrained model
should be far more sample-efficient and CV RMSE should drop sharply (toward the ~2 floor of 39).
Variants: activation h in {identity (= linear model), tanh, atanh-like (z/(1-z^2) soft), learnable monotone
piecewise-linear shared across units}, K in {8, 16, 24}. Inputs: whitened PCA-24 of train+test features.
Training: full-batch Adam, weight decay, 5 random restarts, early stopping on an inner 15% split.
CV: repeat-1 folds (5) for the sweep; full 15 folds for anything promising."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
from cvlib import FOLDS, N, OUT, SEED, X, X_all, h, plt, rmse, save, y
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(8)
mu, sd = X_all.mean(0), X_all.std(0)
Z = (X_all - mu) / sd
ev, Vv = np.linalg.eigh(np.cov(Z.T))
WH = lambda A: ((A - mu) / sd) @ Vv / np.sqrt(ev)


class MonoPL(nn.Module):
    """Shared monotone piecewise-linear activation on [-3, 3] (positive slopes via softplus)."""
    def __init__(self, knots=12):
        super().__init__()
        self.edges = torch.linspace(-3, 3, knots + 1)
        self.raw = nn.Parameter(torch.zeros(knots))

    def forward(self, z):
        slopes = nn.functional.softplus(self.raw) + 1e-3
        out = torch.zeros_like(z) + slopes[0] * (z.clamp(max=self.edges[1]) - self.edges[1])
        for k in range(1, len(slopes)):
            lo, hi = self.edges[k], self.edges[k + 1]
            out = out + slopes[k] * (z.clamp(lo, hi) - lo)
        return out


ACT = {
    "identity": lambda: nn.Identity(),
    "tanh": lambda: nn.Tanh(),
    "unbound": lambda: (lambda z: z / (1 - 0.9 * torch.tanh(z) ** 2)),
    "monoPL": lambda: MonoPL(),
}


class Net(nn.Module):
    def __init__(self, k, act):
        super().__init__()
        self.lin = nn.Linear(24, k)
        self.act = ACT[act]()
        self.out = nn.Linear(k, 1)

    def forward(self, x):
        return self.out(self.act(self.lin(x))).squeeze(-1)


def fit_predict(A, t, B, k, act, restarts=5, wd=1e-3, epochs=3000):
    rng = np.random.default_rng(SEED)
    tm, ts = t.mean(), t.std()
    preds = []
    for r in range(restarts):
        torch.manual_seed(SEED + r)
        idx = rng.permutation(len(A))
        nv = int(0.15 * len(A))
        va, trn = idx[:nv], idx[nv:]
        xa = torch.tensor(A, dtype=torch.float32)
        ta = torch.tensor((t - tm) / ts, dtype=torch.float32)
        net = Net(k, act)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=wd)
        best, best_state, patience = 1e9, None, 0
        for ep in range(epochs):
            opt.zero_grad()
            loss = ((net(xa[trn]) - ta[trn]) ** 2).mean()
            loss.backward()
            opt.step()
            if ep % 20 == 0:
                with torch.no_grad():
                    v = ((net(xa[va]) - ta[va]) ** 2).mean().item()
                if v < best - 1e-4:
                    best, best_state, patience = v, {k_: v_.clone() for k_, v_ in net.state_dict().items()}, 0
                else:
                    patience += 1
                    if patience > 15:
                        break
        net.load_state_dict(best_state)
        with torch.no_grad():
            preds.append(net(torch.tensor(B, dtype=torch.float32)).numpy() * ts + tm)
    return np.mean(preds, 0)


h("SWEEP on repeat 1 (5 folds): activation x hidden units")
rows = []
for act in ["identity", "tanh", "unbound", "monoPL"]:
    for k in ([24] if act == "identity" else [8, 16, 24]):
        o = np.zeros(N)
        for a, b in FOLDS[:5]:
            o[b] = fit_predict(WH(X[a]), y[a], WH(X[b]), k, act)
        rows.append((act, k, rmse(y, o)))
        print(f"  act={act:9s} K={k:2d}  CV RMSE={rmse(y, o):.4f}", flush=True)
res = pd.DataFrame(rows, columns=["activation", "K", "rmse"])
res.to_csv(OUT / "v8_40_structured.csv", index=False)
print("\n  references on the same folds: TabICL plain 7.525, linear OLS ~10.7, best blend ~7.2")

fig, ax = plt.subplots(figsize=(7, 4))
for act, d in res.groupby("activation"):
    ax.plot(d.K, d.rmse, "o-", label=act)
ax.axhline(7.525, c="k", ls="--", label="TabICL (same folds)"); ax.axhline(2.3, c="r", ls=":", label="public top-1")
ax.set_xlabel("hidden units K"); ax.set_ylabel("CV RMSE, repeat 1"); ax.legend(); ax.set_title("structured inversion network")
save(fig, "v8_40_structured.png")
