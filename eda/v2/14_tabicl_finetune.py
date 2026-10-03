"""14 - Fine-tuning TabICLv2 on the competition data (FinetunedTabICLRegressor, pinball loss, early stopping
on an internal 10% split OF THE TRAINING FOLD -> no leakage into the CV validation fold).
Theory: adapting the prior to this dataset can lower bias. Risk: 382 rows (early-stop split = ~30 rows) -> the
fine-tune may memorise. Compared paired against zero-shot TabICL(norm=none) on the same 5x3 folds."""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from tabicl import FinetunedTabICLRegressor, TabICLRegressor

from common import OUT, SEED, h, load, plt, save

warnings.filterwarnings("ignore")
tr, te, F = load()
X, y = tr[F].values, tr.target.values
cv = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(X))
NORM = ["none"]

CONFIGS = {
    "zero-shot": lambda: TabICLRegressor(random_state=SEED, norm_methods=NORM),
    "ft lr=1e-5": lambda: FinetunedTabICLRegressor(epochs=30, learning_rate=1e-5, device="cpu",
                                                   random_state=SEED, norm_methods=NORM),
    "ft lr=3e-6": lambda: FinetunedTabICLRegressor(epochs=30, learning_rate=3e-6, device="cpu",
                                                   random_state=SEED, norm_methods=NORM),
    "ft lr=1e-5 freeze col+row": lambda: FinetunedTabICLRegressor(epochs=30, learning_rate=1e-5, device="cpu",
                                                                  random_state=SEED, norm_methods=NORM,
                                                                  freeze_col=True, freeze_row=True),
}

h("FINE-TUNE vs ZERO-SHOT (5x3 KFold)")
oofs, rows = {}, []
for name, make in CONFIGS.items():
    t = time.time()
    oof = np.zeros((3, len(y)))
    for i, (a, b) in enumerate(cv):
        oof[i // 5, b] = make().fit(X[a], y[a]).predict(X[b])
    oofs[name] = oof
    fold = [np.sqrt(np.mean((y[b] - oof[i // 5, b]) ** 2)) for i, (a, b) in enumerate(cv)]
    rows.append((name, np.mean([np.sqrt(np.mean((y - o) ** 2)) for o in oof]), fold))
    print(f"  {name:28s} RMSE={rows[-1][1]:.4f}  ({time.time() - t:.0f}s)", flush=True)

h("PAIRED PER-FOLD DIFFERENCE vs zero-shot (negative = fine-tune better)")
base = np.array(rows[0][2])
for name, r, fold in rows[1:]:
    d = np.array(fold) - base
    print(f"  {name:28s} mean diff={d.mean():+.4f}  folds better: {(d < 0).sum()}/15  worst={d.max():+.3f}")

pd.DataFrame({k: v.mean(0) for k, v in oofs.items()}).assign(record_id=tr.record_id, target=y) \
    .to_csv(OUT / "v2_14_finetune_oof.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4))
for name, r, fold in rows[1:]:
    ax.plot(np.array(fold) - base, "o-", label=f"{name} ({r:.3f})")
ax.axhline(0, c="k"); ax.set_xlabel("fold"); ax.set_ylabel("RMSE(ft) - RMSE(zero-shot)")
ax.set_title("fine-tuning vs zero-shot, per fold"); ax.legend(fontsize=8)
save(fig, "v2_14_finetune.png")
