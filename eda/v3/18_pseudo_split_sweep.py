"""18 - How many pseudo-labelled query rows should join the context? 17 showed halves (50%) 7.547 -> 4-split (75%)
7.522. Extend to 8-split (87.5%) to find where it saturates, then plot RMSE vs pseudo fraction.
CV emulation check: pseudo/labelled ratio in CV = 0.19 (58/306) vs at test time 0.19 (72/382) for 4-split."""
from pathlib import Path

from cvlib import OUT, h, load_oof, plt, rmse, run_cv, save, y
import numpy as np

src = Path(__file__).with_name("17_pseudo_label.py").read_text().split('h("CONTROLS")')[0]
ns = {"__file__": str(Path(__file__).with_name("17_pseudo_label.py"))}
exec(compile(src, "17_defs", "exec"), ns)  # reuse icl_bag / pseudo definitions without re-running 17
pseudo, icl_bag = ns["pseudo"], ns["icl_bag"]

h("8-SPLIT (87.5% of the query set as pseudo-labelled context)")
run_cv(pseudo(icl_bag, 8), name="ICL-bag4[PL 8-split]")

oof, _ = load_oof()
pts = [(0.0, "ICL-bag4 (v2 model)"), (0.5, "ICL-bag4[PL halves]"), (0.75, "ICL-bag4[PL 4-split]"),
       (0.875, "ICL-bag4[PL 8-split]")]
h("RMSE (bagged OOF) vs pseudo fraction")
for f, c in pts:
    print(f"  fraction={f:<6} {c:24s} OOF RMSE={rmse(y, oof[c]):.4f}")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([p[0] for p in pts], [rmse(y, oof[p[1]]) for p in pts], "o-")
ax.set_xlabel("fraction of query rows used as pseudo-labelled context"); ax.set_ylabel("OOF RMSE (TabICL-bag4)")
ax.set_title("pseudo-label density sweep")
save(fig, "v3_18_pseudo_sweep.png")
