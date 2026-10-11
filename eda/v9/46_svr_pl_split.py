"""46 - How many pseudo-labelled query rows should the SVR see? (4-split from 45 vs 2- and 8-split)
For the foundation models the gain saturated at 4-8 splits (round 3). Same check for the SVR member."""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "v3"))
spec = importlib.util.spec_from_file_location("s45", Path(__file__).with_name("45_svr_pl.py"))
src = Path(__file__).with_name("45_svr_pl.py").read_text().split("ref = blend_scores(FM, S6)")[0]
ns = {"__file__": str(Path(__file__).with_name("45_svr_pl.py"))}
exec(compile(src, "45_defs", "exec"), ns)
import numpy as np

FM, S6, svr_pl, blend_scores, h, rmse, y, OUT = (ns[k] for k in ["FM", "S6", "svr_pl", "blend_scores", "h", "rmse", "y", "OUT"])
ref = blend_scores(FM, S6)
h(f"v6b reference: pooled {ref[0]:.4f} | per-repeat {ref[1]:.4f}")
for ns_ in [2, 4, 8]:
    o = svr_pl(FM, 1.0, n_split=ns_)
    sc = blend_scores(FM, o)
    print(f"  n_split={ns_}: SVR-PL own RMSE={np.mean([rmse(y, x) for x in o]):.4f}  blend delta pooled {sc[0] - ref[0]:+.4f}"
          f"  per-repeat {sc[1] - ref[1]:+.4f}", flush=True)
    np.save(OUT / f"v9_46_svrpl_split{ns_}.npy", o)
