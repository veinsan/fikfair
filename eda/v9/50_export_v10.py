"""Export audit artifacts and execute CPU-only notebook code, never v9 training."""
from pathlib import Path
import contextlib
import io
import json
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
tr=pd.read_csv(ROOT/'data/train.csv')
base=np.load(ROOT/'outputs/order_refinement/discovery_predictions.npz')['v9']
saved=tr[['record_id','target']].copy()
for r in range(3): saved[f'v9_rep{r+1}']=base[r]
saved.to_csv(ROOT/'outputs/order_refinement/v9_oof.csv',index=False)

nb_path=ROOT/'notebooks/v10.ipynb'
nb=json.loads(nb_path.read_text())
ns={'__name__':'__main__'}
count=0
for i,cell in enumerate(nb['cells']):
    if cell['cell_type']!='code': continue
    count+=1
    capture=io.StringIO()
    start=time.perf_counter()
    with contextlib.redirect_stdout(capture):
        exec(compile(cell['source'],f'v10_cell_{i}','exec'),ns)
    cell['execution_count']=count
    text=capture.getvalue()
    cell['outputs']=[{'name':'stdout','output_type':'stream','text':text}] if text else []
    print(f'Cell {i} OK ({time.perf_counter()-start:.1f}s)')
    print(text)
# Preserve the run as evidence, while keeping the delivery notebook unexecuted.
out=ROOT/'outputs/fikfair_v10'
(out/'v10_executed.ipynb').write_text(json.dumps(nb,indent=1))

# Compare implemented notebook validation against the separately run audit script.
audit=np.load(ROOT/'outputs/order_refinement/nested_predictions.npz')
assert np.allclose(ns['rank_oof'],audit['nested_rank'],atol=1e-10)
assert np.allclose(ns['predictions']['v10_half_v9_half_rank'],audit['nested_rank_hedge_v9'],atol=1e-10)
print('Notebook and audit script predictions match.')
