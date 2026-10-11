"""Execute v11 CPU diagnostic notebook with recorded outputs and consistency checks."""
from pathlib import Path
import contextlib
import io
import json
import hashlib
import time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
tr=pd.read_csv(ROOT/'data/train.csv')
base=np.load(ROOT/'outputs/order_refinement/nested_predictions.npz')['nested_rank_hedge_v9']
frame=tr[['record_id','target']].copy()
for r in range(3):frame[f'v10_rep{r+1}']=base[r]
frame.to_csv(ROOT/'outputs/v11_eda/v10_oof.csv',index=False)
nb=json.loads((ROOT/'notebooks/v11.ipynb').read_text());ns={'__name__':'__main__'}
counter=0
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code':continue
    counter+=1;capture=io.StringIO();start=time.perf_counter()
    with contextlib.redirect_stdout(capture):exec(compile(c['source'],f'v11_cell_{i}','exec'),ns)
    c['execution_count']=counter;c['outputs']=[{'name':'stdout','output_type':'stream','text':capture.getvalue()}]
    print('Cell',i,'OK',round(time.perf_counter()-start,2),'seconds',flush=True)
    print(capture.getvalue(),flush=True)
audit=np.load(ROOT/'outputs/v11_eda/nested_predictions.npz')
assert np.allclose(ns['rank_oof'],audit['nested_new_rank'],rtol=0,atol=1e-10)
assert np.allclose(ns['predictions']['v11'],audit['v11'],rtol=0,atol=1e-10)
(ROOT/'outputs/fikfair_v11/v11_executed.ipynb').write_text(json.dumps(nb,indent=1))
sources=['notebooks/v11.ipynb','notebooks/build_v11.py',
         'eda/v9/48_order_refinement.py','eda/v9/51_v11_rank_eda.py',
         'eda/v9/52_v11_nested.py','eda/v9/53_v11_residual_check.py','eda/v9/54_export_v11.py']
source_hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
(ROOT/'outputs/fikfair_v11/source_hashes.json').write_text(json.dumps(source_hashes,indent=2))
print('Notebook and audit match. Executed notebook saved.')
