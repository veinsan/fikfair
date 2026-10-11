"""Run all v12 code cells; assert audit agreement and save executed output."""
from pathlib import Path
import contextlib
import io
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
tr=pd.read_csv(ROOT/'data/train.csv')
audit=np.load(ROOT/'outputs/stability_search/confirmation_oof.npz')
frame=tr[['record_id','target']].copy()
for r in range(3):frame[f'v11_rep{r+1}']=audit['v11'][r]
frame.to_csv(ROOT/'outputs/stability_search/v11_oof.csv',index=False)
nb=json.loads((ROOT/'notebooks/v12.ipynb').read_text());ns={'__name__':'__main__'}
counter=0
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code':continue
    counter+=1;capture=io.StringIO()
    with contextlib.redirect_stdout(capture):exec(compile(c['source'],f'v12_cell_{i}','exec'),ns)
    c['execution_count']=counter;c['outputs']=[{'name':'stdout','output_type':'stream','text':capture.getvalue()}]
    print('Cell',i,'OK',capture.getvalue(),flush=True)
assert np.allclose(ns['rank_oof'],audit['new_component'],rtol=0,atol=1e-10)
assert np.allclose(ns['predictions']['v12'],audit['hedge'],rtol=0,atol=1e-10)
out=ROOT/'outputs/fikfair_v12'
(out/'v12_executed.ipynb').write_text(json.dumps(nb,indent=1))
sources=['notebooks/v12.ipynb','notebooks/build_v12.py','eda/v9/55_stability_search.py',
         'eda/v9/56_stability_confirm.py','eda/v9/57_export_v12.py']
(out/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},indent=2))
print('Notebook predictions agree with separate audit.')
