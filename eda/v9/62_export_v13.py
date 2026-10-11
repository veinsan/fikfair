"""Execute v13, verify audit equivalence, save numerical and visual outputs."""
from pathlib import Path
import contextlib
import io
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
tr=pd.read_csv(ROOT/'data/train.csv');audit=np.load(ROOT/'outputs/deep_v13/nested_oof.npz')
frame=tr[['record_id','target']].copy()
for r in range(3):frame[f'v12_rep{r+1}']=audit['v12'][r]
frame.to_csv(ROOT/'outputs/deep_v13/v12_oof.csv',index=False)
nb=json.loads((ROOT/'notebooks/v13.ipynb').read_text());ns={'__name__':'__main__'};counter=0
for i,c in enumerate(nb['cells']):
    if c['cell_type']!='code':continue
    counter+=1;capture=io.StringIO()
    with contextlib.redirect_stdout(capture):exec(compile(c['source'],f'v13_cell_{i}','exec'),ns)
    c['execution_count']=counter;c['outputs']=[{'name':'stdout','output_type':'stream','text':capture.getvalue()}]
    print('Cell',i,'OK',capture.getvalue(),flush=True)
assert np.allclose(ns['rank_oof'],audit['new'],rtol=0,atol=1e-10)
assert np.allclose(ns['predictions']['v13'],audit['v13'],rtol=0,atol=1e-10)
out=ROOT/'outputs/fikfair_v13';(out/'v13_executed.ipynb').write_text(json.dumps(nb,indent=1))
sources=['notebooks/v13.ipynb','notebooks/build_v13.py']+[str(p.relative_to(ROOT)) for p in sorted((ROOT/'eda/v9').glob('*.py'))]
(out/'source_hashes.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},indent=2))
print('All code cells executed; notebook matches audit. Prediction plot saved.')
