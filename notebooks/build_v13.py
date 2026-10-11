"""Build v13 from the working v12 delivery without changing old artifacts."""
import ast
import json
import re
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
nb=json.loads((ROOT/'notebooks/v12.ipynb').read_text());cells=nb['cells']
for c in cells:c['source']=c['source'].replace('v12','v13').replace('V11','V12').replace('v11','v12')
cells[0]['source']='''# FIK FAIR v13: nonlinear residual correction

Attach official competition data and the audited `submission_v12.csv`. Optionally attach `v12_oof.csv` for paired comparisons. CPU is sufficient. Public v12 RMSE 1.8247135700 is user-reported, not a training objective. Use of file order is reported as permitted by the organiser; this remains an order-dependent method.'''
cells[1]['source']=cells[1]['source'].replace('outputs/stability_search/v12_oof.csv','outputs/deep_v13/v12_oof.csv')+'\nfrom sklearn.kernel_ridge import KernelRidge\n'
head=pd.read_csv(ROOT/'outputs/fikfair_v12/submission_v12.csv').prediction.head().tolist()
cells[2]['source']=re.sub(r'expected_head = np.array\(\[.*?\]\)',f'expected_head = np.array({head!r})',cells[2]['source'])
cells[3]['source']='''## Preprocessing ablations

Twenty candidates test trend bandwidth, regularization, rank interactions, log targets, residual scaling, nonlinear residual prediction, target smoothing, and clipping. Eight kernel variants were added after development sensitivity analysis. All preprocessing is fitted inside the relevant training partition. Inner CV selects a candidate; the final blend is fixed at 50% v12 and 50% selected component. Research-selection bias is not eliminated.'''
def extract(path,names):
    src=path.read_text()
    return '\n\n'.join(ast.get_source_segment(src,x) for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name in names)
cells[4]['source']=extract(ROOT/'eda/v9/48_order_refinement.py',['rank','curve'])+'\n\n'+re.sub(r'\bm\.', '',extract(ROOT/'eda/v9/51_v11_rank_eda.py',['neighbor_signals','warp']))
cells[4]['source']+='\n\n'+re.sub(r'\bn\.(?:m\.)?', '',extract(ROOT/'eda/v9/55_stability_search.py',['local_linear']))
cells[4]['source']+='\n\n'+re.sub(r'\bs\.(?:n\.(?:m\.)?)?', '',extract(ROOT/'eda/v9/59_preprocessing_ablation.py',['predict']))
src=(ROOT/'eda/v9/59_preprocessing_ablation.py').read_text()
config=next(ast.get_source_segment(src,x) for x in ast.parse(src).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIGS' for t in x.targets))
cells[5]['source']=config+'''
CONFIGS.update({f'kernel_a{alpha}_g{gamma}':{'kernel':True,'kernel_alpha':alpha,'gamma_scale':gamma}
               for alpha in [.1,.3,1] for gamma in [1,2,4] if (alpha,gamma)!=(1,1)})
KEYS=list(CONFIGS)

def candidates(A,t,B):
    return {k:predict(A,t,B,c) for k,c in CONFIGS.items()}

'''+extract(ROOT/'eda/v9/52_v11_nested.py',['select']).replace('def select(','def select_rule(')
cells[7]['source']=cells[7]['source'].replace('560100 + i','6000 + i')
cells[9]['source']=cells[9]['source'].replace('560199','6099').replace('1.9549491170','1.8247135700')
cells[10]['source']='''## Evidence and limitations

Nested development RMSE: v12 3.1789, v13 3.0156. All three repeats and 15/15 folds improved. Conditional record-cluster bootstrap delta interval: [-0.2369, -0.0726]. Methods were designed on this same dataset, so research-selection bias remains.

Near-neighbor group stress favored the kernel component, but proximity groups are not verified real-world entities. Neither nested CV nor reproduction proves absence of all leakage or guarantees public/private equality. File order is a dataset-construction signal and must be disclosed. The inherited baseline includes pseudo-labeling. End-to-end reproduction needs the ancestor pipelines, not only this notebook.'''
for i,c in enumerate(cells):
    c['id']=f'v13-{i:03}'
    if c['cell_type']=='code':ast.parse(c['source']);c['execution_count']=None;c['outputs']=[]
(ROOT/'notebooks/v13.ipynb').write_text(json.dumps(nb,indent=1));print('Built v13 notebook')
