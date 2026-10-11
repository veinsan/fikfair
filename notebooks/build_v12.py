"""Reuse the audited v11 notebook; replace only residual model and baseline."""
import ast
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
nb=json.loads((ROOT/'notebooks/v11.ipynb').read_text())
cells=nb['cells']
for c in cells:
    c['source']=c['source'].replace('v11','v12').replace('V10','V11').replace('v10','v11')
cells[0]['source']='''# FIK FAIR v12: stability-focused residual correction

Attach official competition data and the actual `submission_v11.csv`; optionally attach `v11_oof.csv` for paired validation. CPU is sufficient. The user reports v11 public RMSE 1.9549491170; it is not used in fitting.

The new component uses boundary-corrected local-linear trends and feature residual × rank interactions. Inner CV chooses between the original and new component. Final weights are fixed: 50% v11, 50% selected component. Permission to use file order is reported by the team, not independently verified. Preserve input order and dependency versions for reproduction.'''
cells[1]['source']=cells[1]['source'].replace("outputs/v12_eda/v11_oof.csv","outputs/stability_search/v11_oof.csv")
head=__import__('pandas').read_csv(ROOT/'outputs/fikfair_v11/submission_v11.csv').prediction.head().tolist()
cells[2]['source']=re.sub(r'expected_head = np.array\(\[.*?\]\)',f'expected_head = np.array({head!r})',cells[2]['source'])
cells[2]['source']=cells[2]['source'].replace('# For this particular v11 run the chosen rule was mean(quantile, sorted).','# Baseline is the exact v11 artifact from the preceding experiment.')
cells[3]['source']='''## Two residual candidates

Both use the same training-only PCA-neighbor rank correction and target curve. The old component removes Gaussian-smoothed feature/target trends. The new component uses local-linear trends, reducing boundary bias, and includes residual feature × centered-rank interactions so coefficients may vary along the ordered file. Ridge alpha stays fixed at 10.

Inner CV chooses one of these two. Neither validation labels nor public scores fit transformations or blend weights. Candidate discovery used the same dataset, so research-selection bias remains.'''
def extract(path,names):
    src=path.read_text()
    return '\n\n'.join(ast.get_source_segment(src,x) for x in ast.parse(src).body if isinstance(x,ast.FunctionDef) and x.name in names)
cells[4]['source']=extract(ROOT/'eda/v9/48_order_refinement.py',['rank','curve'])+'\n\n'+re.sub(r'\bm\.', '',extract(ROOT/'eda/v9/51_v11_rank_eda.py',['neighbor_signals','warp']))
cells[4]['source']+='\n\n'+re.sub(r'\bn\.(?:m\.)?', '',extract(ROOT/'eda/v9/55_stability_search.py',['local_linear','candidates'])).replace('def candidates(','def all_candidates(')
cells[5]['source']="""KEYS = ['gaussian_interaction0_ridge10', 'local_linear_interaction1_ridge10']

def candidates(A,t,B):
    all_p=all_candidates(A,t,B)
    return {k:all_p[k] for k in KEYS}

"""+extract(ROOT/'eda/v9/52_v11_nested.py',['select']).replace('def select(','def select_rule(')
cells[7]['source']=cells[7]['source'].replace('5200 + i','560100 + i')
cells[9]['source']=cells[9]['source'].replace('5299','560199').replace('2.1068098327','1.9549491170')
cells[10]['source']='''## Evidence and limits

Local nested development RMSE: v11 3.3031; fixed 50:50 refinement 3.1789; improvement in 14/15 folds and all three repeats. Record-cluster bootstrap delta interval [-0.1871, -0.0538], conditional on stored predictions.

On 200 additional 96-row query partitions, the new standalone component improved mean RMSE 3.4692 → 3.3586 and simulated private RMSE p95 4.6794 → 4.5641. Those are component comparisons, not the full v11/v12 ensemble. Each query was divided into disjoint 29/67 subsets. These are repeated partitions of the same 382 records, not independent datasets or a forecast of Kaggle private RMSE. Public/private equality cannot be guaranteed.

Same-environment repeat export is checked below; cross-hardware reproducibility and generalization are separate questions. End-to-end reproduction also requires the v9 → v10 → v11 baseline pipeline and its artifacts.'''
for i,c in enumerate(cells):
    c['id']=f'v12-{i:03}'
    if c['cell_type']=='code':
        ast.parse(c['source']);c['execution_count']=None;c['outputs']=[]
(ROOT/'notebooks/v12.ipynb').write_text(json.dumps(nb,indent=1))
print('Built v12:',len(cells),'cells')
