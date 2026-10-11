"""Build a standalone, CPU-only nested rank/residual refinement notebook."""
import ast
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
nb=json.loads((ROOT/'notebooks/v10.ipynb').read_text())
cells=nb['cells']
cells[0]['source']='''# FIK FAIR v11: rank correction with feature residuals

The team reports public RMSE 2.1068098327 for v10. This experiment tests whether features explain departures from the ordered target trend, and averages a selected rank/residual estimator with v10 at fixed 50:50 weights. No leaderboard score is used in fitting.

Attach official competition files and the actual `submission_v10.csv`. Optionally attach `v10_oof.csv` to reproduce paired comparisons. CPU is sufficient. For a strict rerun, preserve file order and use the library versions saved in `requirements-v11.txt`. The organiser permission for order use is reported by the team.'''
cells[1]['source']=cells[1]['source'].replace('V9','V10').replace('v9','v10').replace("rglob('submission.csv')","rglob('submission_v10.csv')").replace("root / 'results/v10/submission.csv'","root / 'outputs/fikfair_v10/submission_v10.csv'").replace("root / 'outputs/order_refinement/v10_oof.csv'","root / 'outputs/v11_eda/v10_oof.csv'").replace("OUT = root / 'outputs/fikfair_v10'","OUT = root / 'outputs/fikfair_v11'")
cells[1]['source']+='''
from sklearn.linear_model import Ridge
import scipy
import sklearn
import sys
import hashlib
import platform
VERSIONS = {'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__}
print('Environment:', VERSIONS)
'''
cells[2]['source']=cells[2]['source'].replace('V9','V10').replace('v9','v10').replace('[92.368574, 82.252801, 77.549386, 73.350033, 70.529185]','[93.291060, 84.052009, 78.284577, 73.876091, 71.114928]')
cells[3]['source']='''## Candidate models

Nine candidates test PCA dimension averaging, subsampling, a Brownian-bridge rank prior, and regularized feature residuals. Residual models subtract smooth position trends from training features and targets, fit Ridge to the remaining variation, and predict deviations around the query rank trend. Everything is fitted within the relevant training partition.

The choice between candidates uses inner five-fold validation. The candidate catalogue was designed after exploring this dataset, so research-selection bias is not eliminated by nested evaluation.'''
def extract(path,names):
    src=path.read_text();tree=ast.parse(src)
    return '\n\n'.join(ast.get_source_segment(src,node) for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names)
extra=re.sub(r'\bm\.', '', extract(ROOT/'eda/v9/51_v11_rank_eda.py',['neighbor_signals','warp','all_candidates']))
extra+='\n\n'+re.sub(r'\bn\.(?:m\.)?', '', extract(ROOT/'eda/v9/53_v11_residual_check.py',['residual_candidates']))
cells[4]['source']+='\n\n'+extra
keys=['gaussian_dim8','gaussian_dim12','dimension_mean','bridge_0.5','subbag16','ridge10_w0.5','ridge10_w1.0','ridge100_w0.5','ridge100_w1.0']
cells[5]['source']='KEYS = '+repr(keys)+'\n\n'+extract(ROOT/'eda/v9/52_v11_nested.py',['candidates','select']).replace('n.all_candidates','all_candidates').replace('res.residual_candidates','residual_candidates').replace('def select(','def select_rule(')
cells[6]['source']='''## Nested validation

The 15 outer folds match the archived v10 predictions. For each fold, candidate selection uses only the outer training rows. All transforms, smoothing trends and residual models are fitted there. The final blend weight remains 50:50.

Optional v10 OOF is the prior nested local audit baseline, not a claim of bitwise identical Kaggle predictions. The baseline and candidate catalogue were developed on this same dataset; results remain development estimates.'''
cells[7]['source']=cells[7]['source'].replace('100 + i','5200 + i').replace("predictions['quantile']","curve(y[a], rank(len(b)))").replace('v9','v10').replace("predictions['v10_half_v10_half_rank']","predictions['v11']")
# The previous notebook names its final blend v10_half_v9_half_rank.
cells[7]['source']=cells[7]['source'].replace("predictions['v10_half_v10_half_rank']","predictions['v11']")
cells[8]['source']='''## Export and provenance

Choose one candidate using five-fold CV on all training rows, then average its test prediction with the exact v10 input file. A manifest records source-file hashes, dependency versions, selected model, seed, and output hash. The export is rerun to verify byte-identical CSV output in this environment. No submission is uploaded automatically.'''
cells[9]['source']='''selected, inner_scores = select_rule(X, y, 5299)
new_test = candidates(X, y, Xte)[selected]
final_test = .5 * (v10_test + new_test)
assert np.isfinite(final_test).all()
assert set(template.record_id) == set(te.record_id) and template.record_id.is_unique

def submission_frame(p):
    result = template[['record_id']].merge(
        pd.DataFrame({'record_id': te.record_id, 'prediction': p}),
        on='record_id', how='left', validate='one_to_one')
    assert len(result) == len(te) and np.isfinite(result.prediction).all()
    return result

result = submission_frame(final_test)
csv_text = result.to_csv(index=False)
output_path = OUT / 'submission_v11.csv'
output_path.write_text(csv_text)
repeat_selected, _ = select_rule(X, y, 5299)
repeat_new = candidates(X, y, Xte)[repeat_selected]
repeat_text = submission_frame(.5 * (v10_test + repeat_new)).to_csv(index=False)
assert selected == repeat_selected and csv_text == repeat_text

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

manifest = {
    'selected': selected, 'full_training_cv': inner_scores,
    'v10_weight': .5, 'new_model_weight': .5, 'selection_seed': 5299,
    'versions': VERSIONS, 'platform': platform.platform(),
    'input_sha256': {str(p): sha256_file(p) for p in [DATA_DIR/'train.csv', DATA_DIR/'test.csv',
                     DATA_DIR/'submission_template.csv', V10_PATH]},
    'submission_sha256': sha256_file(output_path),
    'same_environment_repeat_identical': True,
    'v10_public_reported_by_user': 2.1068098327, 'v11_public': None,
}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
(OUT / 'requirements-v11.txt').write_text('\\n'.join(f'{k}=={v}' for k,v in VERSIONS.items() if k != 'python')+'\\n')
changes = pd.DataFrame({'record_id': te.record_id, 'v10': v10_test,
                       'new_model': new_test, 'v11': final_test, 'change': final_test-v10_test})
changes.to_csv(OUT / 'prediction_changes.csv', index=False)
print('Selected:', selected)
print('Repeated export is byte-identical:', True)
print('RMS prediction change:', np.sqrt(np.mean((final_test-v10_test)**2)))
print('Submission SHA256:', manifest['submission_sha256'])
print(result.head().to_string(index=False))
'''
cells[10]['source']='''The local nested audit gave mean fold RMSE 3.4008 for v10 and 3.3031 for v11. All three repeats improved. The paired bootstrap conditions on the stored predictions and does not cover all method-selection uncertainty. These validation gains do not guarantee a lower public score. CPU execution is sufficient; foundation models are not retrained.'''
cells[11]['source']=cells[11]['source'].replace('v9_test','v10_test').replace('rank_test','new_test').replace("label='v9'","label='v10'").replace("label='v10')","label='v11')").replace("label='rank correction'","label='rank + residual'").replace("ylabel='v10 minus v9'","ylabel='v11 minus v10'").replace("'v10_predictions.png'","'v11_predictions.png'")
# Keep the baseline and final plot labels distinct.
cells[11]['source']=cells[11]['source'].replace("axes[0].plot(rank(len(te)), v10_test, label='v11')","axes[0].plot(rank(len(te)), v10_test, label='v10')")
for i,cell in enumerate(cells):
    cell['id']=f'v11-{i:03}'
    if cell['cell_type']=='code':
        ast.parse(cell['source']);cell['execution_count']=None;cell['outputs']=[]
(ROOT/'notebooks/v11.ipynb').write_text(json.dumps(nb,indent=1))
print('Wrote v11 notebook:',len(cells),'cells')
