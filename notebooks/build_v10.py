"""Build a CPU-only refinement notebook without changing the executed v9."""
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def md(s):
    cells.append({'cell_type':'markdown','metadata':{},'source':s})
def code(s):
    ast.parse(s)
    cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s})

md('''# FIK FAIR v10: feature-guided rank correction

The v9 run scored 2.3865618144 on the public leaderboard, as reported by the team. This refinement uses feature similarity to adjust the mapping between positions in two separately ordered files. The main candidate averages this rank model with the existing v9 submission. It does not retrain foundation models.

Attach the competition data and the **actual v9 submission.csv** as notebook inputs. Optionally attach `v9_oof.csv` from `outputs/order_refinement` to reproduce the paired validation comparison. CPU is sufficient; no GPU, internet or authentication token is needed. Prior organiser approval for using file order is reported by the team, not independently verified here.''')
code('''from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import betainc
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, RepeatedKFold

ON_KAGGLE = Path('/kaggle/input').exists()
# Set explicit paths here if multiple candidate inputs are attached.
DATA_DIR = None
V9_PATH = None
V9_OOF_PATH = None
if ON_KAGGLE:
    input_root = Path('/kaggle/input')
    if DATA_DIR is None:
        paths = []
        for path in input_root.rglob('train.csv'):
            columns = pd.read_csv(path, nrows=0).columns
            if {'record_id', 'target', 'feature_24'} <= set(columns) and (path.parent / 'test.csv').exists():
                paths.append(path.parent)
        if len(paths) != 1:
            raise ValueError(f'Set DATA_DIR explicitly; found {paths}')
        DATA_DIR = paths[0]
    if V9_PATH is None:
        paths = list(input_root.rglob('submission.csv'))
        if len(paths) != 1:
            raise ValueError(f'Attach only the v9 submission, or set V9_PATH explicitly: {paths}')
        V9_PATH = paths[0]
    if V9_OOF_PATH is None:
        paths = list(input_root.rglob('v9_oof.csv'))
        if len(paths) == 1:
            V9_OOF_PATH = paths[0]
    OUT = Path('/kaggle/working')
else:
    root = Path.cwd()
    if not (root / 'data/train.csv').exists():
        root = root.parent
    DATA_DIR = Path(DATA_DIR) if DATA_DIR is not None else root / 'data'
    V9_PATH = Path(V9_PATH) if V9_PATH is not None else root / 'results/v9/submission.csv'
    V9_OOF_PATH = Path(V9_OOF_PATH) if V9_OOF_PATH is not None else root / 'outputs/order_refinement/v9_oof.csv'
    OUT = root / 'outputs/fikfair_v10'
OUT.mkdir(parents=True, exist_ok=True)
DATA_DIR, V9_PATH = Path(DATA_DIR), Path(V9_PATH)
print('Data:', DATA_DIR, '\\nv9:', V9_PATH, '\\nOutput:', OUT)
''')
code('''tr = pd.read_csv(DATA_DIR / 'train.csv')
te = pd.read_csv(DATA_DIR / 'test.csv')
template = pd.read_csv(DATA_DIR / 'submission_template.csv')
FEATURES = [c for c in tr.columns if c.startswith('feature_')]
X, y, Xte = tr[FEATURES].to_numpy(), tr.target.to_numpy(), te[FEATURES].to_numpy()
assert len(FEATURES) == 24
assert np.isfinite(X).all() and np.isfinite(Xte).all() and np.isfinite(y).all()
assert tr.record_id.is_unique and te.record_id.is_unique
assert not set(tr.record_id) & set(te.record_id)
v9_file = pd.read_csv(V9_PATH)
assert list(v9_file.columns) == ['record_id', 'prediction']
assert len(v9_file) == len(te) and v9_file.record_id.is_unique
assert set(v9_file.record_id) == set(te.record_id)
v9_test = te[['record_id']].merge(v9_file, on='record_id', validate='one_to_one').prediction.to_numpy()
assert np.isfinite(v9_test).all()
# For this particular v9 run the chosen rule was mean(quantile, sorted).
# Checking known predictions prevents accidentally attaching another version.
expected_head = np.array([92.368574, 82.252801, 77.549386, 73.350033, 70.529185])
assert np.allclose(v9_test[:5], expected_head, atol=2e-5, rtol=0), 'This is not the v9 submission audited here'
baseline_oof = None
if V9_OOF_PATH is not None and Path(V9_OOF_PATH).exists():
    saved = pd.read_csv(V9_OOF_PATH)
    assert saved.record_id.is_unique and set(saved.record_id) == set(tr.record_id)
    saved = tr[['record_id']].merge(saved, on='record_id', validate='one_to_one')
    assert np.allclose(saved.target, y)
    baseline_oof = saved[['v9_rep1', 'v9_rep2', 'v9_rep3']].to_numpy().T
print('Train/test:', X.shape, Xte.shape, '| archived baseline OOF:', baseline_oof is not None)
''')
md('''## Rank correction

Within each training partition, 12 whitened principal components define feature distance. The weighted ranks of 10 or 30 nearby training rows estimate where query rows belong. A leave-one-out training trend removes local calibration bias, and a smooth monotone correction adjusts the query ranks. Target values are used only to map corrected ranks to scores.

Neighbour count, correction strength, and light smoothing are selected inside each outer training fold. No outer-validation labels are used for that selection. The final 50:50 average with v9 is fixed, not fitted.''')
src=(ROOT/'eda/v9/48_order_refinement.py').read_text()
tree=ast.parse(src)
code('\n\n'.join(ast.get_source_segment(src,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['rank','curve','beta_curve','neighbor_warp']))
code('''def candidates(A, t, B):
    out = {'quantile': curve(t, rank(len(B))), 'beta': beta_curve(t, len(B))}
    for k in [10, 30]:
        for alpha in [.6, 1., 1.5]:
            u = neighbor_warp(A, B, k, alpha)
            for smooth in [0, .005]:
                out[f'k{k}_a{alpha}_s{smooth}'] = curve(t, u, smooth)
    return out

def select_rule(A, t, seed):
    loss = {}
    for a, b in KFold(5, shuffle=True, random_state=seed).split(A):
        a, b = np.sort(a), np.sort(b)
        assert not set(a) & set(b)
        for key, p in candidates(A[a], t[a], A[b]).items():
            loss[key] = loss.get(key, 0.) + np.sum((p - t[b]) ** 2)
    return min(loss, key=loss.get), {k: float(np.sqrt(v / len(t))) for k, v in loss.items()}
''')
md('''## Nested validation

The outer 5-fold, 3-repeat split matches the archived feature OOF. Each outer training partition gets its own inner 5-fold selection. Optional v9 OOF was reconstructed from earlier local model artifacts; its mean fold RMSE 3.6132 matches the Kaggle table rounded to 3.613. It is not claimed to be bitwise identical to Kaggle predictions.

Method design used this same small dataset, so nested evaluation reduces parameter-selection leakage but does not erase research-selection bias. The 0.7/0.3 feature weights and v9 rule were chosen in earlier research.''')
code('''start = time.perf_counter()
folds = list(RepeatedKFold(n_splits=5, n_repeats=3, random_state=42).split(X))
rank_oof, quantile_oof = np.zeros((3, len(y))), np.zeros((3, len(y)))
fold_rows = []
for i, (a, b) in enumerate(folds):
    a, b = np.sort(a), np.sort(b)
    assert not set(a) & set(b)
    selected, inner_scores = select_rule(X[a], y[a], 100 + i)
    predictions = candidates(X[a], y[a], X[b])
    rank_oof[i // 5, b] = predictions[selected]
    quantile_oof[i // 5, b] = predictions['quantile']
    fold_rows.append({'fold': i, 'selected': selected, 'inner_rmse': inner_scores[selected]})
    print(f'Fold {i+1}/15: {selected}')

predictions = {'quantile': quantile_oof, 'nested_rank': rank_oof}
if baseline_oof is not None:
    predictions['v9'] = baseline_oof
    predictions['v10_half_v9_half_rank'] = .5 * (baseline_oof + rank_oof)
rows = []
for name, p in predictions.items():
    fs = [np.sqrt(np.mean((p[i//5, b] - y[b])**2)) for i, (_, b) in enumerate(folds)]
    reps = [np.sqrt(np.mean((pp-y)**2)) for pp in p]
    rows.append({'method': name, 'mean_fold_rmse': np.mean(fs), 'mean_repeat_rmse': np.mean(reps),
                 'repeat1': reps[0], 'repeat2': reps[1], 'repeat3': reps[2]})
scores = pd.DataFrame(rows)
print(scores.round(4).to_string(index=False))
print(f'CPU diagnostic elapsed: {time.perf_counter()-start:.1f} seconds')
scores.to_csv(OUT / 'validation.csv', index=False)
pd.DataFrame(fold_rows).to_csv(OUT / 'fold_selection.csv', index=False)
np.savez(OUT / 'validation_predictions.npz', **predictions)
''')
md('''## Final prediction

Choose the rank rule by cross-validation on all labelled training rows, then predict test ranks without any test labels. The main submission averages this estimate with the actual v9 file. The rank-only file is a secondary diagnostic candidate that uses no pseudo-labels or foundation models. Neither file is submitted automatically.''')
code('''selected, inner_scores = select_rule(X, y, 4900)
rank_test = candidates(X, y, Xte)[selected]
final_test = .5 * (v9_test + rank_test)
assert np.isfinite(rank_test).all() and np.isfinite(final_test).all()
assert set(template.record_id) == set(te.record_id) and template.record_id.is_unique

def save_submission(p, filename):
    submission = template[['record_id']].merge(
        pd.DataFrame({'record_id': te.record_id, 'prediction': p}),
        on='record_id', how='left', validate='one_to_one')
    assert len(submission) == len(te)
    assert np.isfinite(submission.prediction).all()
    submission.to_csv(OUT / filename, index=False)
    return submission

final_submission = save_submission(final_test, 'submission_v10.csv')
save_submission(rank_test, 'submission_rank_only.csv')
manifest = {'selected_rank_rule': selected, 'full_training_cv': inner_scores,
            'v9_weight': .5, 'rank_weight': .5, 'v9_path': str(V9_PATH),
            'public_score_v9_reported_by_user': 2.3865618144,
            'public_score_v10': None}
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
changes = pd.DataFrame({'record_id': te.record_id, 'v9': v9_test,
                       'rank_only': rank_test, 'v10': final_test, 'change': final_test-v9_test})
changes.to_csv(OUT / 'prediction_changes.csv', index=False)
print('Selected:', selected)
print('RMS change from v9:', np.sqrt(np.mean((final_test-v9_test)**2)))
print(final_submission.head().to_string(index=False))
print('Largest changes:')
print(changes.loc[changes.change.abs().nlargest(10).index].round(4).to_string(index=False))
''')
md('''The original analysis measured mean fold RMSE 3.6132 for v9 and 3.4008 for the fixed average with nested rank correction. All three repeat-level RMSE values improved. This does not imply subtracting 0.2124 from the public score. Public and private scores depend on their unknown subsets and target values.''')
code('''import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(rank(len(te)), v9_test, label='v9')
axes[0].plot(rank(len(te)), rank_test, label='rank correction', alpha=.8)
axes[0].plot(rank(len(te)), final_test, label='v10')
axes[0].set(xlabel='Position within test file', ylabel='Prediction', title='Final candidate predictions')
axes[0].legend()
axes[1].plot(rank(len(te)), final_test-v9_test)
axes[1].axhline(0, color='black', lw=1)
axes[1].set(xlabel='Position within test file', ylabel='v10 minus v9', title='Changes do not use test labels')
fig.tight_layout()
fig.savefig(OUT / 'v10_predictions.png', dpi=150)
print('Saved plot:', OUT / 'v10_predictions.png')
''')

for i,c in enumerate(cells): c['id']=f'v10-{i:03}'
nb={'nbformat':4,'nbformat_minor':5,'metadata':{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':'3.12'},
    'kaggle':{'isInternetEnabled':False,'isGpuEnabled':False}},'cells':cells}
(ROOT/'notebooks/v10.ipynb').write_text(json.dumps(nb,indent=1))
print('Wrote',len(cells),'cells to notebooks/v10.ipynb')
