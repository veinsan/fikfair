"""Read-only analysis of saved predictions; writes diagnostic artifacts only.

Row-order checks are dataset leakage diagnostics, not submission recipes.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import nnls
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'score_gap_audit'
OUT.mkdir(exist_ok=True)
tr = pd.read_csv(ROOT / 'data/train.csv')
te = pd.read_csv(ROOT / 'data/test.csv')
y = tr.target.to_numpy()
n = len(y)
features = tr.filter(like='feature_')
rmse = lambda a, b: float(np.sqrt(np.mean((a-b)**2)))
lib = np.load(ROOT / 'outputs/eda/v7_members.npz')
fm = lib['FM']
svr = np.load(ROOT / 'outputs/eda/v9_svrpl_avg248.npy')
assert fm.shape == svr.shape == (3, n)
for r in (1, 2, 3):
    saved = pd.read_csv(ROOT / f'outputs/eda/v5_25_oof_rep{r}.csv')
    assert np.array_equal(saved.record_id, tr.record_id)
    assert np.allclose(saved.target, y)
pred = np.zeros(n)
for a,b in KFold(10, shuffle=True, random_state=42).split(y):
    mat = np.c_[fm.mean(0), svr.mean(0)]
    w,_ = nnls(mat[a], y[a])
    pred[b] = mat[b] @ (w/w.sum())
err = (y-pred)**2
rng = np.random.default_rng(47)
scores=[]
for _ in range(100):
    idx = np.argsort(rng.random((2000,n)), axis=1)[:,:29]
    scores.extend(np.sqrt(err[idx].mean(1)))
scores=np.array(scores)
metrics = {
    'train_shape': list(tr.shape), 'test_shape': list(te.shape),
    'missing_train': int(tr.isna().sum().sum()),
    'duplicate_features': int(features.duplicated().sum()),
    'target_range': [float(y.min()),float(y.max())],
    'row_target_spearman': float(spearmanr(np.arange(n),y).statistic),
    'id_target_spearman': float(spearmanr(tr.record_id.str[1:].astype(int),y).statistic),
    'v8_saved_oof_meta_cv_rmse': rmse(y,pred),
    'public29_simulation_count':len(scores),
    'public29_rmse_quantiles':dict(zip(['0.1%','1%','5%','50%','95%'],np.quantile(scores,[.001,.01,.05,.5,.95]).tolist())),
    'public29_count_below_3':int((scores<3).sum()),
    'public29_count_below_2_3':int((scores<2.3).sum()),
    'tail_over70_n':int((y>70).sum()),
    'tail_over70_sse_share':float(err[y>70].sum()/err.sum()),
    'rmse_without_tail':rmse(y[y<=70],pred[y<=70]),
}
# Fixed original positions retain information about held-out ordering: deliberately
# optimistic diagnostic, NOT a clean prediction benchmark.
position = (np.arange(n)+.5)/n
order_oof=np.zeros(n)
smooth_oof=np.zeros(n)
for a,b in KFold(5,shuffle=True,random_state=42).split(y):
    model=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(position[a],y[a])
    order_oof[b]=model.predict(position[b])
    smooth=make_pipeline(SplineTransformer(n_knots=8),Ridge(alpha=1))
    smooth.fit(position[a,None],y[a])
    smooth_oof[b]=smooth.predict(position[b,None])
metrics['fixed_original_position_diagnostic_rmse']=rmse(y,order_oof)
metrics['fixed_original_position_spline_diagnostic_rmse']=rmse(y,smooth_oof)
# More realistic split simulation: discard global positions and independently
# normalize row ranks within each partition, preserving only file order.
split_rows=[]
rank_public=[]
for seed,(a,b) in enumerate(ShuffleSplit(n_splits=200,test_size=.2,random_state=47).split(y)):
    a,b=np.sort(a),np.sort(b)
    qa=(np.arange(len(a))+.5)/len(a)
    qb=(np.arange(len(b))+.5)/len(b)
    model=IsotonicRegression(increasing=False,out_of_bounds='clip').fit(qa,y[a])
    p=model.predict(qb)
    split_rows.append({'split':seed,'rank_only_rmse':rmse(y[b],p)})
    for _ in range(200):
        ix=rng.choice(len(b),29,replace=False)
        rank_public.append(rmse(y[b][ix],p[ix]))
splits=pd.DataFrame(split_rows)
splits.to_csv(OUT/'order_split_diagnostics.csv',index=False)
metrics['independent_rank_diagnostic_rmse_quantiles']=dict(zip(['5%','50%','95%'],np.quantile(splits.rank_only_rmse,[.05,.5,.95]).tolist()))
metrics['independent_rank_fraction_below3']=float((splits.rank_only_rmse<3).mean())
metrics['independent_rank_public29_fraction_below2_3']=float((np.array(rank_public)<2.3).mean())
# Feature-based predictions and feature trajectories corroborate ordering in test
# without ever accessing test labels.
row_corr=[]
for c in features:
    row_corr.append({'feature':c,'train_row_rho':spearmanr(np.arange(n),tr[c]).statistic,
                     'test_row_rho':spearmanr(np.arange(len(te)),te[c]).statistic})
corr=pd.DataFrame(row_corr)
corr.to_csv(OUT/'feature_order_correlations.csv',index=False)
metrics['train_test_row_feature_rho_correlation']=float(np.corrcoef(corr.train_row_rho,corr.test_row_rho)[0,1])
versions=[]
for path in sorted((ROOT/'results').glob('*/submission.csv')):
    sub=pd.read_csv(path)
    joined=te[['record_id']].merge(sub,on='record_id',validate='one_to_one')
    versions.append({'version':path.parent.name,'row_prediction_spearman':spearmanr(np.arange(len(te)),joined.prediction).statistic})
pd.DataFrame(versions).to_csv(OUT/'submission_order_correlations.csv',index=False)
pd.DataFrame({'record_id':tr.record_id,'target':y,'v8_oof':pred,'position_diagnostic':order_oof,'squared_error':err}).to_csv(OUT/'residuals.csv',index=False)
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(15,4))
ax[0].scatter(np.arange(n),y,s=9,alpha=.7)
ax[0].set(xlabel='Original CSV row position',ylabel='Training target',title='Row order carries target information')
ax[1].scatter(y,pred,s=10,label='24-feature saved OOF',alpha=.6)
ax[1].scatter(y,smooth_oof,s=10,label='Row-position diagnostic',alpha=.6)
ax[1].plot([30,105],[30,105],color='black',lw=1)
ax[1].set(xlabel='Target',ylabel='Prediction',title='Diagnostic only: order is a leakage channel')
ax[1].legend(fontsize=8)
ax[2].hist(scores,bins=70,density=True,alpha=.6,label='Feature OOF subsets')
ax[2].hist(rank_public,bins=70,density=True,alpha=.6,label='Independent-rank split subsets')
ax[2].axvline(2.3,color='black',ls='--',label='2.3 reference')
ax[2].set(xlabel='RMSE on 29 sampled rows',title='Conditional simulations, not actual LB odds')
ax[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT/'score_gap_audit.png',dpi=160)
