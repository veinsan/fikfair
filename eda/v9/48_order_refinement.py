"""Diagnose rank-transfer error using saved, same-fold feature predictions.

No foundation models are trained and no leaderboard feedback is fitted.
New candidates are exploratory until confirmed on held-out outer splits.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
from scipy.special import betainc
from scipy.spatial.distance import cdist
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/order_refinement'
OUT.mkdir(exist_ok=True)

def rank(n):
    return (np.arange(n)+.5)/n

def curve(t, u, smooth=0):
    q=rank(len(t))
    ys=IsotonicRegression(increasing=False).fit_transform(q,t)
    if smooth:
        ys=gaussian_filter1d(ys, smooth*len(t),mode='nearest')
    return np.interp(u,q,ys)

def beta_curve(t, m):
    # Integrate the smoothed training rank curve against each query order
    # statistic's Beta(j, m+1-j) marginal. This is a modelling approximation.
    q=rank(len(t))
    ys=IsotonicRegression(increasing=False).fit_transform(q,t)
    edges=np.r_[0,(q[1:]+q[:-1])/2,1]
    j=np.arange(1,m+1)[:,None]
    w=np.diff(betainc(j,m+1-j,edges[None,:]),axis=1)
    return w@ys

def neighbor_warp(A,B,k=20,alpha=.5):
    # Fit feature transform and calibration on the current training partition.
    mu,sd=A.mean(0),A.std(0)
    pc=PCA(n_components=12).fit((A-mu)/sd)
    ZA=pc.transform((A-mu)/sd)/np.sqrt(pc.explained_variance_)
    ZB=pc.transform((B-mu)/sd)/np.sqrt(pc.explained_variance_)
    DAA=cdist(ZA,ZA); np.fill_diagonal(DAA,np.inf)
    DBA=cdist(ZB,ZA)
    qa,qb=rank(len(A)),rank(len(B))
    def local(D):
        idx=np.argpartition(D,k-1,axis=1)[:,:k]
        d=np.take_along_axis(D,idx,axis=1)
        w=1/np.maximum(d,.05)**2
        return np.sum(w*qa[idx],axis=1)/w.sum(1)
    ref=local(DAA); query=local(DBA)
    ref_trend=gaussian_filter1d(ref,max(1,.05*len(A)),mode='nearest')
    delta=query-np.interp(qb,qa,ref_trend)
    delta=gaussian_filter1d(delta,max(1,.10*len(B)),mode='nearest')
    newq=np.clip(qb+alpha*4*qb*(1-qb)*delta,0,1)
    return IsotonicRegression(increasing=True).fit_transform(qb,newq)

def evaluate():
    tr=pd.read_csv(ROOT/'data/train.csv'); y=tr.target.to_numpy()
    X=tr.filter(like='feature_').to_numpy()
    for r in [1,2,3]:
        ref=pd.read_csv(ROOT/f'outputs/eda/v5_25_oof_rep{r}.csv')
        assert np.array_equal(ref.record_id,tr.record_id)
        assert np.allclose(ref.target,y)
    fm=np.load(ROOT/'outputs/eda/v7_members.npz')['FM']
    sv=np.load(ROOT/'outputs/eda/v9_svrpl_avg248.npy')
    feature=.7*fm+.3*sv
    folds=list(RepeatedKFold(n_splits=5,n_repeats=3,random_state=42).split(X))
    preds={}; rows=[]; diagnostic=[]
    for i,(a,b) in enumerate(folds):
        a,b=np.sort(a),np.sort(b); r=i//5
        q=curve(y[a],rank(len(b))); f=np.sort(feature[r,b])[::-1]
        cs={'v9':.5*(q+f),'quantile':q,'sorted_feature':f}
        for w in [.25,.75]:
            cs[f'quantile_weight_{w}']=w*q+(1-w)*f
        for s in [.005,.015,.03]:
            smooth=curve(y[a],rank(len(b)),s)
            cs[f'smooth_{s}']=smooth
            cs[f'smooth_{s}_blend']=.5*(smooth+f)
        beta=beta_curve(y[a],len(b))
        cs['beta']=beta; cs['beta_blend']=.5*(beta+f)
        iso=IsotonicRegression(increasing=False).fit_transform(rank(len(a)),y[a])
        cs['pchip_blend']=.5*(PchipInterpolator(rank(len(a)),iso)(np.clip(rank(len(b)),rank(len(a))[0],rank(len(a))[-1]))+f)
        for k in [10,30]:
            for alpha in [.3,.6,1.]:
                uq=neighbor_warp(X[a],X[b],k,alpha)
                aligned=curve(y[a],uq)
                cs[f'warp_k{k}_a{alpha}']=aligned
                cs[f'warp_k{k}_a{alpha}_blend']=.5*(aligned+f)
        for name,p in cs.items():
            if name not in preds: preds[name]=np.zeros((3,len(y)))
            preds[name][r,b]=p
            rows.append({'candidate':name,'fold':i,'repeat':r,'rmse':np.sqrt(np.mean((p-y[b])**2))})
        for j,idx in enumerate(b):
            diagnostic.append({'repeat':r,'fold':i,'record_id':tr.record_id.iloc[idx],
                'global_position':int(idx),'query_rank':rank(len(b))[j],
                'reference_rank':np.searchsorted(a,idx)/len(a),'target':y[idx],
                'quantile':q[j],'sorted_feature':f[j],'v9':cs['v9'][j]})
    scores=pd.DataFrame(rows)
    table=scores.groupby('candidate').rmse.agg(['mean','std'])
    for name,p in preds.items():
        table.loc[name,'pooled_sse_rmse']=np.sqrt(np.mean((p-y[None,:])**2))
        table.loc[name,'delta_vs_v9']=table.loc[name,'mean']-table.loc['v9','mean']
        table.loc[name,'fold_wins']=int(np.sum(scores.loc[scores.candidate==name,'rmse'].values<scores.loc[scores.candidate=='v9','rmse'].values))
    table.sort_values('mean').to_csv(OUT/'discovery_scores.csv')
    scores.to_csv(OUT/'discovery_folds.csv',index=False)
    pd.DataFrame(diagnostic).to_csv(OUT/'v9_rank_residuals.csv',index=False)
    np.savez(OUT/'discovery_predictions.npz',**preds)
    print(table.sort_values('mean').round(4).to_string())
    d=pd.DataFrame(diagnostic)
    d['error']=d.target-d.v9
    d['rank_gap']=d.reference_rank-d.query_rank
    d['band']=pd.cut(d.query_rank,[0,.1,.25,.5,.75,.9,1])
    print('\nError by within-file position:')
    print(d.groupby('band',observed=True).agg(n=('error','size'),bias=('error','mean'),rmse=('error',lambda z:np.sqrt(np.mean(z*z)))).round(3))
    print('\nCorr rank displacement vs quantile error',np.corrcoef(d.rank_gap,d.target-d['quantile'])[0,1])

if __name__=='__main__':
    evaluate()
