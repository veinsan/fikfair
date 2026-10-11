"""Coherent 96-row query validation, partitioned into disjoint public/private.

Uses rank/feature methods only on new splits, never old foundation OOF across
incompatible folds. Candidate catalogue and simulation seeds are fixed here.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit

spec=importlib.util.spec_from_file_location('r51',Path(__file__).with_name('51_v11_rank_eda.py'))
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
OUT=n.m.ROOT/'outputs/stability_search';OUT.mkdir(exist_ok=True)

def local_linear(z,h=.05):
    q=n.m.rank(len(z));dx=q[None,:]-q[:,None]
    w=np.exp(-.5*(dx/h)**2)
    s0=w.sum(1);s1=(w*dx).sum(1);s2=(w*dx**2).sum(1)
    S=w*(s2[:,None]-s1[:,None]*dx)/(s0*s2-s1*s1)[:,None]
    return S@z

def candidates(A,t,B):
    qa=n.m.rank(len(A));u=n.warp(A,B)
    base=n.m.curve(t,u,.005)
    mu,sd=A.mean(0),A.std(0);Z=(A-mu)/sd;Q=(B-mu)/sd
    out={}
    for smoother in ['gaussian','local_linear']:
        if smoother=='gaussian':
            tx=gaussian_filter1d(Z,.05*len(A),axis=0,mode='nearest')
            ty=gaussian_filter1d(t,.05*len(A),mode='nearest')
        else:tx,ty=local_linear(Z),local_linear(t)
        RX=Z-tx;RQ=Q-np.column_stack([np.interp(u,qa,tx[:,j]) for j in range(Z.shape[1])])
        for interaction in [False,True]:
            RA=np.c_[RX,RX*(2*qa[:,None]-1)] if interaction else RX
            RB=np.c_[RQ,RQ*(2*u[:,None]-1)] if interaction else RQ
            for alpha in [10,100]:
                p=base+Ridge(alpha=alpha).fit(RA,t-ty).predict(RB)
                key=f'{smoother}_interaction{int(interaction)}_ridge{alpha}'
                out[key]=p
    baseline=out['gaussian_interaction0_ridge10']
    # A fixed conservative hedge for each extension.
    for key,p in list(out.items()):
        if key!='gaussian_interaction0_ridge10':out[key+'_half']=.5*(p+baseline)
    return out

def main():
    tr=pd.read_csv(n.m.ROOT/'data/train.csv');X=tr.filter(like='feature_').to_numpy();y=tr.target.to_numpy()
    splits=list(ShuffleSplit(n_splits=100,test_size=96,random_state=550001).split(X))
    rng=np.random.default_rng(550002)
    scores=[];paired=[];predictions={};fold_ids=[]
    for i,(a,b) in enumerate(splits):
        a,b=np.sort(a),np.sort(b);assert not set(a)&set(b)
        assert len(b)==96
        pools=[rng.permutation(96) for _ in range(50)]
        fold_ids.append(b)
        for key,p in candidates(X[a],y[a],X[b]).items():
            predictions.setdefault(key,[]).append(p)
            e2=(p-y[b])**2
            scores.append({'split':i,'candidate':key,'rmse':np.sqrt(e2.mean()),
                          'top10_rank_rmse':np.sqrt(e2[:10].mean())})
            for j,ix in enumerate(pools):
                pub,private=ix[:29],ix[29:]
                assert len(private)==67 and not set(pub)&set(private)
                paired.append({'split':i,'partition':j,'candidate':key,
                   'public':np.sqrt(e2[pub].mean()),'private':np.sqrt(e2[private].mean())})
        if i%20==0:print('96-row query split',i,'done',flush=True)
    scores=pd.DataFrame(scores);paired=pd.DataFrame(paired)
    paired['gap']=(paired.private-paired.public).abs()
    summary=scores.groupby('candidate').agg(mean_query_rmse=('rmse','mean'),
          query_p90=('rmse',lambda z:z.quantile(.9)),mean_tail_rmse=('top10_rank_rmse','mean'))
    for key,g in paired.groupby('candidate'):
        summary.loc[key,'mean_private']=g.private.mean()
        summary.loc[key,'private_p90']=g.private.quantile(.9)
        summary.loc[key,'private_p95']=g.private.quantile(.95)
        summary.loc[key,'mean_absolute_gap']=g.gap.mean()
    summary.sort_values('mean_query_rmse').to_csv(OUT/'discovery_summary.csv')
    scores.to_csv(OUT/'query_scores.csv',index=False)
    paired.to_csv(OUT/'paired_public_private.csv',index=False)
    np.savez(OUT/'query_predictions.npz',indices=np.array(fold_ids),**{k:np.array(v) for k,v in predictions.items()})
    print(summary.sort_values('mean_query_rmse').round(4).to_string(),flush=True)
    # The actual v11 archived OOF: approximate 96-row cohorts across folds.
    # These are NOT coherent new query predictions; label separately.
    old=np.load(n.m.ROOT/'outputs/v11_eda/nested_predictions.npz')['v11']
    rows=[]
    for _ in range(20000):
        r=int(rng.integers(3));ix=rng.choice(len(y),96,replace=False)
        e2=(y[ix]-old[r,ix])**2
        rows.append((np.sqrt(e2[:29].mean()),np.sqrt(e2[29:].mean())))
    pp=np.array(rows)
    summary2={'kind':'approximate resampling of archived v11 OOF across different folds',
       'public_quantiles':np.quantile(pp[:,0],[.05,.5,.95]).tolist(),
       'private_quantiles':np.quantile(pp[:,1],[.05,.5,.95]).tolist(),
       'absolute_gap_quantiles':np.quantile(abs(pp[:,0]-pp[:,1]),[.5,.9,.95]).tolist(),
       'fraction_private_worse':float(np.mean(pp[:,1]>pp[:,0]))}
    (OUT/'v11_sampling_only.json').write_text(json.dumps(summary2,indent=2))
    print('Archived v11 sampling-only simulation',summary2)

if __name__=='__main__':main()
