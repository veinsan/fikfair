"""Train-only preprocessing ablations; save numbers and inspectable figures."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import ShuffleSplit

spec=importlib.util.spec_from_file_location('s',Path(__file__).with_name('55_stability_search.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
OUT=s.n.m.ROOT/'outputs/deep_v13';OUT.mkdir(exist_ok=True)
CONFIGS={'baseline':{},'narrow':{'h':.025},'wide':{'h':.10},
 'ridge3':{'alpha':3},'ridge30':{'alpha':30},'quadratic_rank':{'degree':2},
 'log_target':{'log':True},'residual_scaled':{'scaled':True},
 'kernel':{'kernel':True},'smooth_curve':{'smooth':.015},
 'linear_target_curve':{'linear_curve':True},'clip_features':{'clip':True}}

def predict(A,t,B,config,diagnostic=False):
    qa=s.n.m.rank(len(A));u=s.n.warp(A,B)
    mu,sd=A.mean(0),A.std(0);sd=np.maximum(sd,1e-12)
    Z=(A-mu)/sd;Q=(B-mu)/sd
    if config.get('clip'):
        lo,hi=np.quantile(Z,[.01,.99],axis=0);Z=np.clip(Z,lo,hi);Q=np.clip(Q,lo,hi)
    target=np.log(t) if config.get('log') else t
    h=config.get('h',.05);tx=s.local_linear(Z,h);ty=s.local_linear(target,h)
    RX=Z-tx;RQ=Q-np.column_stack([np.interp(u,qa,tx[:,j]) for j in range(Z.shape[1])])
    degree=config.get('degree',1)
    RA=np.column_stack([RX*(2*qa[:,None]-1)**d for d in range(degree+1)])
    RB=np.column_stack([RQ*(2*u[:,None]-1)**d for d in range(degree+1)])
    resid=target-ty;scale=np.ones(len(t));query_scale=np.ones(len(B))
    if config.get('scaled'):
        scale=np.sqrt(np.maximum(s.local_linear(resid**2,.1),1.))
        query_scale=np.interp(u,qa,scale)
    model=KernelRidge(alpha=config.get('kernel_alpha',1),kernel='rbf',gamma=config.get('gamma_scale',1)/RA.shape[1]) if config.get('kernel') else Ridge(alpha=config.get('alpha',10))
    residual=model.fit(RA,resid/scale).predict(RB)*query_scale
    base=np.interp(u,qa,ty) if config.get('linear_curve') else s.n.m.curve(target,u,config.get('smooth',.005))
    pred=base+residual
    if config.get('log'):pred=np.exp(pred)
    assert np.isfinite(pred).all()
    return (pred,(Z,tx,RX,ty,resid)) if diagnostic else pred

def main():
    tr=pd.read_csv(s.n.m.ROOT/'data/train.csv');X=tr.filter(like='feature_').to_numpy();y=tr.target.to_numpy()
    rows=[];pairs=[];rng=np.random.default_rng(5902)
    for i,(a,b) in enumerate(ShuffleSplit(n_splits=120,test_size=96,random_state=5901).split(X)):
        a,b=np.sort(a),np.sort(b);assert not set(a)&set(b)
        pools=[rng.permutation(96) for _ in range(20)]
        for k,c in CONFIGS.items():
            p=predict(X[a],y[a],X[b],c);e=(p-y[b])**2
            if i==0 and k=='baseline':
                old=s.candidates(X[a],y[a],X[b])['local_linear_interaction1_ridge10']
                assert np.allclose(p,old,atol=1e-10,rtol=0)
            rows.append(dict(split=i,candidate=k,rmse=np.sqrt(e.mean()),tail_rmse=np.sqrt(e[:10].mean())))
            for ix in pools:
                pub,pri=np.sqrt(e[ix[:29]].mean()),np.sqrt(e[ix[29:]].mean())
                pairs.append(dict(split=i,candidate=k,public=pub,private=pri,gap=abs(pub-pri)))
        if i%30==0:print('split',i,flush=True)
    df=pd.DataFrame(rows);pp=pd.DataFrame(pairs)
    table=df.groupby('candidate').agg(rmse=('rmse','mean'),tail_rmse=('tail_rmse','mean'))
    table['private_p95']=pp.groupby('candidate').private.quantile(.95)
    table['gap']=pp.groupby('candidate').gap.mean()
    table.sort_values('rmse').to_csv(OUT/'ablation_summary.csv');df.to_csv(OUT/'ablation_splits.csv',index=False)
    pp.to_csv(OUT/'ablation_pairs.csv',index=False);print(table.sort_values('rmse').round(5).to_string())
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    table.sort_values('rmse').rmse.plot.barh(ax=ax[0]);ax[0].set_title('120 query splits: component RMSE')
    ax[1].scatter(table.rmse,table.private_p95)
    for k,row in table.iterrows():ax[1].annotate(k,(row.rmse,row.private_p95),fontsize=7)
    ax[1].set(xlabel='Mean query RMSE',ylabel='Simulated private p95')
    fig.tight_layout();fig.savefig(OUT/'ablation.png',dpi=140);plt.close(fig)
    a,b=next(ShuffleSplit(n_splits=1,test_size=96,random_state=5901).split(X));a,b=np.sort(a),np.sort(b)
    _,(Z,tx,RX,ty,resid)=predict(X[a],y[a],X[b],{},True)
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    ax[0,0].plot(Z[:,0],label='Scaled feature 1');ax[0,0].plot(tx[:,0],label='Trend');ax[0,0].legend()
    ax[0,1].plot(RX[:,0]);ax[0,1].set_title('Feature 1 after trend removal')
    ax[1,0].scatter(np.arange(len(a)),y[a],s=6);ax[1,0].plot(ty,color='orange');ax[1,0].set_title('Training target and local-linear trend')
    ax[1,1].scatter(np.arange(len(a)),resid,s=6);ax[1,1].set_title('Training residual: remaining unequal variance')
    fig.tight_layout();fig.savefig(OUT/'preprocessing.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
