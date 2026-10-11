"""Nested selection, near-neighbor group stress, and learning curves."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import KFold,RepeatedKFold,ShuffleSplit,GroupShuffleSplit

spec=importlib.util.spec_from_file_location('a',Path(__file__).with_name('59_preprocessing_ablation.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
CATALOG=dict(a.CONFIGS)
CATALOG.update({f'kernel_a{alpha}_g{gamma}':{'kernel':True,'kernel_alpha':alpha,'gamma_scale':gamma}
               for alpha in [.1,.3,1] for gamma in [1,2,4] if (alpha,gamma)!=(1,1)})

def select(X,y,seed):
    loss={k:0. for k in CATALOG}
    for u,v in KFold(5,shuffle=True,random_state=seed).split(X):
        u,v=np.sort(u),np.sort(v)
        for k,c in CATALOG.items():loss[k]+=float(np.sum((a.predict(X[u],y[u],X[v],c)-y[v])**2))
    return min(loss,key=loss.get),{k:float(np.sqrt(v/len(y))) for k,v in loss.items()}

def main():
    tr=pd.read_csv(a.s.n.m.ROOT/'data/train.csv');X=tr.filter(like='feature_').to_numpy();y=tr.target.to_numpy()
    base=np.load(a.s.n.m.ROOT/'outputs/stability_search/confirmation_oof.npz')['hedge']
    pred=np.zeros_like(base);choices=[];folds=list(RepeatedKFold(n_splits=5,n_repeats=3,random_state=42).split(X))
    for i,(u,v) in enumerate(folds):
        u,v=np.sort(u),np.sort(v);assert not set(u)&set(v)
        key,_=select(X[u],y[u],6000+i);choices.append(key)
        pred[i//5,v]=a.predict(X[u],y[u],X[v],CATALOG[key]);print('outer',i,key,flush=True)
    hedge=.5*(base+pred);summary={'choices':choices,'scores':{}}
    for key,p in [('v12',base),('new',pred),('v13',hedge)]:
        summary['scores'][key]={'mean_fold':float(np.mean([np.sqrt(np.mean((p[i//5,v]-y[v])**2)) for i,(_,v) in enumerate(folds)])),
            'repeats':np.sqrt(np.mean((p-y)**2,axis=1)).tolist()}
    summary['fold_wins']=int(sum(np.mean((hedge[i//5,v]-y[v])**2)<np.mean((base[i//5,v]-y[v])**2) for i,(_,v) in enumerate(folds)))
    rng=np.random.default_rng(60042);delta=[]
    for _ in range(5000):
        ix=rng.integers(len(y),size=len(y));delta.append(np.sqrt(np.mean((hedge[:,ix]-y[ix])**2))-np.sqrt(np.mean((base[:,ix]-y[ix])**2)))
    summary['conditional_record_bootstrap_delta95']=np.quantile(delta,[.025,.975]).tolist()
    np.savez(a.OUT/'nested_oof.npz',v12=base,new=pred,v13=hedge)
    (a.OUT/'nested.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
    z=(X-X.mean(0))/X.std(0);D=cdist(z,z);rows=[];group_info={}
    for threshold in [.35,.7]:
        count,g=connected_components(D<threshold,directed=False);sizes=np.bincount(g)
        group_info[str(threshold)]={'groups':int(count),'max_size':int(sizes.max()),'non_singleton_records':int(sizes[sizes>1].sum())}
        for i,(u,v) in enumerate(GroupShuffleSplit(n_splits=80,test_size=.25,random_state=6010).split(X,groups=g)):
            u,v=np.sort(u),np.sort(v);assert not set(g[u])&set(g[v])
            for k in ['baseline','kernel','kernel_a0.3_g2']:
                p=a.predict(X[u],y[u],X[v],CATALOG[k]);rows.append(dict(threshold=threshold,split=i,candidate=k,rmse=np.sqrt(np.mean((p-y[v])**2))))
    df=pd.DataFrame(rows);df.to_csv(a.OUT/'group_stress.csv',index=False)
    (a.OUT/'group_definition.json').write_text(json.dumps(group_info,indent=2))
    print('Label-blind proximity groups, not verified entities:',group_info,flush=True)
    print(df.groupby(['threshold','candidate']).rmse.mean().to_string(),flush=True)
    learning=[]
    for i,(u,v) in enumerate(ShuffleSplit(n_splits=60,test_size=96,random_state=6020).split(X)):
        v=np.sort(v)
        for size in [143,214,286]:
            aa=np.sort(u[:size])
            for k in ['baseline','kernel','kernel_a0.3_g2']:
                p=a.predict(X[aa],y[aa],X[v],CATALOG[k])
                learning.append(dict(split=i,size=size,candidate=k,rmse=np.sqrt(np.mean((p-y[v])**2))))
    lc=pd.DataFrame(learning);lc.to_csv(a.OUT/'learning_curve.csv',index=False)
    print(lc.groupby(['size','candidate']).rmse.mean().to_string(),flush=True)
    fig,ax=plt.subplots(1,3,figsize=(15,4))
    for k,c in [('v12',base),('v13',hedge)]:
        ax[0].plot([np.sqrt(np.mean((c[i//5,v]-y[v])**2)) for i,(_,v) in enumerate(folds)],label=k)
    ax[0].legend();ax[0].set_title('Nested outer fold RMSE')
    df.groupby(['threshold','candidate']).rmse.mean().unstack().plot.bar(ax=ax[1]);ax[1].set_title('Near-neighbor group stress')
    lc.groupby(['size','candidate']).rmse.mean().unstack().plot(ax=ax[2]);ax[2].set_title('Learning curve: same 96 queries')
    fig.tight_layout();fig.savefig(a.OUT/'confirmation.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
