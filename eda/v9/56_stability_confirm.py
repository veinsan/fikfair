"""Confirm discovery on new partitions; all preprocessing uses training only.

Fresh partitions are not fresh data. Research-selection optimism remains.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold, ShuffleSplit

spec=importlib.util.spec_from_file_location('search',Path(__file__).with_name('55_stability_search.py'))
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
KEYS=['gaussian_interaction0_ridge10','local_linear_interaction1_ridge10']

def main():
    tr=pd.read_csv(s.n.m.ROOT/'data/train.csv');X=tr.filter(like='feature_').to_numpy();y=tr.target.to_numpy()
    q=s.n.m.rank(100)
    assert np.allclose(s.local_linear(np.ones(100)),1)
    assert np.allclose(s.local_linear(q),q)
    # Separate prediction reproducibility from correctness/generalization.
    p=s.candidates(X[:280],y[:280],X[280:])
    p2=s.candidates(X[:280],y[:280],X[280:])
    assert all(np.array_equal(p[k],p2[k]) and np.isfinite(p[k]).all() for k in p)
    rows=[];rng=np.random.default_rng(560002)
    for i,(a,b) in enumerate(ShuffleSplit(n_splits=200,test_size=96,random_state=560001).split(X)):
        a,b=np.sort(a),np.sort(b);cs=s.candidates(X[a],y[a],X[b])
        pools=[rng.permutation(96) for _ in range(25)]
        for k in KEYS:
            e=(cs[k]-y[b])**2
            for j,ix in enumerate(pools):
                rows.append(dict(split=i,partition=j,candidate=k,query=np.sqrt(e.mean()),
                    public=np.sqrt(e[ix[:29]].mean()),private=np.sqrt(e[ix[29:]].mean())))
    df=pd.DataFrame(rows);df['gap']=abs(df.public-df.private)
    df.to_csv(s.OUT/'confirmation_partitions.csv',index=False)
    summary=df.groupby('candidate').agg(query_rmse=('query','mean'),private_mean=('private','mean'),
        private_p95=('private',lambda x:x.quantile(.95)),gap_mean=('gap','mean'))
    summary.to_csv(s.OUT/'confirmation_summary.csv')
    print(summary.to_string(),flush=True)
    # Select either old or proposed component using only each outer training set.
    base=np.load(s.n.m.ROOT/'outputs/v11_eda/nested_predictions.npz')['v11']
    raw=np.zeros_like(base);choices=[]
    folds=list(RepeatedKFold(n_splits=5,n_repeats=3,random_state=42).split(X))
    for i,(a,b) in enumerate(folds):
        a,b=np.sort(a),np.sort(b);loss={k:0. for k in KEYS}
        for aa,bb in KFold(5,shuffle=True,random_state=560100+i).split(a):
            aa,bb=a[np.sort(aa)],a[np.sort(bb)]
            assert not set(aa)&set(b) and not set(bb)&set(b)
            for k,p in s.candidates(X[aa],y[aa],X[bb]).items():
                if k in loss:loss[k]+=float(np.sum((p-y[bb])**2))
        key=min(loss,key=loss.get);raw[i//5,b]=s.candidates(X[a],y[a],X[b])[key]
        choices.append(key)
    hedge=.5*(base+raw)
    np.savez(s.OUT/'confirmation_oof.npz',v11=base,new_component=raw,hedge=hedge)
    result={'selected_per_fold':choices,'scores':{}}
    for k,p in [('v11',base),('new_component',raw),('hedge',hedge)]:
        result['scores'][k]={'repeat_rmse':np.sqrt(np.mean((p-y)**2,axis=1)).tolist(),
            'mean_fold_rmse':float(np.mean([np.sqrt(np.mean((p[i//5,b]-y[b])**2)) for i,(_,b) in enumerate(folds)]))}
    delta=[]
    for _ in range(5000):
        ix=rng.integers(len(y),size=len(y))
        delta.append(np.sqrt(np.mean((hedge[:,ix]-y[ix])**2))-np.sqrt(np.mean((base[:,ix]-y[ix])**2)))
    result['record_bootstrap_delta95']=np.quantile(delta,[.025,.975]).tolist()
    result['fold_wins']=sum(np.mean((hedge[i//5,b]-y[b])**2)<np.mean((base[i//5,b]-y[b])**2) for i,(_,b) in enumerate(folds)).item()
    # Sampling diagnostic only: these cohorts combine predictions from different
    # outer folds, unlike the coherent 96-row component test above.
    rng=np.random.default_rng(560777)
    groups=[(int(rng.integers(3)),rng.choice(len(y),96,replace=False)) for _ in range(20000)]
    result['approximate_oof_sampling']={}
    for key,p in [('v11',base),('v12',hedge)]:
        pairs=np.array([(np.sqrt(np.mean((p[r,ix[:29]]-y[ix[:29]])**2)),
                         np.sqrt(np.mean((p[r,ix[29:]]-y[ix[29:]])**2))) for r,ix in groups])
        result['approximate_oof_sampling'][key]={
            'mean_absolute_gap':float(np.mean(abs(pairs[:,0]-pairs[:,1]))),
            'private_p95':float(np.quantile(pairs[:,1],.95)),
            'first38_records_rmse':float(np.sqrt(np.mean((p[:,:38]-y[:38])**2))),
            'first38_records_sse_share':float(np.sum((p[:,:38]-y[:38])**2)/np.sum((p-y)**2))}
    (s.OUT/'confirmation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
