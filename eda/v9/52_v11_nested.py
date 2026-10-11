"""Nested evaluation of rank and feature-residual estimators, with fixed v10 hedge."""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold,RepeatedKFold,ShuffleSplit

spec=importlib.util.spec_from_file_location('rank51',Path(__file__).with_name('51_v11_rank_eda.py'))
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
spec=importlib.util.spec_from_file_location('rank49',Path(__file__).with_name('49_order_nested_validation.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
spec=importlib.util.spec_from_file_location('rank53',Path(__file__).with_name('53_v11_residual_check.py'))
res=importlib.util.module_from_spec(spec);spec.loader.exec_module(res)
KEYS=['gaussian_dim8','gaussian_dim12','dimension_mean','bridge_0.5','subbag16',
      'ridge10_w0.5','ridge10_w1.0','ridge100_w0.5','ridge100_w1.0']

def candidates(A,t,B):
    all_p=n.all_candidates(A,t,B)
    all_p.update(res.residual_candidates(A,t,B))
    return {k:all_p[k] for k in KEYS}

def select(A,t,seed):
    losses={k:0. for k in KEYS}
    for a,b in KFold(5,shuffle=True,random_state=seed).split(A):
        a,b=np.sort(a),np.sort(b)
        assert not set(a)&set(b)
        for key,p in candidates(A[a],t[a],A[b]).items():
            losses[key]+=np.sum((p-t[b])**2)
    return min(losses,key=losses.get),{k:float(np.sqrt(v/len(t))) for k,v in losses.items()}

def main():
    tr=pd.read_csv(n.m.ROOT/'data/train.csv');X=tr.filter(like='feature_').to_numpy();y=tr.target.to_numpy()
    archived=np.load(n.m.ROOT/'outputs/order_refinement/nested_predictions.npz')
    base=archived['nested_rank_hedge_v9']
    folds=list(RepeatedKFold(n_splits=5,n_repeats=3,random_state=42).split(X))
    raw=np.zeros_like(base);rows=[]
    for i,(a,b) in enumerate(folds):
        a,b=np.sort(a),np.sort(b)
        key,_=select(X[a],y[a],5200+i)
        p=candidates(X[a],y[a],X[b])[key];raw[i//5,b]=p
        rows.append({'fold':i,'selected':key,'v10':np.sqrt(np.mean((base[i//5,b]-y[b])**2)),
                     'v11':np.sqrt(np.mean((.5*(p+base[i//5,b])-y[b])**2))})
        print('outer',i,key,round(rows[-1]['v11'],4),flush=True)
    pred={'v10':base,'nested_new_rank':raw,'v11':.5*(raw+base)}
    scores=[]
    for key,p in pred.items():
        fs=[np.sqrt(np.mean((p[i//5,b]-y[b])**2)) for i,(_,b) in enumerate(folds)]
        reps=[np.sqrt(np.mean((pp-y)**2)) for pp in p]
        scores.append({'candidate':key,'mean_fold_rmse':np.mean(fs),'mean_repeat_rmse':np.mean(reps),
                      'rep1':reps[0],'rep2':reps[1],'rep3':reps[2]})
    pd.DataFrame(scores).to_csv(n.OUT/'nested_scores.csv',index=False)
    pd.DataFrame(rows).to_csv(n.OUT/'nested_selections.csv',index=False)
    np.savez(n.OUT/'nested_predictions.npz',**pred)
    rng=np.random.default_rng(5200);a=(pred['v11']-y)**2;b=(base-y)**2
    deltas=[]
    for _ in range(5000):
        ix=rng.integers(0,len(y),len(y))
        deltas.append(np.sqrt(a[:,ix].mean())-np.sqrt(b[:,ix].mean()))
    pub=[]
    for _ in range(10000):
        r=int(rng.integers(3));ix=rng.choice(len(y),29,replace=False)
        pub.append(np.sqrt(a[r,ix].mean())-np.sqrt(b[r,ix].mean()))
    summary={'record_bootstrap_delta95':np.quantile(deltas,[.025,.975]).tolist(),
             'record_bootstrap_fraction_improves':float(np.mean(np.array(deltas)<0)),
             'conditional_public_win_fraction':float(np.mean(np.array(pub)<0)),
             'conditional_public_delta_5_50_95':np.quantile(pub,[.05,.5,.95]).tolist(),
             'fold_wins':int(sum(row['v11']<row['v10'] for row in rows))}
    (n.OUT/'uncertainty.json').write_text(json.dumps(summary,indent=2)+'\n')
    key,cv=select(X,y,5299)
    (n.OUT/'full_selection.json').write_text(json.dumps({'selected':key,'cv_rmse':cv},indent=2)+'\n')
    print(pd.DataFrame(scores).round(4).to_string(index=False),flush=True)
    print(summary,'FINAL',key,cv,flush=True)
    # Independent partition stress test for fixed rank estimators only.
    # Does not recycle foundation OOF across incompatible folds.
    stress=[]
    for i,(a,b) in enumerate(ShuffleSplit(n_splits=200,test_size=.2,random_state=520052).split(X)):
        a,b=np.sort(a),np.sort(b)
        cs=candidates(X[a],y[a],X[b])
        row={'split':i}
        for name,p in cs.items():row[name]=np.sqrt(np.mean((p-y[b])**2))
        stress.append(row)
    pd.DataFrame(stress).to_csv(n.OUT/'fresh_split_stress.csv',index=False)
    print('Fresh fixed-method splits:',pd.DataFrame(stress).drop(columns='split').mean().to_dict(),flush=True)

if __name__=='__main__':main()
