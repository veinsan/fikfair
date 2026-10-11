"""Nested selection of rank alignment; no pretrained-model training.

Outer folds are exactly those of archived OOF. Inner folds never see outer
validation features/labels when selecting a parameter. A separate fixed-method
split stress test compares rank-only alternatives without reusing old OOF.
"""
import importlib.util
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold, ShuffleSplit

spec=importlib.util.spec_from_file_location('order48',Path(__file__).with_name('48_order_refinement.py'))
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ROOT,OUT=m.ROOT,m.OUT

# Defined from a narrow follow-up to discovery, not by leaderboard scores.
CONFIGS=[(k,a,s) for k in [10,30] for a in [.6,1.,1.5] for s in [0,.005]]

def candidates(X,y,B):
    out={'quantile':m.curve(y,m.rank(len(B))),'beta':m.beta_curve(y,len(B))}
    # Cache warps across smoothing variants.
    for k in [10,30]:
        for alpha in [.6,1.,1.5]:
            u=m.neighbor_warp(X,B,k,alpha)
            for s in [0,.005]:
                out[f'k{k}_a{alpha}_s{s}']=m.curve(y,u,s)
    return out

def main():
    tr=pd.read_csv(ROOT/'data/train.csv'); X=tr.filter(like='feature_').to_numpy(); y=tr.target.to_numpy()
    archive=np.load(OUT/'discovery_predictions.npz')
    outer=list(RepeatedKFold(n_splits=5,n_repeats=3,random_state=42).split(X))
    output={'v9':archive['v9'],'quantile':archive['quantile']}
    output['nested_rank']=np.zeros((3,len(y)))
    output['nested_rank_hedge_v9']=np.zeros((3,len(y)))
    selections=[]; all_fixed={}
    for i,(a,b) in enumerate(outer):
        a,b=np.sort(a),np.sort(b); r=i//5
        accum={}
        for aa,bb in KFold(5,shuffle=True,random_state=100+i).split(a):
            aa,bb=a[np.sort(aa)],a[np.sort(bb)]
            for key,p in candidates(X[aa],y[aa],X[bb]).items():
                accum[key]=accum.get(key,0)+np.sum((p-y[bb])**2)
        chosen=min(accum,key=accum.get)
        p_all=candidates(X[a],y[a],X[b])
        p=p_all[chosen]
        output['nested_rank'][r,b]=p
        output['nested_rank_hedge_v9'][r,b]=.5*(p+archive['v9'][r,b])
        for key,pp in p_all.items():
            if key not in all_fixed: all_fixed[key]=np.zeros((3,len(y)))
            all_fixed[key][r,b]=pp
        selections.append({'fold':i,'selected':chosen,'inner_rmse':np.sqrt(accum[chosen]/len(a)),
            'outer_rmse':np.sqrt(np.mean((p-y[b])**2)),
            'v9_rmse':np.sqrt(np.mean((archive['v9'][r,b]-y[b])**2))})
        print('fold',i,'selected',chosen,'RMSE',round(selections[-1]['outer_rmse'],4),flush=True)
    def report(pool):
        rows=[]
        for key,p in pool.items():
            fold=[np.sqrt(np.mean((p[i//5,b]-y[b])**2)) for i,(_,b) in enumerate(outer)]
            rep=[np.sqrt(np.mean((pp-y)**2)) for pp in p]
            rows.append({'candidate':key,'mean_fold_rmse':np.mean(fold),'mean_repeat_rmse':np.mean(rep),
                'rep1':rep[0],'rep2':rep[1],'rep3':rep[2],
                'fold_wins_v9':sum(f<s['v9_rmse'] for f,s in zip(fold,selections))})
        return pd.DataFrame(rows).sort_values('mean_fold_rmse')
    table=report(output); table.to_csv(OUT/'nested_scores.csv',index=False)
    report(all_fixed).to_csv(OUT/'sensitivity_scores.csv',index=False)
    pd.DataFrame(selections).to_csv(OUT/'nested_selection.csv',index=False)
    np.savez(OUT/'nested_predictions.npz',**output)
    np.savez(OUT/'sensitivity_predictions.npz',**all_fixed)
    print('\nNESTED\n',table.round(4).to_string(index=False),flush=True)
    # Paired record bootstrap: keep all repeat errors for a record together.
    rng=np.random.default_rng(4900); bs=[]
    for key in ['nested_rank','nested_rank_hedge_v9']:
        se=(output[key]-y)**2; base=(output['v9']-y)**2
        differences=[]
        for _ in range(5000):
            ix=rng.integers(0,len(y),len(y))
            differences.append(np.sqrt(se[:,ix].mean())-np.sqrt(base[:,ix].mean()))
        bs.append({'candidate':key,'bootstrap_fraction_improves':float(np.mean(np.array(differences)<0)),
            'delta_rmse_95_percentile':np.quantile(differences,[.025,.975]).tolist()})
    (OUT/'nested_bootstrap.json').write_text(json.dumps(bs,indent=2)+'\n')
    print('\nBOOTSTRAP',json.dumps(bs),flush=True)
    # Fresh partitions, fixed method: replication on same dataset, not a new holdout.
    rows=[]
    for i,(a,b) in enumerate(ShuffleSplit(n_splits=200,test_size=.2,random_state=20261005).split(X)):
        a,b=np.sort(a),np.sort(b)
        q=m.curve(y[a],m.rank(len(b)))
        u=m.neighbor_warp(X[a],X[b],10,1.)
        w=m.curve(y[a],u)
        rows.append({'split':i,'quantile_rmse':np.sqrt(np.mean((y[b]-q)**2)),
            'warp_rmse':np.sqrt(np.mean((y[b]-w)**2))})
    stress=pd.DataFrame(rows); stress.to_csv(OUT/'fresh_split_stress.csv',index=False)
    print('\nFRESH SPLITS',stress[['quantile_rmse','warp_rmse']].mean().to_dict(),
        'win fraction',float((stress.warp_rmse<stress.quantile_rmse).mean()),flush=True)
    # Fit configuration selection on full training via inner CV, without test labels.
    accum={}
    for a,b in KFold(5,shuffle=True,random_state=4900).split(X):
        a,b=np.sort(a),np.sort(b)
        for key,p in candidates(X[a],y[a],X[b]).items():
            accum[key]=accum.get(key,0)+np.sum((p-y[b])**2)
    selected=min(accum,key=accum.get)
    (OUT/'full_training_selection.json').write_text(json.dumps({'selected':selected,'cv_rmse':{k:float(np.sqrt(v/len(y))) for k,v in accum.items()}},indent=2)+'\n')
    print('\nFULL-TRAINING SELECTION',selected,flush=True)

if __name__=='__main__':main()
