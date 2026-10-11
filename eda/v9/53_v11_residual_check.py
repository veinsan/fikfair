"""Test whether features explain departures from the ordered target trend."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold

spec=importlib.util.spec_from_file_location('rank51',Path(__file__).with_name('51_v11_rank_eda.py'))
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)

def residual_candidates(A,t,B):
    qa,qb=n.m.rank(len(A)),n.warp(A,B)
    mu,sd=A.mean(0),A.std(0)
    Z=(A-mu)/sd;Q=(B-mu)/sd
    smooth_x=gaussian_filter1d(Z,.05*len(A),axis=0,mode='nearest')
    smooth_y=gaussian_filter1d(t,.05*len(A),mode='nearest')
    RX=Z-smooth_x
    RQ=Q-np.column_stack([np.interp(qb,qa,smooth_x[:,j]) for j in range(Z.shape[1])])
    base=n.m.curve(t,qb,.005)
    out={'none':base}
    for alpha in [10,100]:
        model=Ridge(alpha=alpha).fit(RX,t-smooth_y)
        residual=model.predict(RQ)
        for weight in [.5,1.]:out[f'ridge{alpha}_w{weight}']=base+weight*residual
    return out

def main():
    tr=pd.read_csv(n.m.ROOT/'data/train.csv');X=tr.filter(like='feature_').to_numpy();y=tr.target.to_numpy()
    v10=np.load(n.m.ROOT/'outputs/order_refinement/nested_predictions.npz')['nested_rank_hedge_v9']
    folds=list(RepeatedKFold(n_splits=5,n_repeats=3,random_state=42).split(X))
    pred={};rows=[]
    for i,(a,b) in enumerate(folds):
        a,b=np.sort(a),np.sort(b)
        for key,p in residual_candidates(X[a],y[a],X[b]).items():
            if key not in pred:pred[key]=np.zeros_like(v10)
            pred[key][i//5,b]=.5*(p+v10[i//5,b])
    for key,p in pred.items():
        fs=[np.sqrt(np.mean((p[i//5,b]-y[b])**2)) for i,(_,b) in enumerate(folds)]
        rows.append({'candidate':key,'mean_fold_rmse':np.mean(fs),
          'rep1':np.sqrt(np.mean((p[0]-y)**2)),'rep2':np.sqrt(np.mean((p[1]-y)**2)),'rep3':np.sqrt(np.mean((p[2]-y)**2))})
    table=pd.DataFrame(rows);table.to_csv(n.OUT/'residual_candidates.csv',index=False)
    print(table.round(4).to_string(index=False))

if __name__=='__main__':main()
