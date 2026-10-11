"""CPU-only exploration of the residuals left by v10.

Each prediction sees only that fold's training labels and query features/order.
Discovery scores are not a final selection-adjusted estimate.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import RepeatedKFold

spec=importlib.util.spec_from_file_location('rank48',Path(__file__).with_name('48_order_refinement.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OUT=m.ROOT/'outputs/v11_eda';OUT.mkdir(exist_ok=True)

def neighbor_signals(A,B,k=10,dim=12):
    mu,sd=A.mean(0),A.std(0)
    pc=PCA(n_components=dim,svd_solver='full').fit((A-mu)/sd)
    ZA=pc.transform((A-mu)/sd)/np.sqrt(pc.explained_variance_)
    ZB=pc.transform((B-mu)/sd)/np.sqrt(pc.explained_variance_)
    DA=cdist(ZA,ZA);np.fill_diagonal(DA,np.inf)
    DB=cdist(ZB,ZA)
    qa=m.rank(len(A))
    def local(D):
        ix=np.argpartition(D,k-1,axis=1)[:,:k]
        weights=1/np.maximum(np.take_along_axis(D,ix,axis=1),.05)**2
        return (weights*qa[ix]).sum(1)/weights.sum(1)
    return local(DA),local(DB)

def warp(A,B,k=10,dim=12,kind='gaussian',strength=1.):
    ref,query=neighbor_signals(A,B,k,dim)
    qa,qb=m.rank(len(A)),m.rank(len(B))
    trend=gaussian_filter1d(ref,.05*len(A),mode='nearest')
    error=query-np.interp(qb,qa,trend)
    if kind=='bridge':
        # Gaussian approximation to train/query empirical rank displacement.
        # min(u,v)-u*v is the Brownian bridge covariance kernel.
        K=(np.minimum.outer(qb,qb)-np.outer(qb,qb))*(1/len(A)+1/len(B))*strength
        derivative=np.clip(np.interp(qb,qa,np.gradient(trend,qa)),.1,3.)
        sigma2=max(float(np.mean((ref-trend)**2)),1e-4)
        cov=K*derivative[:,None]*derivative[None,:]+np.eye(len(B))*sigma2
        correction=(K*derivative[None,:])@np.linalg.solve(cov,error)
    else:
        correction=strength*4*qb*(1-qb)*gaussian_filter1d(error,max(1,.1*len(B)),mode='nearest')
    return IsotonicRegression(increasing=True).fit_transform(qb,np.clip(qb+correction,0,1))

def all_candidates(A,t,B):
    out={}
    for dim in [8,12,16,20]:
        u=warp(A,B,dim=dim)
        out[f'gaussian_dim{dim}']=m.curve(t,u,.005)
    out['dimension_mean']=np.mean([out[f'gaussian_dim{d}'] for d in [8,12,16,20]],axis=0)
    for scale in [.5,1.,2.]:
        u=warp(A,B,kind='bridge',strength=scale)
        out[f'bridge_{scale}']=m.curve(t,u,.005)
    rng=np.random.default_rng(5100)
    bags=[]
    for _ in range(16):
        a=np.sort(rng.choice(len(A),int(.9*len(A)),replace=False))
        bags.append(m.curve(t[a],warp(A[a],B),.005))
    out['subbag16']=np.mean(bags,axis=0)
    out['subbag_half_full']=.5*(out['subbag16']+out['gaussian_dim12'])
    return out

def main():
    tr=pd.read_csv(m.ROOT/'data/train.csv');X=tr.filter(like='feature_').to_numpy();y=tr.target.to_numpy()
    archive=np.load(m.ROOT/'outputs/order_refinement/nested_predictions.npz')
    base=archive['nested_rank_hedge_v9']
    fold=list(RepeatedKFold(n_splits=5,n_repeats=3,random_state=42).split(X))
    pred={'v10':base};rows=[]
    for i,(a,b) in enumerate(fold):
        a,b=np.sort(a),np.sort(b)
        for key,p in all_candidates(X[a],y[a],X[b]).items():
            for name,value in [(key,p),(key+'_hedge',.5*(p+base[i//5,b]))]:
                if name not in pred:pred[name]=np.zeros((3,len(y)))
                pred[name][i//5,b]=value
        print('fold',i,'done',flush=True)
    for key,p in pred.items():
        fs=[np.sqrt(np.mean((p[i//5,b]-y[b])**2)) for i,(_,b) in enumerate(fold)]
        reps=[np.sqrt(np.mean((row-y)**2)) for row in p]
        rows.append({'candidate':key,'mean_fold_rmse':np.mean(fs),'rep1':reps[0],'rep2':reps[1],'rep3':reps[2],
            'fold_wins':sum(fs[i]<np.sqrt(np.mean((base[i//5,b]-y[b])**2)) for i,(_,b) in enumerate(fold))})
    table=pd.DataFrame(rows).sort_values('mean_fold_rmse')
    table.to_csv(OUT/'discovery.csv',index=False);np.savez(OUT/'discovery_predictions.npz',**pred)
    print(table.round(4).to_string(index=False))

if __name__=='__main__':main()
