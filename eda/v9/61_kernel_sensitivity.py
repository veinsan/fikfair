"""Sensitivity diagnostic, not an extra tuning loop for the frozen v13 rule."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit

spec=importlib.util.spec_from_file_location('a',Path(__file__).with_name('59_preprocessing_ablation.py'))
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

def main():
    tr=pd.read_csv(a.s.n.m.ROOT/'data/train.csv');X=tr.filter(like='feature_').to_numpy();y=tr.target.to_numpy()
    rows=[];paired=[];rng=np.random.default_rng(6102)
    for i,(u,v) in enumerate(ShuffleSplit(n_splits=150,test_size=96,random_state=6101).split(X)):
        u,v=np.sort(u),np.sort(v)
        configs={'baseline':{}}
        configs.update({f'a{alpha}_g{gamma}':{'kernel':True,'kernel_alpha':alpha,'gamma_scale':gamma} for alpha in [.01,.03,.1,.3,1,3] for gamma in [.5,1,2,4,8]})
        pools=[rng.permutation(96) for _ in range(20)]
        for k,c in configs.items():
            p=a.predict(X[u],y[u],X[v],c);e=(p-y[v])**2
            rows.append(dict(split=i,candidate=k,rmse=np.sqrt(e.mean())))
            if k in ['baseline','a1_g1','a0.3_g2','a0.1_g1']:
                for ix in pools:
                    pub,pri=np.sqrt(e[ix[:29]].mean()),np.sqrt(e[ix[29:]].mean())
                    paired.append(dict(split=i,candidate=k,public=pub,private=pri,gap=abs(pub-pri)))
    df=pd.DataFrame(rows);pp=pd.DataFrame(paired);df.to_csv(a.OUT/'sensitivity.csv',index=False);pp.to_csv(a.OUT/'fresh_pairs.csv',index=False)
    table=df.groupby('candidate').rmse.mean().sort_values();print(table.to_string())
    summary=pp.groupby('candidate').agg(private_mean=('private','mean'),private_p95=('private',lambda z:z.quantile(.95)),gap=('gap','mean'))
    summary.to_csv(a.OUT/'fresh_pairs_summary.csv');print(summary.to_string())
    fig,ax=plt.subplots(1,2,figsize=(12,5))
    alphas=[.01,.03,.1,.3,1,3];gammas=[.5,1,2,4,8]
    grid=np.array([[table[f'a{alpha}_g{gamma}'] for gamma in gammas] for alpha in alphas])
    im=ax[0].imshow(grid,cmap='viridis_r');fig.colorbar(im,ax=ax[0])
    ax[0].set_xticks(range(len(gammas)),gammas);ax[0].set_yticks(range(len(alphas)),alphas)
    for i in range(len(alphas)):
        for j in range(len(gammas)):ax[0].text(j,i,f'{grid[i,j]:.3f}',ha='center',va='center',fontsize=8)
    ax[0].set(title='Development RMSE: 150 partitions',xlabel='Gamma multiplier',ylabel='Kernel alpha')
    for k,g in pp.groupby('candidate'):ax[1].hist(g.gap,bins=40,histtype='step',density=True,label=k)
    ax[1].legend();ax[1].set_title('Disjoint 29/67 subset gaps')
    fig.tight_layout();fig.savefig(a.OUT/'sensitivity.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
