"""Nested 96-query prediction with paired, disjoint public/private simulation."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit

spec=importlib.util.spec_from_file_location('n',Path(__file__).with_name('60_nested_and_group_check.py'))
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)

def main():
    tr=pd.read_csv(n.a.s.n.m.ROOT/'data/train.csv');X=tr.filter(like='feature_').to_numpy();y=tr.target.to_numpy()
    rng=np.random.default_rng(6302);rows=[];selections=[]
    for i,(u,v) in enumerate(ShuffleSplit(n_splits=50,test_size=96,random_state=6301).split(X)):
        u,v=np.sort(u),np.sort(v);assert not set(u)&set(v)
        key,_=n.select(X[u],y[u],630100+i);selections.append(dict(split=i,selected=key))
        ps={'old_component':n.a.predict(X[u],y[u],X[v],{}),
            'nested_new_component':n.a.predict(X[u],y[u],X[v],n.CATALOG[key])}
        pools=[rng.permutation(96) for _ in range(50)]
        for name,p in ps.items():
            e=(p-y[v])**2
            for j,ix in enumerate(pools):
                assert len(set(ix[:29])&set(ix[29:]))==0
                pub,pri=np.sqrt(e[ix[:29]].mean()),np.sqrt(e[ix[29:]].mean())
                rows.append(dict(split=i,partition=j,candidate=name,query_rmse=np.sqrt(e.mean()),public=pub,private=pri,gap=abs(pub-pri)))
        if i%10==0:print('nested 96-query split',i,key,flush=True)
    df=pd.DataFrame(rows);df.to_csv(n.a.OUT/'coherent_nested_pairs.csv',index=False)
    pd.DataFrame(selections).to_csv(n.a.OUT/'coherent_nested_selection.csv',index=False)
    summary=df.groupby('candidate').agg(query_rmse=('query_rmse','mean'),private_mean=('private','mean'),
        private_p95=('private',lambda z:z.quantile(.95)),gap_mean=('gap','mean'))
    summary.to_csv(n.a.OUT/'coherent_nested_summary.csv');print(summary.to_string(),flush=True)
    fig,ax=plt.subplots(1,2,figsize=(11,4))
    df.groupby(['split','candidate']).query_rmse.first().unstack().plot(ax=ax[0]);ax[0].set_title('50 nested held-out query sets')
    for k,g in df.groupby('candidate'):ax[1].hist(g.private,bins=40,histtype='step',density=True,label=k)
    ax[1].legend();ax[1].set_title('Simulated private error, not actual leaderboard')
    fig.tight_layout();fig.savefig(n.a.OUT/'coherent_nested.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
