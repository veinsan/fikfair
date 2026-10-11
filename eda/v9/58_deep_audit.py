"""Read-only data audit with numerical artifacts and diagnostic figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/deep_v13';OUT.mkdir(exist_ok=True)

def main():
    tr=pd.read_csv(ROOT/'data/train.csv');te=pd.read_csv(ROOT/'data/test.csv')
    cols=tr.filter(like='feature_').columns;X=tr[cols].to_numpy();T=te[cols].to_numpy();y=tr.target.to_numpy()
    assert len(cols)==24 and np.isfinite(X).all() and np.isfinite(T).all() and np.isfinite(y).all()
    assert tr.record_id.is_unique and te.record_id.is_unique and not set(tr.record_id)&set(te.record_id)
    p=np.load(ROOT/'outputs/stability_search/confirmation_oof.npz')['hedge']
    assert p.shape==(3,len(tr)) and np.isfinite(p).all()
    z=(X-X.mean(0))/X.std(0);zt=(T-X.mean(0))/X.std(0)
    d=cdist(z,z);np.fill_diagonal(d,np.inf);dt=cdist(zt,z)
    ix=d.argmin(1)
    nearest=pd.DataFrame({'record_id':tr.record_id,'neighbor':tr.record_id.iloc[ix].to_numpy(),
        'distance':d.min(1),'target_difference':abs(y-y[ix])})
    nearest.sort_values('distance').to_csv(OUT/'nearest_pairs.csv',index=False)
    err=(p-y).mean(0);e2=((p-y)**2).mean(0)
    rows=tr[['record_id','target']].copy();rows['position']=np.arange(len(tr));rows['residual']=err;rows['squared_error']=e2
    rows.sort_values('squared_error',ascending=False).to_csv(OUT/'worst_records.csv',index=False)
    summary={'train_rows':len(tr),'test_rows':len(te),'missing_train':int(tr.isna().sum().sum()),'missing_test':int(te.isna().sum().sum()),
      'exact_train_duplicate_rows':int(tr.duplicated(subset=cols).sum()),
      'exact_test_duplicate_rows':int(te.duplicated(subset=cols).sum()),
      'exact_train_test_pairs':int((dt==0).sum()),'nearest_train_distance_quantiles':np.quantile(d.min(1),[0,.01,.1,.5]).tolist(),
      'nearest_test_distance_quantiles':np.quantile(dt.min(1),[0,.01,.1,.5]).tolist(),
      'first38_sse_share':float(e2[:38].sum()/e2.sum()),'top10_errors_sse_share':float(np.sort(e2)[-10:].sum()/e2.sum()),
      'spearman_position_target':float(pd.Series(np.arange(len(y))).corr(pd.Series(y),method='spearman'))}
    (OUT/'audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    corr=tr[list(cols)+['target']].corr(method='spearman');corr.to_csv(OUT/'correlation.csv')
    fig,ax=plt.subplots(2,2,figsize=(12,9))
    im=ax[0,0].imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');fig.colorbar(im,ax=ax[0,0]);ax[0,0].set_title('Spearman correlation: features then target')
    ax[0,1].scatter(np.arange(len(y)),y,s=8,label='Target');ax[0,1].plot(p.mean(0),label='v12 OOF mean');ax[0,1].legend()
    ax[1,0].scatter(np.arange(len(y)),err,s=8);ax[1,0].axhline(0,color='k');ax[1,0].set_title('OOF residual vs original position')
    ax[1,1].hist(d.min(1),bins=30,alpha=.6,label='Train nearest');ax[1,1].hist(dt.min(1),bins=30,alpha=.6,label='Test nearest');ax[1,1].legend()
    fig.tight_layout();fig.savefig(OUT/'audit.png',dpi=140);plt.close(fig)
    print(nearest.sort_values('distance').head(10).to_string(index=False));print(rows.nlargest(10,'squared_error').to_string(index=False))

if __name__=='__main__':main()
