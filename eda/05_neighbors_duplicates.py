"""05 - Near-duplicates & neighbourhood structure (leakage risk).
478 records of env/climate indicators could contain the same entity several times (e.g. region x year).
If near-identical rows exist, random KFold leaks: the model memorises the twin and CV looks too good.
We measure NN distances train-train vs test-train and build similarity groups for GroupKFold."""
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from common import h, load, plt, save

tr, te, F = load()
sc = StandardScaler().fit(pd.concat([tr[F], te[F]]))
Xtr, Xte, y = sc.transform(tr[F]), sc.transform(te[F]), tr.target.values

h("NEAREST-NEIGHBOUR DISTANCES (standardized euclidean)")
nn = NearestNeighbors(n_neighbors=6).fit(Xtr)
d_trtr, i_trtr = nn.kneighbors(Xtr)
d_trtr, i_trtr = d_trtr[:, 1:], i_trtr[:, 1:]
d_tetr, i_tetr = nn.kneighbors(Xte, n_neighbors=5)
print("train->train 1-NN dist: ", np.percentile(d_trtr[:, 0], [0, 1, 5, 25, 50, 75, 95]).round(3))
print("test ->train 1-NN dist: ", np.percentile(d_tetr[:, 0], [0, 1, 5, 25, 50, 75, 95]).round(3))
# pure random points in 24-D would have 1-NN dist roughly this:
rnd = np.random.default_rng(0).standard_normal(Xtr.shape)
dr = NearestNeighbors(n_neighbors=2).fit(rnd).kneighbors(rnd)[0][:, 1]
print("iid-gaussian 1-NN dist: ", np.percentile(dr, [0, 5, 50, 95]).round(3), "(reference)")

h("DOES THE NEIGHBOUR'S TARGET PREDICT MY TARGET? (smoothness / twin leakage)")
for k in [1, 3, 5]:
    p = y[i_trtr[:, :k]].mean(1)
    print(f"  {k}-NN leave-one-out RMSE = {np.sqrt(np.mean((y - p) ** 2)):.3f}")
diff1 = np.abs(y - y[i_trtr[:, 0]])
q = pd.qcut(d_trtr[:, 0], 5)
print("\n|y - y_nn1| by NN-distance quintile:")
print(pd.DataFrame({"q": q, "absdiff": diff1}).groupby("q", observed=True).absdiff.describe()[["count", "mean", "50%"]].round(3).to_string())

h("SIMILARITY GROUPS for GroupKFold")
for thr in [0.5, 1.0, 1.5, 2.0, 2.5]:
    A = (NearestNeighbors(radius=thr).fit(Xtr).radius_neighbors_graph(Xtr, mode="connectivity"))
    n, lab = connected_components(A, directed=False)
    sizes = np.bincount(lab)
    print(f"  radius {thr:.1f}: {n:3d} groups | largest={sizes.max():3d} | singletons={(sizes == 1).sum():3d} | rows in groups>=2: {(sizes[lab] > 1).sum()}")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].hist(d_trtr[:, 0], bins=40, alpha=0.6, density=True, label="train->train")
ax[0].hist(d_tetr[:, 0], bins=40, alpha=0.6, density=True, label="test->train")
ax[0].hist(dr, bins=40, alpha=0.4, density=True, label="iid gaussian ref")
ax[0].set_title("1-NN distance"); ax[0].legend()
ax[1].scatter(d_trtr[:, 0], diff1, s=8); ax[1].set_xlabel("1-NN distance"); ax[1].set_ylabel("|y - y_nn|")
ax[1].set_title("closer twin -> closer target?")
ax[2].scatter(y[i_trtr[:, 0]], y, s=8); ax[2].plot([30, 101], [30, 101], "r--")
ax[2].set_xlabel("target of nearest neighbour"); ax[2].set_ylabel("target"); ax[2].set_title("1-NN target agreement")
save(fig, "05_neighbors.png")
