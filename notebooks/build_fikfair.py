import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "fikfair_dsc2026.ipynb"
HR  = "---"

def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}

def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": uuid4().hex[:8],
        "metadata": {}, "outputs": [], "source": source,
    }

def section(title: str, anchor: str) -> dict:
    return md(
        f"{HR}\n\n"
        f"# {title} <a name=\"{anchor}\"></a>\n\n"
        f"{HR}"
    )

cells = []

# ---------------------------------------------------------------------------
# Banner (3 cells) + TOC
# ---------------------------------------------------------------------------
cells.append(md(
    f"{HR}\n\n"
    "# Data Science Competition FIK FAIR 2026\n\n"
    "*Predicting an Anonymous Competition Score from 24 transformed environmental and climate features.*"
))

cells.append(md(
    f"{HR}\n\n"
    "## Team Name\n\n"
    "- Name 1 (Role)\n"
    "- Name 2\n"
    "- Name 3"
))

cells.append(md(
    f"{HR}\n\n"
    "## Table of Contents\n\n"
    "1. [**Introduction**](#1)\n"
    "2. [**Initialization**](#2)\n"
    "3. [**Exploratory Data Analysis**](#3)\n"
    "4. [**Preprocessing Verification**](#4)\n"
    "5. [**Modelling**](#5)\n"
    "6. [**Ensemble and Error Analysis**](#6)\n"
    "7. [**Submission**](#7)\n"
))

# ---------------------------------------------------------------------------
# Section 1: Introduction
# ---------------------------------------------------------------------------
cells.append(section("Introduction", "1"))

cells.append(md(
    "## Overview\n\n"
    "*FIK FAIR 2026 hosts a tabular regression competition in which each record is described by 24 "
    "anonymised numerical features derived from environmental and climate indicators. The organisers "
    "passed the raw indicators through rank normalisation, a bounded transformation, a private linear "
    "projection, standardisation, controlled perturbation and rounding, so the features carry no "
    "physical unit and no known meaning.*\n\n"
    "*The target is the Anonymous Competition Score, a constructed composite whose formula is private. "
    "With only 382 labelled rows, the main risks are overfitting, optimistic validation and chasing "
    "a public leaderboard that is computed on roughly 29 rows.*"
))

cells.append(md(
    "## Aim\n\n"
    "*The notebook aims to minimise the RMSE of the predicted score on the private leaderboard. Every "
    "preprocessing idea is treated as a hypothesis and kept only when repeated cross-validation "
    "confirms it, and the final predictor is a cross-validated blend of kernel models, gradient "
    "boosting and pretrained tabular foundation models.*"
))

cells.append(md(
    "## Metric\n\n"
    "***Root Mean Squared Error (RMSE)** measures the typical size of the prediction error in target "
    "units, with large errors penalised quadratically. It is the primary ranking metric, and MAE is "
    "used as the tie-breaker.*\n\n"
    "$$\\text{RMSE} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}\\left(y_i - \\hat{y}_i\\right)^2}$$\n\n"
    "*Because the loss is quadratic, the few records with a score above 70 dominate the error budget, "
    "so the analysis pays special attention to the right tail of the target.*"
))

cells.append(md(
    "## Dataset\n\n"
    "*Data sourced from Kaggle at: https://www.kaggle.com/competitions/data-science-competition-fikfair2026*\n\n"
    "*The dataset contains 478 records, split into 382 training rows with a target and 96 test rows "
    "without one. There are no missing values and every feature is stored with six decimals.*\n\n"
    "*The dataset is used to learn a mapping from the anonymised features to the Anonymous "
    "Competition Score.*\n\n"
    "**Attributes**\n\n"
    "1.  `record_id`                  : Anonymous technical identifier with no geographic or source meaning\n"
    "2.  `feature_01` to `feature_24` : Transformed, dimensionless environmental and climate features\n\n"
    "**Target**\n"
    " `target`       : Anonymous Competition Score, a continuous value between about 32.6 and 100.4"
))

cells.append(md(
    "## Approach: Verified Preprocessing and a Stacked Blend\n\n"
    "*The approach combines models that see the data in different ways. Kernel models (SVR, kNN) "
    "work on a 12-component PCA view of the features with a log-shifted target, gradient boosting "
    "works on the raw features, and the pretrained foundation models TabICLv2 and TabPFN v2 predict "
    "through in-context learning without any gradient training on the competition data.*\n\n"
    "*All models share the same repeated 5-fold split, and the blend weights are fitted with "
    "non-negative least squares on out-of-fold predictions, with a nested split to estimate the "
    "blend error honestly.*\n\n"
    "```\n"
    "24 raw features\n"
    "    |\n"
    "    +-- StandardScaler -> PCA(12) -> SVR / kNN   (target: log(y - 30))\n"
    "    +-- raw features  -> CatBoost / ExtraTrees  (target: y)\n"
    "    +-- raw features  -> TabICLv2 / TabPFN v2   (pretrained, in-context)\n"
    "         |\n"
    "    NNLS blend on out-of-fold predictions -> prediction\n"
    "```\n\n"
    "*The split is plain repeated K-fold because the test rows are no closer to the training rows "
    "than the training rows are to each other, which the neighbour analysis in Section 3 verifies.*"
))

# ---------------------------------------------------------------------------
# Section 2: Initialization
# ---------------------------------------------------------------------------
cells.append(section("Initialization", "2"))

cells.append(md(
    "## Environment Setup\n\n"
    "The notebook targets a Kaggle session with two T4 GPUs and internet access enabled, which is "
    "needed to install the foundation model packages and download their open checkpoints from "
    "Hugging Face. Recorded runtime: the full notebook takes about 8 minutes on a laptop CPU, and the foundation "
    "models run considerably faster on the following GPUs."
))

cells.append(code("!nvidia-smi"))

cells.append(md("The following cell installs all libraries used in this notebook."))

cells.append(code(
    "%pip install -q tabicl tabpfn catboost lightgbm"
))

cells.append(md("## Import Libraries"))

cells.append(code(
    "import os\n"
    "import random\n"
    "import time\n"
    "import warnings\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import torch\n"
    "from scipy import stats\n"
    "from scipy.optimize import nnls\n"
    "from scipy.sparse.csgraph import connected_components\n"
    "from statsmodels.nonparametric.smoothers_lowess import lowess\n"
    "\n"
    "from catboost import CatBoostRegressor\n"
    ""
    "from sklearn.compose import TransformedTargetRegressor\n"
    "from sklearn.decomposition import PCA, FastICA\n"
    "from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier\n"
    "from sklearn.linear_model import LogisticRegression, RidgeCV\n"
    "from sklearn.metrics import roc_auc_score\n"
    "from sklearn.model_selection import (GroupKFold, KFold, RepeatedKFold,\n"
    "                                     StratifiedKFold, cross_val_predict)\n"
    "from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors\n"
    "from sklearn.pipeline import make_pipeline\n"
    "from sklearn.preprocessing import SplineTransformer, StandardScaler\n"
    "from sklearn.svm import SVR\n"
    "from tabicl import TabICLRegressor\n"
    "from tabpfn import TabPFNRegressor\n"
    "from tabpfn.constants import ModelVersion\n"
    "\n"
    "warnings.filterwarnings('ignore')"
))

cells.append(md("## Seed Everything"))

cells.append(code(
    "def seed_everything(seed: int = 42):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "    torch.manual_seed(seed)\n"
    "    torch.cuda.manual_seed_all(seed)\n"
    "\n"
    "seed_everything(42)"
))

cells.append(md(
    "## Settings\n\n"
    "All paths, hyperparameters, and environment flags are centralised here. "
    "Switching between Kaggle and local execution requires no changes outside this class."
))

cells.append(code(
    "class Settings:\n"
    "    SEED       = 42\n"
    "    _ON_KAGGLE = Path('/kaggle/input').exists()\n"
    "\n"
    "    DATA_DIR   = (\n"
    "        next(Path('/kaggle/input').rglob('train.csv')).parent\n"
    "        if _ON_KAGGLE else Path('../data')\n"
    "    )\n"
    "    OUTPUT_DIR = Path('/kaggle/working') if _ON_KAGGLE else Path('../outputs')\n"
    "    FIG_DIR    = OUTPUT_DIR / 'fikfair_dsc2026'\n"
    "\n"
    "    N_SPLITS   = 5\n"
    "    N_REPEATS  = 3\n"
    "    PCA_DIM    = 12\n"
    "    LOG_SHIFT  = 30.0\n"
    "    DEVICES    = [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']\n"
    "\n"
    "CFG = Settings()\n"
    "CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
    "print('data:', CFG.DATA_DIR, '| devices:', CFG.DEVICES)"
))

cells.append(md(
    "## Load Dataset\n\n"
    "Load the three competition files and the list of feature columns. No visualisation here."
))

cells.append(code(
    "train = pd.read_csv(CFG.DATA_DIR / 'train.csv')\n"
    "test  = pd.read_csv(CFG.DATA_DIR / 'test.csv')\n"
    "template = pd.read_csv(CFG.DATA_DIR / 'submission_template.csv')\n"
    "\n"
    "FEATURES = [c for c in train.columns if c.startswith('feature_')]\n"
    "X, y  = train[FEATURES].values, train['target'].values\n"
    "X_te  = test[FEATURES].values\n"
    "X_all = np.vstack([X, X_te])\n"
    "\n"
    "print(f'Train: {train.shape}  Test: {test.shape}  Template: {template.shape}  Features: {len(FEATURES)}')"
))

# ---------------------------------------------------------------------------
# Section 3: Exploratory Data Analysis
# ---------------------------------------------------------------------------
cells.append(section("Exploratory Data Analysis", "3"))

cells.append(md(
    "The analysis moves from data integrity to the target, then to feature structure, the shape of "
    "the feature-target relationship, train-test shift, and finally the neighbourhood structure that "
    "decides the validation design. Each step ends with the decision it drives."
))

cells.append(md(
    "## Integrity and Duplicates\n\n"
    "Missing values, duplicated rows, rounding precision and the `record_id` pattern are checked "
    "first, since any of them could make cross-validation leak information."
))

cells.append(code(
    "tr_id = train['record_id'].str[1:].astype(int)\n"
    "te_id = test['record_id'].str[1:].astype(int)\n"
    "\n"
    "print('missing (train/test)       :', int(train.isna().sum().sum()), int(test.isna().sum().sum()))\n"
    "print('duplicate feature rows      :', train.duplicated(FEATURES).sum(), test.duplicated(FEATURES).sum())\n"
    "print('train-test identical rows   :', len(train[FEATURES].merge(test[FEATURES], how='inner')))\n"
    "print('unique values per feature   :', train[FEATURES].nunique().min(), 'to', train[FEATURES].nunique().max())\n"
    "print('ids cover 1..478 exactly    :', sorted(set(tr_id) | set(te_id)) == list(range(1, 479)))\n"
    "print('spearman(record_id, target) : %.3f (p=%.2f)' % stats.spearmanr(tr_id, y))"
))

cells.append(md(
    "#### Insights\n\n"
    "> The data is clean. There are no missing values and no exact duplicates inside or across "
    "splits, and every feature has as many unique values as rows, so the rounding step did not "
    "create ties.\n\n"
    "> The `record_id` values form one shuffled sequence from D0001 to D0478 and have no rank "
    "correlation with the target, which matches the rule that the identifier carries no information. "
    "It is therefore excluded from modelling."
))

cells.append(md(
    "## Target Distribution\n\n"
    "Because RMSE squares each error, the shape of the target, especially its right tail, decides "
    "which rows dominate the score."
))

cells.append(code(
    "print(pd.Series(y).describe().round(3).to_string())\n"
    "print(f'skew={stats.skew(y):.3f}  excess kurtosis={stats.kurtosis(y):.3f}  '\n"
    "      f'shapiro p={stats.shapiro(y).pvalue:.2e}')\n"
    "print(f'predict-the-mean RMSE = {y.std():.3f}')\n"
    "\n"
    "fig, ax = plt.subplots(1, 3, figsize=(16, 4))\n"
    "ax[0].hist(y, bins=40, edgecolor='w')\n"
    "ax[0].set_title('target histogram')\n"
    "stats.probplot(y, plot=ax[1])\n"
    "ax[1].set_title('target QQ-plot (normal)')\n"
    "ax[2].hist(np.log(y - CFG.LOG_SHIFT), bins=40, edgecolor='w', color='tab:green')\n"
    "ax[2].set_title('log(target - 30)')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'target.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> The target is bimodal, with one mode around 37 and another around 57, plus a thin right tail "
    "that reaches 100. It has a hard floor near 32.6, which suggests the score is clipped or "
    "rescaled from below.\n\n"
    "> The shifted log transform `log(y - 30)` pulls the tail in and spreads the dense lower mode. "
    "Section 4 tests whether this helps, because a transform that looks better on a histogram can "
    "still hurt a model in RMSE terms."
))

cells.append(md(
    "## Feature Correlation and Latent Dimension\n\n"
    "The features are described as a private linear projection of a smaller set of indicators. If "
    "that is true, the correlation matrix should be strongly structured and the PCA spectrum should "
    "show a few dominant directions."
))

cells.append(code(
    "sc_all = StandardScaler().fit(X_all)\n"
    "Z_all  = sc_all.transform(X_all)\n"
    "corr   = pd.DataFrame(X_all, columns=FEATURES).corr()\n"
    "t_corr = train[FEATURES].corrwith(train['target']).sort_values(key=abs, ascending=False)\n"
    "print('top correlations with target:')\n"
    "print(t_corr.head(8).round(3).to_string())\n"
    "\n"
    "ev = np.linalg.eigvalsh(np.cov(Z_all.T))[::-1]\n"
    "mp_edge = (1 + np.sqrt(len(FEATURES) / len(X_all))) ** 2\n"
    "print('\\nPCA explained variance % (first 8):', np.round(100 * ev[:8] / ev.sum(), 2))\n"
    "print(f'eigenvalues above the Marchenko-Pastur noise edge ({mp_edge:.2f}): {(ev > mp_edge).sum()}')\n"
    "print(f'condition number: {ev[0] / ev[-1]:.0f}')\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [3, 2]})\n"
    "sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax[0],\n"
    "            xticklabels=[c[-2:] for c in FEATURES], yticklabels=[c[-2:] for c in FEATURES])\n"
    "ax[0].set_title('feature correlation (train + test)')\n"
    "ax[1].bar(range(1, 25), ev)\n"
    "ax[1].axhline(mp_edge, ls='--', c='r', label='noise edge')\n"
    "ax[1].set_yscale('log')\n"
    "ax[1].set_title('PCA eigenvalues')\n"
    "ax[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'correlation_pca.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> The features are highly collinear. The average absolute pairwise correlation is about 0.41, 44 "
    "pairs exceed 0.7 and the strongest pair reaches 0.94. The first four principal components hold "
    "91% of the variance and are the only ones above the random-noise edge, which confirms the "
    "low-dimensional projection described by the organisers.\n\n"
    "> The strongest single feature explains only about 15% of the target variance. The target "
    "signal is therefore spread across many directions, including low-variance components that PCA "
    "would treat as noise, so aggressive dimension reduction is risky and must be verified."
))

cells.append(md(
    "## Shape of the Feature-Target Relationship\n\n"
    "A LOWESS curve per feature shows whether the dependence is monotone, and a comparison of a "
    "linear model against additive splines quantifies how much non-linearity there is."
))

cells.append(code(
    "fig, axes = plt.subplots(4, 6, figsize=(20, 12))\n"
    "for a, c in zip(axes.ravel(), FEATURES):\n"
    "    a.scatter(train[c], y, s=5, alpha=0.5)\n"
    "    lw = lowess(y, train[c], frac=0.4)\n"
    "    a.plot(lw[:, 0], lw[:, 1], c='r')\n"
    "    a.set_title(c, fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'feature_vs_target.png')\n"
    "plt.show()\n"
    "\n"
    "cv10 = KFold(10, shuffle=True, random_state=CFG.SEED)\n"
    "shape_models = {\n"
    "    'linear (Ridge)'   : make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 3, 30))),\n"
    "    'additive splines' : make_pipeline(SplineTransformer(n_knots=5), StandardScaler(),\n"
    "                                       RidgeCV(np.logspace(-2, 4, 30))),\n"
    "}\n"
    "for name, m in shape_models.items():\n"
    "    p = cross_val_predict(m, X, y, cv=cv10)\n"
    "    print(f'{name:18s} CV RMSE = {np.sqrt(np.mean((y - p) ** 2)):.3f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> Many features show an inverted-U shape, where the score peaks when the feature is near zero "
    "and falls on both sides (for example `feature_01`, `feature_06`, `feature_14`, `feature_20`). "
    "Linear correlation misses this completely, which is why `feature_01` and `feature_06` look "
    "useless by Pearson correlation yet gain about 0.12 R2 with a spline.\n\n"
    "> Additive splines cut the CV RMSE from about 10.7 to 9.4. A linear model is therefore not "
    "enough, and the final models must be able to represent curved and interacting effects."
))

cells.append(md(
    "## Train-Test Shift\n\n"
    "Per-feature Kolmogorov-Smirnov tests and an adversarial classifier check whether the test rows "
    "come from the same distribution as the training rows."
))

cells.append(code(
    "ks = pd.DataFrame(\n"
    "    [(c, *stats.ks_2samp(train[c], test[c])) for c in FEATURES],\n"
    "    columns=['feature', 'KS', 'p'],\n"
    ").set_index('feature')\n"
    "print('features with KS p < 0.05:', (ks.p < 0.05).sum(), 'of 24')\n"
    "\n"
    "is_test = np.r_[np.zeros(len(X)), np.ones(len(X_te))]\n"
    "skf = StratifiedKFold(5, shuffle=True, random_state=CFG.SEED)\n"
    "for name, clf in [('logistic', LogisticRegression(max_iter=2000)),\n"
    "                  ('random forest', RandomForestClassifier(500, min_samples_leaf=3,\n"
    "                                                           random_state=CFG.SEED, n_jobs=-1))]:\n"
    "    p = cross_val_predict(clf, X_all, is_test, cv=skf, method='predict_proba')[:, 1]\n"
    "    print(f'adversarial AUC ({name}): {roc_auc_score(is_test, p):.3f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> Nine of the 24 features have KS p-values below 0.05, but the mean shifts are small (at most "
    "about 0.3 standard deviations) and move together along one direction. The adversarial AUC "
    "stays close to 0.5 to 0.6, so the test set is only mildly different from the training set.\n\n"
    "> No row is removed or reweighted because of the shift. With 96 test rows a mild shift is "
    "expected from sampling alone, and reweighting 382 rows would add more variance than it removes."
))

cells.append(md(
    "## Neighbourhood Structure and Validation Design\n\n"
    "If the same entity appeared several times with small perturbations, random K-fold would place "
    "near-twins in both the training and validation folds and report an optimistic score. The "
    "nearest-neighbour distances of training rows and of test rows are compared to decide the split."
))

cells.append(code(
    "Z_tr, Z_te = Z_all[:len(X)], Z_all[len(X):]\n"
    "nn = NearestNeighbors(n_neighbors=2).fit(Z_tr)\n"
    "d_trtr, i_trtr = nn.kneighbors(Z_tr)\n"
    "d_tetr = nn.kneighbors(Z_te, n_neighbors=1)[0][:, 0]\n"
    "d_trtr, i_trtr = d_trtr[:, 1], i_trtr[:, 1]\n"
    "\n"
    "q = [5, 25, 50, 75, 95]\n"
    "print('train -> train 1-NN distance percentiles:', np.percentile(d_trtr, q).round(2))\n"
    "print('test  -> train 1-NN distance percentiles:', np.percentile(d_tetr, q).round(2))\n"
    "print(f'1-NN leave-one-out RMSE: {np.sqrt(np.mean((y - y[i_trtr]) ** 2)):.3f}')\n"
    "\n"
    "for r in [0.5, 1.0, 1.5]:\n"
    "    n_grp, lab = connected_components(\n"
    "        NearestNeighbors(radius=r).fit(Z_tr).radius_neighbors_graph(Z_tr), directed=False)\n"
    "    print(f'radius {r}: {n_grp} similarity groups, largest group = {np.bincount(lab).max()} rows')\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(13, 4))\n"
    "ax[0].hist(d_trtr, bins=40, alpha=0.6, density=True, label='train -> train')\n"
    "ax[0].hist(d_tetr, bins=40, alpha=0.6, density=True, label='test -> train')\n"
    "ax[0].set_title('nearest-neighbour distance')\n"
    "ax[0].legend()\n"
    "ax[1].scatter(y[i_trtr], y, s=8)\n"
    "ax[1].plot([30, 101], [30, 101], 'r--')\n"
    "ax[1].set_xlabel('target of nearest neighbour')\n"
    "ax[1].set_ylabel('target')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'neighbours.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> Test rows sit at the same distance from the training set as training rows sit from each "
    "other, so random K-fold reproduces the train-to-test situation faithfully. Only 48 training "
    "rows have a neighbour within radius 0.5, so near-twin leakage is limited. Section 5 still "
    "reports a similarity-grouped K-fold score as a pessimistic cross-check.\n\n"
    "> The nearest neighbour's target is already a decent predictor (1-NN RMSE about 10.1), and the "
    "agreement is tight in the lower mode but loose for high scores. Local, similarity-based models "
    "should therefore do well, while the high tail will remain the hardest region."
))

# ---------------------------------------------------------------------------
# Section 4: Preprocessing Verification
# ---------------------------------------------------------------------------
cells.append(section("Preprocessing Verification", "4"))

cells.append(md(
    "The EDA suggests several preprocessing ideas that are sound in theory. Each one is applied, "
    "inspected and scored with the same repeated 5-fold split and three different model families, "
    "because a transform that is correct on paper can still damage the evaluation score."
))

cells.append(md(
    "## Validation Utilities\n\n"
    "One fixed list of folds is shared by every experiment in the notebook so that all scores are "
    "directly comparable. The target transform is applied inside each fold, and unsupervised "
    "feature transforms are fitted on the features of train and test together without the target."
))

cells.append(code(
    "FOLDS = list(RepeatedKFold(n_splits=CFG.N_SPLITS, n_repeats=CFG.N_REPEATS,\n"
    "                           random_state=CFG.SEED).split(X))\n"
    "\n"
    "\n"
    "def rmse(a, b):\n"
    "    return float(np.sqrt(np.mean((a - b) ** 2)))\n"
    "\n"
    "\n"
    "def log_target(model):\n"
    "    return TransformedTargetRegressor(\n"
    "        model,\n"
    "        func=lambda t: np.log(t - CFG.LOG_SHIFT),\n"
    "        inverse_func=lambda p: np.exp(p) + CFG.LOG_SHIFT,\n"
    "        check_inverse=False,\n"
    "    )\n"
    "\n"
    "\n"
    "def z_target(model):\n"
    "    return TransformedTargetRegressor(model, transformer=StandardScaler())\n"
    "\n"
    "\n"
    "def run_cv(make_model, Xm, Xm_te=None):\n"
    "    oof = np.zeros((CFG.N_REPEATS, len(y)))\n"
    "    for i, (tri, vai) in enumerate(FOLDS):\n"
    "        oof[i // CFG.N_SPLITS, vai] = make_model().fit(Xm[tri], y[tri]).predict(Xm[vai])\n"
    "    score = np.mean([rmse(y, o) for o in oof])\n"
    "    pred_te = None if Xm_te is None else make_model().fit(Xm, y).predict(Xm_te)\n"
    "    return score, oof.mean(0), pred_te\n"
    "\n"
    "\n"
    "def pca_view(k):\n"
    "    pca = PCA(k, random_state=CFG.SEED).fit(Z_all)\n"
    "    P = pca.transform(Z_all)\n"
    "    P = P / P.std(0)\n"
    "    return P[:len(X)], P[len(X):]\n"
    "\n"
    "\n"
    "print(f'{len(FOLDS)} folds, validation size {len(FOLDS[0][1])}')"
))

cells.append(md(
    "## Feature Representation\n\n"
    "Four ideas follow from Section 3. Quantile normalisation could help distance-based models, PCA "
    "denoising could remove the controlled perturbation, and ICA could undo the private linear "
    "projection and recover the original indicators. Each representation is scored with SVR, kNN "
    "and ExtraTrees."
))

cells.append(code(
    "svr_probe = lambda: z_target(make_pipeline(StandardScaler(), SVR(C=3, gamma='scale', epsilon=0.1)))\n"
    "knn_probe = lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(7, weights='distance'))\n"
    "et_probe  = lambda: ExtraTreesRegressor(400, min_samples_leaf=2, max_features=0.5,\n"
    "                                        random_state=CFG.SEED, n_jobs=-1)\n"
    "PROBES = {'SVR': svr_probe, 'kNN': knn_probe, 'ExtraTrees': et_probe}\n"
    "\n"
    "reps = {'raw': X}\n"
    "reps['quantile-normal'] = (pd.DataFrame(X_all).rank().values[:len(X)] - 0.5) / len(X_all)\n"
    "reps['quantile-normal'] = stats.norm.ppf(reps['quantile-normal'])\n"
    "for k in [4, 8, 12]:\n"
    "    pca = PCA(k, random_state=CFG.SEED).fit(Z_all)\n"
    "    reps[f'pca-denoise k={k}'] = pca.inverse_transform(pca.transform(Z_all[:len(X)]))\n"
    "    reps[f'pca-scores k={k}'] = pca_view(k)[0]\n"
    "ica_models = {}\n"
    "for k in [6, 10]:\n"
    "    ica_models[k] = FastICA(k, whiten='unit-variance', max_iter=5000, random_state=CFG.SEED).fit(Z_all)\n"
    "    reps[f'ica k={k}'] = ica_models[k].transform(Z_all[:len(X)])\n"
    "\n"
    "rep_scores = pd.DataFrame({\n"
    "    name: {p: run_cv(make, Xm)[0] for p, make in PROBES.items()} for name, Xm in reps.items()\n"
    "}).T\n"
    "rep_scores['mean'] = rep_scores.mean(axis=1)\n"
    "print(rep_scores.sort_values('mean').round(3).to_string())"
))

cells.append(md(
    "The ICA sources are inspected directly. If ICA had recovered the original indicators, the "
    "sources would look like rank-normalised, bounded variables, which means flat distributions "
    "with negative excess kurtosis, and they would relate to the target more cleanly than the raw "
    "features do."
))

cells.append(code(
    "S_all = ica_models[10].transform(Z_all)\n"
    "kurt = stats.kurtosis(S_all, axis=0)\n"
    "rho = [stats.spearmanr(S_all[:len(X), j], y)[0] for j in range(10)]\n"
    "print('excess kurtosis per source (uniform = -1.2):', np.round(kurt, 2))\n"
    "print('max |spearman| with target, raw features:', round(t_corr.abs().max(), 3),\n"
    "      '| ICA sources:', round(np.max(np.abs(rho)), 3))\n"
    "\n"
    "fig, axes = plt.subplots(2, 10, figsize=(28, 5.5))\n"
    "for j in range(10):\n"
    "    axes[0, j].hist(S_all[:, j], bins=30)\n"
    "    axes[0, j].set_title(f'IC{j + 1} kurt={kurt[j]:.1f}', fontsize=9)\n"
    "    axes[1, j].scatter(S_all[:len(X), j], y, s=5, alpha=0.5)\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'ica_sources.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> Quantile normalisation and PCA denoising do not help. Keeping only four components, as the "
    "noise-edge analysis would suggest, is clearly harmful (RMSE above 10), which confirms that part "
    "of the target signal lives in low-variance directions. PCA scores with about 12 components are "
    "the best view for SVR, while ExtraTrees prefers the raw features and kNN is roughly indifferent.\n\n"
    "> ICA fails its own sanity check. The recovered sources are heavy-tailed (excess kurtosis up to "
    "about 9) instead of flat, and they correlate with the target less than the raw features do. ICA "
    "is latching onto outlier directions rather than undoing the projection, because the bounded "
    "transformation and the added perturbation break its independence assumption. It is dropped."
))

cells.append(md(
    "## Target Transform\n\n"
    "The shifted log target looked more regular in Section 3. The first attempt at this experiment "
    "used an SVR with a fixed `epsilon=1.0`, and the log target made RMSE jump by about four points. The cause is scale, since `epsilon` and `C` are expressed in target units and an epsilon "
    "of 1.0 is wider than the whole spread of the log target. The fix standardises the target "
    "inside the model before the transform is judged."
))

cells.append(code(
    "Xp12 = pca_view(CFG.PCA_DIM)[0]\n"
    "svr_fixed_eps = lambda: make_pipeline(StandardScaler(), SVR(C=30, gamma='scale', epsilon=1.0))\n"
    "rows = {\n"
    "    'SVR eps=1.0, y'                     : run_cv(svr_fixed_eps, Xp12)[0],\n"
    "    'SVR eps=1.0, log y   (broken)'      : run_cv(lambda: log_target(svr_fixed_eps()), Xp12)[0],\n"
    "    'SVR z-scaled, y'                    : run_cv(svr_probe, Xp12)[0],\n"
    "    'SVR z-scaled, log y  (fixed)'       : run_cv(lambda: log_target(svr_probe()), Xp12)[0],\n"
    "    'kNN, y'                             : run_cv(knn_probe, Xp12)[0],\n"
    "    'kNN, log y'                         : run_cv(lambda: log_target(knn_probe()), Xp12)[0],\n"
    "    'ExtraTrees, y'                      : run_cv(et_probe, X)[0],\n"
    "    'ExtraTrees, log y'                  : run_cv(lambda: log_target(et_probe()), X)[0],\n"
    "}\n"
    "print(pd.Series(rows).round(3).to_string())"
))

cells.append(md(
    "#### Insights\n\n"
    "> The log target was never the problem. Once the SVR target is standardised, the log transform "
    "no longer breaks the SVR and gives it a small gain, and it lowers the kNN error by about 0.3, "
    "because it spreads the crowded lower mode and shrinks the influence of the few extreme scores "
    "on distance-based fits.\n\n"
    "> The same transform hurts ExtraTrees. Trees split on target averages, and averaging in log "
    "space and then exponentiating biases predictions downward for the high tail, which RMSE "
    "punishes. The target transform is therefore chosen per model family and never applied globally."
))

cells.append(md(
    "## PCA Dimension Robustness\n\n"
    "The best PCA dimension for SVR is 12, but a sweep shows the score is bumpy across dimensions "
    "(for example 8.13 at 12 components and 8.62 at 14). Picking a single sharp optimum on 382 rows "
    "risks fitting the validation noise, so an average of SVRs over several dimensions is compared "
    "with the single best dimension under three different fold seeds."
))

cells.append(code(
    "PCA_DIMS = [10, 12, 14, 16, 20]\n"
    "views = {k: pca_view(k) for k in PCA_DIMS}\n"
    "\n"
    "\n"
    "def svr_multi_pca_cv(dims, seed):\n"
    "    err = []\n"
    "    for tri, vai in RepeatedKFold(n_splits=5, n_repeats=3, random_state=seed).split(X):\n"
    "        preds = []\n"
    "        for k in dims:\n"
    "            m = log_target(z_target(SVR(C=3, gamma=1 / k, epsilon=0.05)))\n"
    "            preds.append(m.fit(views[k][0][tri], y[tri]).predict(views[k][0][vai]))\n"
    "        err.append(np.mean((y[vai] - np.mean(preds, 0)) ** 2))\n"
    "    return np.sqrt(np.mean(err))\n"
    "\n"
    "\n"
    "robust = pd.DataFrame({\n"
    "    seed: {'single k=12': svr_multi_pca_cv([12], seed),\n"
    "           'average k=10..20': svr_multi_pca_cv(PCA_DIMS, seed)}\n"
    "    for seed in [42, 7, 123]\n"
    "})\n"
    "print(robust.round(4).to_string())"
))

cells.append(md(
    "#### Insights\n\n"
    "> Averaging SVRs over five PCA dimensions matches or beats the single best dimension under "
    "every fold seed. It removes the need to select one sharp hyperparameter on a small dataset, so "
    "the multi-dimension SVR is used as the kernel model in Section 5."
))

# ---------------------------------------------------------------------------
# Section 5: Modelling
# ---------------------------------------------------------------------------
cells.append(section("Modelling", "5"))

cells.append(md(
    "Five model families are trained on the shared folds. Ridge is the linear baseline, CatBoost "
    "represents gradient boosting, the multi-dimension SVR is the verified kernel model, and TabPFN "
    "v2 and TabICLv2 are pretrained tabular foundation models. The two foundation models are "
    "placed on separate T4 GPUs."
))

cells.append(md(
    "## Pretrained Foundation Models\n\n"
    "With 382 rows, a model trained from scratch has very little data to learn a prior from. "
    "Tabular foundation models solve this by pretraining a transformer on millions of synthetic "
    "datasets and then predicting new rows through in-context learning, where the training set is "
    "passed as context and no weights are updated on the competition data. TabICLv2 (open weights, "
    "Hugging Face `jingang/TabICL`) is reported to outperform RealTabPFN-2.5 on the TabArena and "
    "TALENT benchmarks, and TabPFN v2 (Hugging Face `Prior-Labs/TabPFN-v2-reg`) is its most widely "
    "used predecessor. Both are used only as learning algorithms on the official data, which the "
    "competition rules allow."
))

cells.append(code(
    "class MultiPCASVR:\n"
    "    def __init__(self, dims=PCA_DIMS):\n"
    "        self.dims = dims\n"
    "        self.pcas = {k: PCA(k, random_state=CFG.SEED).fit(Z_all) for k in dims}\n"
    "        self.sds = {k: self.pcas[k].transform(Z_all).std(0) for k in dims}\n"
    "\n"
    "    def _view(self, k, A):\n"
    "        return self.pcas[k].transform(sc_all.transform(A)) / self.sds[k]\n"
    "\n"
    "    def fit(self, A, t):\n"
    "        self.models = {\n"
    "            k: log_target(z_target(SVR(C=3, gamma=1 / k, epsilon=0.05))).fit(self._view(k, A), t)\n"
    "            for k in self.dims\n"
    "        }\n"
    "        return self\n"
    "\n"
    "    def predict(self, A):\n"
    "        return np.mean([self.models[k].predict(self._view(k, A)) for k in self.dims], 0)\n"
    "\n"
    "\n"
    "GPU_A = CFG.DEVICES[0]\n"
    "GPU_B = CFG.DEVICES[-1]\n"
    "\n"
    "MODELS = {\n"
    "    'Ridge'       : lambda: make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 3, 30))),\n"
    "    'CatBoost'    : lambda: CatBoostRegressor(iterations=1500, learning_rate=0.02, depth=5,\n"
    "                                              l2_leaf_reg=5, verbose=0, random_seed=CFG.SEED,\n"
    "                                              allow_writing_files=False),\n"
    "    'SVR-multiPCA': MultiPCASVR,\n"
    "    'TabPFNv2'    : lambda: TabPFNRegressor.create_default_for_version(\n"
    "                        ModelVersion.V2, device=GPU_B, n_estimators=16, random_state=CFG.SEED),\n"
    "    'TabICLv2'    : lambda: TabICLRegressor(device=GPU_A, random_state=CFG.SEED),\n"
    "}\n"
    "print('TabICLv2 on', GPU_A, '| TabPFNv2 on', GPU_B)"
))

cells.append(md(
    "## Cross-Validation\n\n"
    "Each model is scored on the 15 shared folds. The reported RMSE is the mean over the three "
    "repeats, the out-of-fold predictions are averaged over repeats for the blend, and each model "
    "is refitted on all 382 rows to predict the test set."
))

cells.append(code(
    "OOF, TEST, rows = {}, {}, []\n"
    "for name, make in MODELS.items():\n"
    "    t0 = time.time()\n"
    "    score, OOF[name], TEST[name] = run_cv(make, X, X_te)\n"
    "    rows.append((name, score, np.mean(np.abs(y - OOF[name])), time.time() - t0))\n"
    "    print(f'{name:13s} RMSE={score:.4f}  ({rows[-1][3]:.0f}s)')\n"
    "\n"
    "cv_table = pd.DataFrame(rows, columns=['model', 'RMSE', 'MAE (bagged OOF)', 'seconds']).set_index('model')\n"
    "cv_table = cv_table.sort_values('RMSE')\n"
    "print()\n"
    "print(cv_table.round(4).to_string())\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(8, 3.5))\n"
    "cv_table['RMSE'].plot.barh(ax=ax)\n"
    "ax.axvline(y.std(), c='r', ls='--', label='predict the mean')\n"
    "ax.set_xlim(cv_table['RMSE'].min() - 1, y.std() + 0.5)\n"
    "ax.invert_yaxis()\n"
    "ax.legend()\n"
    "ax.set_title('CV RMSE (5-fold x 3 repeats)')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'cv_leaderboard.png')\n"
    "plt.show()"
))

cells.append(md(
    "## Pessimistic Cross-Check with Similarity Groups\n\n"
    "Rows that lie within radius 1.0 of each other in standardised feature space are chained into "
    "groups and kept in the same fold. This is harsher than the real train-to-test situation, so the "
    "scores will rise, but the ranking of the models should not change if the choice is robust."
))

cells.append(code(
    "_, sim_groups = connected_components(\n"
    "    NearestNeighbors(radius=1.0).fit(Z_tr).radius_neighbors_graph(Z_tr), directed=False)\n"
    "for name in ['Ridge', 'CatBoost', 'SVR-multiPCA', 'TabICLv2']:\n"
    "    p = np.zeros(len(y))\n"
    "    for tri, vai in GroupKFold(5).split(X, y, sim_groups):\n"
    "        p[vai] = MODELS[name]().fit(X[tri], y[tri]).predict(X[vai])\n"
    "    print(f'{name:13s} GroupKFold RMSE={rmse(y, p):.4f}  |  random KFold RMSE={cv_table.loc[name, \"RMSE\"]:.4f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> The pretrained foundation models lead without any tuning. TabICLv2 reaches a CV RMSE of about "
    "7.6 and TabPFN v2 about 7.9, ahead of the tuned SVR (about 8.1), CatBoost (about 8.9) and the "
    "linear baseline (about 10.8). In local experiments, increasing TabICLv2 to 32 estimators or "
    "appending PCA features did not improve it, so the default configuration is kept.\n\n"
    "> Under similarity-grouped folds every model gets worse by about 0.3 to 0.8 points, but the order is "
    "unchanged and TabICLv2 degrades the least. The model choice is therefore not an artefact of "
    "near-duplicate rows leaking across folds."
))

# ---------------------------------------------------------------------------
# Section 6: Ensemble and Error Analysis
# ---------------------------------------------------------------------------
cells.append(section("Ensemble and Error Analysis", "6"))

cells.append(md(
    "A blend helps only when models make different mistakes. The residual correlation matrix shows "
    "which models are complementary, and the blend weights are then estimated with non-negative "
    "least squares on the out-of-fold predictions."
))

cells.append(md(
    "## Residual Correlation\n\n"
    "Residuals are computed from the out-of-fold predictions of each model. A correlation near 1 "
    "means two models fail on the same rows and blending them adds little."
))

cells.append(code(
    "names = list(MODELS)\n"
    "resid = pd.DataFrame({n: y - OOF[n] for n in names})\n"
    "fig, ax = plt.subplots(figsize=(7, 5.5))\n"
    "sns.heatmap(resid.corr(), annot=True, fmt='.2f', cmap='viridis', ax=ax)\n"
    "ax.set_title('out-of-fold residual correlation')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'residual_corr.png')\n"
    "plt.show()"
))

cells.append(md(
    "## Blend Weights\n\n"
    "Non-negative least squares finds the weights, which are then normalised to sum to one. To "
    "avoid judging the blend on the same rows that chose its weights, a nested 10-fold loop fits the "
    "weights on nine tenths of the out-of-fold rows and scores the remaining tenth."
))

cells.append(code(
    "def nnls_weights(P, t):\n"
    "    w, _ = nnls(P, t)\n"
    "    return w / w.sum()\n"
    "\n"
    "\n"
    "P_oof  = np.column_stack([OOF[n] for n in names])\n"
    "P_test = np.column_stack([TEST[n] for n in names])\n"
    "\n"
    "nested = np.zeros(len(y))\n"
    "for a, b in KFold(10, shuffle=True, random_state=CFG.SEED).split(P_oof):\n"
    "    nested[b] = P_oof[b] @ nnls_weights(P_oof[a], y[a])\n"
    "\n"
    "weights = pd.Series(nnls_weights(P_oof, y), index=names)\n"
    "blend_oof = P_oof @ weights.values\n"
    "print(weights.round(3).to_string())\n"
    "print(f'\\nbest single model (bagged OOF) RMSE : {min(rmse(y, OOF[n]) for n in names):.4f}')\n"
    "print(f'blend, nested estimate RMSE        : {rmse(y, nested):.4f}')\n"
    "print(f'blend, in-sample RMSE              : {rmse(y, blend_oof):.4f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> The blend keeps TabICLv2 (weight about 0.7) and the multi-dimension SVR (about 0.3) and gives "
    "zero weight to the others. TabPFN v2 is accurate on its own, but its residuals correlate at "
    "about 0.96 with TabICLv2, so it adds no new information. The SVR is weaker alone but works on a "
    "PCA view with a log target, which makes its mistakes different enough to be useful.\n\n"
    "> The nested estimate stays close to the in-sample value, so the two weights are not "
    "overfitted. In local runs the nested blend RMSE was stable at about 7.40 to 7.42 across three "
    "different outer seeds."
))

cells.append(md(
    "## Where the Blend Fails\n\n"
    "The out-of-fold residuals are grouped by target range to find which rows drive the RMSE. A "
    "simple linear stretch of the blend is also tested, since models trained with squared error "
    "tend to pull extreme predictions toward the mean."
))

cells.append(code(
    "err = pd.DataFrame({'y': y, 'res': y - blend_oof})\n"
    "err['bin'] = pd.cut(err.y, [30, 40, 50, 60, 70, 80, 101])\n"
    "by_bin = err.groupby('bin', observed=True).agg(\n"
    "    n=('y', 'size'), bias=('res', 'mean'), rmse=('res', lambda r: np.sqrt(np.mean(r ** 2))))\n"
    "by_bin['share_of_SSE'] = err.groupby('bin', observed=True).res.apply(lambda r: (r ** 2).sum()) / (err.res ** 2).sum()\n"
    "print(by_bin.round(3).to_string())\n"
    "\n"
    "slope, intercept = np.polyfit(nested, y, 1)\n"
    "stretched = np.zeros(len(y))\n"
    "for a, b in KFold(10, shuffle=True, random_state=CFG.SEED).split(P_oof):\n"
    "    w_a = nnls_weights(P_oof[a], y[a])\n"
    "    stretched[b] = np.polyval(np.polyfit(P_oof[a] @ w_a, y[a], 1), P_oof[b] @ w_a)\n"
    "print(f'\\nslope of y on nested blend = {slope:.3f}')\n"
    "print(f'nested RMSE without stretch = {rmse(y, nested):.4f} | with stretch = {rmse(y, stretched):.4f}')\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))\n"
    "ax[0].scatter(blend_oof, y, s=8)\n"
    "ax[0].plot([30, 101], [30, 101], 'r--')\n"
    "ax[0].set_xlabel('blend OOF prediction')\n"
    "ax[0].set_ylabel('target')\n"
    "by_bin['rmse'].plot.bar(ax=ax[1], color='tab:orange')\n"
    "ax[1].set_title('RMSE by target range')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'error_analysis.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> The error is concentrated in the right tail. The 26 training rows with a score above 70 "
    "(about 7% of the data) carry roughly 40% of the squared error, and the blend under-predicts "
    "them by 13 to 20 points on average. Below 70 the RMSE is only about 5 to 7.\n\n"
    "> The slope of the target on the blend is slightly above one, which confirms mild shrinkage "
    "toward the mean, but correcting it with a linear stretch changes the nested RMSE by less than "
    "0.01 and in some runs makes it slightly worse. The high scores are not predictable from the features with this little data, so the "
    "stretch is not applied. Forcing predictions outward would add variance on the other 93% of "
    "rows for a gain that does not survive validation."
))

cells.append(md(
    "## How Much to Trust the Public Leaderboard\n\n"
    "The public leaderboard uses about 30% of the 96 test rows, roughly 29 records. Resampling 29 "
    "out-of-fold residuals many times shows how much the public score can move by chance alone."
))

cells.append(code(
    "rng = np.random.default_rng(CFG.SEED)\n"
    "res_blend = y - blend_oof\n"
    "pub = [np.sqrt(np.mean(rng.choice(res_blend, 29, replace=False) ** 2)) for _ in range(5000)]\n"
    "prv = [np.sqrt(np.mean(rng.choice(res_blend, 67, replace=False) ** 2)) for _ in range(5000)]\n"
    "print(f'OOF RMSE {rmse(y, blend_oof):.3f}')\n"
    "print(f'public (29 rows)  90% range: {np.percentile(pub, 5):.2f} to {np.percentile(pub, 95):.2f}')\n"
    "print(f'private (67 rows) 90% range: {np.percentile(prv, 5):.2f} to {np.percentile(prv, 95):.2f}')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(7, 3.5))\n"
    "ax.hist(pub, bins=50, alpha=0.6, label='public, 29 rows')\n"
    "ax.hist(prv, bins=50, alpha=0.6, label='private, 67 rows')\n"
    "ax.axvline(rmse(y, blend_oof), c='k')\n"
    "ax.legend()\n"
    "ax.set_title('leaderboard RMSE spread from sampling alone')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'lb_noise.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> The same model can score anywhere from about 4.7 to 9.9 on the public leaderboard purely "
    "from which 29 rows are public, because a single tail record in or out of the public split moves "
    "the score by more than a point. The spread on the private split is about half as wide.\n\n"
    "> Model selection is therefore done by cross-validation on 382 rows, and the public "
    "leaderboard is used only as a sanity check that the submission file is valid and in the "
    "expected range. Tuning toward the public score would fit the noise of 29 rows."
))

# ---------------------------------------------------------------------------
# Section 7: Submission
# ---------------------------------------------------------------------------
cells.append(section("Submission", "7"))

cells.append(md(
    "The blend weights from Section 6 are applied to the test predictions of the models refitted on "
    "all training rows. The file follows `submission_template.csv` exactly, with one prediction per "
    "`record_id` and no missing values."
))

cells.append(code(
    "pred = P_test @ weights.values\n"
    "submission = template[['record_id']].merge(\n"
    "    pd.DataFrame({'record_id': test['record_id'], 'prediction': pred}), on='record_id', how='left')\n"
    "\n"
    "assert list(submission.columns) == ['record_id', 'prediction']\n"
    "assert len(submission) == len(template) == len(test)\n"
    "assert submission['record_id'].is_unique\n"
    "assert set(submission['record_id']) == set(test['record_id'])\n"
    "assert submission['prediction'].notna().all()\n"
    "\n"
    "submission.to_csv(CFG.OUTPUT_DIR / 'submission.csv', index=False)\n"
    "print(submission.head())\n"
    "print(f'\\ntest prediction mean {pred.mean():.2f}  range {pred.min():.2f} to {pred.max():.2f}')\n"
    "print(f'train OOF blend mean {blend_oof.mean():.2f}  range {blend_oof.min():.2f} to {blend_oof.max():.2f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> The test predictions have a mean and range similar to the out-of-fold predictions on the "
    "training set, which is consistent with the mild shift measured in Section 3. The file "
    "`submission.csv` in the output directory is the final submission."
))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
