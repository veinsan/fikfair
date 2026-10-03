import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "v7.ipynb"
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
    "*Predicting an Anonymous Competition Score from 24 transformed environmental and climate features "
    "(version 7, GPU-only variance candidates with an adoption threshold).*"
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
    "4. [**Deep Analysis of the Data Generating Process**](#4)\n"
    "5. [**Preprocessing Verification**](#5)\n"
    "6. [**Modelling**](#6)\n"
    "7. [**Ensemble, Selection Protocol and Limits**](#7)\n"
    "8. [**Submission**](#8)\n"
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
    "\n\n"
    "*This is the second version of the solution. Version 1 (TabICLv2 blended with a kernel SVR) scored "
    "7.947 on the public leaderboard against a nested cross-validation estimate of 7.41, a gap that is "
    "well inside the sampling noise of 29 public rows. Version 2 re-examines the data generating process "
    "in depth, tests every new idea against the same folds, and changes the model only where the "
    "cross-validated evidence is consistent.*"
    "\n\n"
    "*Version 2 scored 7.902 on the public leaderboard (version 1: 7.947), moving in the direction its "
    "cross-validation predicted. Version 3 tests four more ideas (alternative feature bases, test-time "
    "jitter, a correctly back-transformed log target and transductive pseudo-labelling). Only "
    "pseudo-labelling survives validation, and the analysis links it to the sample-density limit "
    "found in version 2.*"
    "\n\n"
    "*Version 3 scored 7.888 on the public leaderboard, again in the direction cross-validation "
    "predicted. Version 4 keeps the version 3 model unchanged and adds the evidence that its gain is "
    "trustworthy: the test set has the same geometry as the random validation folds, the gain does not "
    "shrink as labelled data grows, and no further variant beats it by more than the seed noise.*"
    "\n\n"
    "*Version 5 adds the gated Prior Labs checkpoints (TabPFN 2.5, 2.6, 3 and 3.5). Only TabPFN-3.5 is "
    "competitive, and with the same pseudo-labelling it matches TabICLv2. Averaging the two "
    "pseudo-labelled foundation models, which come from different pretraining priors, reduces variance "
    "beyond what seed bagging of one model achieves.*"
    "\n\n"
    "*Version 5 tied version 3 on the public leaderboard (7.888). Version 6 searches for blend members "
    "whose errors differ from the foundation models. Networks trained from scratch and rough kernels "
    "fail, but a less regularised SVR, averaged over a plateau of settings, makes more independent "
    "errors and improves the blend.*"
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
    "## Approach: Transductive Pseudo-Labelling and a Seed-Bagged Blend\n\n"
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
    "    +-- StandardScaler -> PCA(10..20) -> SVR, C in {10, 30}, two length-scales (target: log(y - 30))\n"
    "    +-- raw features  -> CatBoost / ExtraTrees  (target: y)\n"
    "    +-- raw features  -> TabICLv2 (norm=none, 4 seeds) / TabPFN v2 / TabDPT\n"
    "    +-- raw features  -> TabICLv2 + pseudo-labelled query rows in the context (version 3)\n"
    "    +-- raw features  -> TabPFN-3.5 + pseudo-labelled query rows in the context (version 5)\n"
    "         |\n"
    "    NNLS blend on out-of-fold predictions -> prediction\n"
    "```\n\n"
    "*Version 2 adds three candidates to the pool (TabICLv2 without internal power normalisation, "
    "bagged over four random states, the real-data foundation model TabDPT, and a fine-tuned TabICLv2) "
    "and a selection protocol that decides between them with nested, multi-seed cross-validation, a "
    "paired bootstrap and a leaderboard simulation instead of the public score.*\n\n"
    "*Version 3 adds transductive pseudo-labelling. The unlabeled rows being predicted are first "
    "labelled by TabICLv2, and then each quarter of them is predicted again with the other three quarters "
    "added to the context, so no row ever sees its own pseudo label. This densifies the context around "
    "every query, which targets the sample-density limit identified by the variogram.*\n\n"
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
    "needed to install the foundation model packages and download their checkpoints. The TabPFN 2.5, 2.6, "
    "3 and 3.5 checkpoints are gated by Prior Labs: the licence must be accepted once at ux.priorlabs.ai "
    "and the API key stored as a Kaggle secret named `TABPFN_TOKEN` (Add-ons, Secrets). Recorded runtime: about 60 minutes on a laptop CPU (Intel Core Ultra 7 155H). On the "
    "following two T4 GPUs the foundation models run many times faster (version 1 took under a minute "
    "per foundation model on this hardware)."
))

cells.append(code("!nvidia-smi"))

cells.append(md("The following cell installs all libraries used in this notebook."))

cells.append(code(
    "%pip install -q tabicl tabpfn tabdpt transformers catboost lightgbm"
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
    "from scipy.spatial.distance import cdist, pdist\n"
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
    "from tabdpt import TabDPTRegressor\n"
    "from tabicl import FinetunedTabICLRegressor, TabICLRegressor\n"
    "from tabpfn import TabPFNRegressor\n"
    "from tabpfn.constants import ModelVersion\n"
    "\n"
    "\n"
    "try:\n"
    "    from kaggle_secrets import UserSecretsClient\n"
    "except ImportError:\n"
    "    UserSecretsClient = None\n"
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
    "    FIG_DIR    = OUTPUT_DIR / 'fikfair_v7'\n"
    "\n"
    "    N_SPLITS   = 5\n"
    "    N_REPEATS  = 3\n"
    "    PCA_DIM    = 12\n"
    "    LOG_SHIFT  = 30.0\n"
    "    ICL_SEEDS  = [0, 1, 2, 3]\n"
    "    ICL_NORM   = ['none']\n"
    "    PL_SPLIT   = 4\n"
    "    SVR_CELLS  = [(0.7, 10), (0.7, 30), (1.0, 10), (1.0, 30)]\n"
    "    ICL_SEEDS8 = [0, 1, 2, 3, 4, 5, 6, 7]\n"
    "    ADOPT_MIN_GAIN = 0.015\n"
    "    DEVICES    = [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']\n"
    "\n"
    "CFG = Settings()\n"
    "CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
    "if UserSecretsClient is not None and 'TABPFN_TOKEN' not in os.environ:\n"
    "    os.environ['TABPFN_TOKEN'] = UserSecretsClient().get_secret('TABPFN_TOKEN')\n"
    "print('data:', CFG.DATA_DIR, '| devices:', CFG.DEVICES,\n"
    "      '| TabPFN token set:', bool(os.environ.get('TABPFN_TOKEN')) or Path('~/.cache/tabpfn/auth_token').expanduser().exists())"
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
    "print('spearman(record_id, target) : %.3f (p=%.2f)' % stats.spearmanr(tr_id, y))\n"
    "\n"
    "y_by_id = y[np.argsort(tr_id.values)]\n"
    "lag1 = np.corrcoef(y_by_id[:-1], y_by_id[1:])[0, 1]\n"
    "perm_rng = np.random.default_rng(42)\n"
    "null_lag1 = [np.corrcoef(p[:-1], p[1:])[0, 1] for p in (perm_rng.permutation(y) for _ in range(2000))]\n"
    "print('lag-1 autocorrelation of target along id order: %.3f (permutation 95%% band %.3f to %.3f)' % (\n"
    "    lag1, np.percentile(null_lag1, 2.5), np.percentile(null_lag1, 97.5)))"
))

cells.append(md(
    "#### Insights\n\n"
    "> The data is clean. There are no missing values and no exact duplicates inside or across "
    "splits, and every feature has as many unique values as rows, so the rounding step did not "
    "create ties.\n\n"
    "> The `record_id` values form one shuffled sequence from D0001 to D0478 and have no rank "
    "correlation with the target, which matches the rule that the identifier carries no information. "
    "It is therefore excluded from modelling. The target also shows no serial correlation along the "
    "identifier order (the lag-1 autocorrelation lies inside the permutation band), so consecutive "
    "identifiers are not the same source entity and random folds do not leak through neighbouring ids."
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
    "Section 5 tests whether this helps, because a transform that looks better on a histogram can "
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
    "rows have a neighbour within radius 0.5, so near-twin leakage is limited. Section 6 still "
    "reports a similarity-grouped K-fold score as a pessimistic cross-check.\n\n"
    "> The nearest neighbour's target is already a decent predictor (1-NN RMSE about 10.1), and the "
    "agreement is tight in the lower mode but loose for high scores. Local, similarity-based models "
    "should therefore do well, while the high tail will remain the hardest region."
))

# ---------------------------------------------------------------------------
# Section 4: Deep Analysis of the Data Generating Process
# ---------------------------------------------------------------------------
cells.append(section("Deep Analysis of the Data Generating Process", "4"))

cells.append(md(
    "Section 3 described the data. This section asks why it looks the way it does. Each subsection "
    "states a hypothesis about how the organisers built the features or the target, derives what the "
    "data must show if the hypothesis is true, checks it, and then measures with cross-validation "
    "whether acting on it improves the prediction."
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
    "## Can the Transformation Pipeline Be Inverted?\n\n"
    "The features were produced by rank normalisation, a bounded transformation, a private linear "
    "projection, standardisation, perturbation and rounding. Three consequences are testable. If the "
    "standardisation used all 478 rows, the combined train and test features have exactly zero mean and "
    "unit variance. If fewer than 24 indicators were projected, the smallest eigenvalues sit on a flat "
    "noise floor. If the sources were rank-normalised and bounded, a full 24-component ICA should "
    "recover flat, sub-Gaussian sources with negative excess kurtosis."
))

cells.append(code(
    "print('train only : mean range [%.3f, %.3f]  sd range [%.3f, %.3f]' % (\n"
    "    X.mean(0).min(), X.mean(0).max(), X.std(0).min(), X.std(0).max()))\n"
    "print('train+test : mean range [%.4f, %.4f]  sd range [%.4f, %.4f]' % (\n"
    "    X_all.mean(0).min(), X_all.mean(0).max(), X_all.std(0).min(), X_all.std(0).max()))\n"
    "\n"
    "ev_full = np.linalg.eigvalsh(np.cov(X_all.T))[::-1]\n"
    "print('\\nsmallest 8 eigenvalues :', np.round(ev_full[-8:], 4))\n"
    "print('log10 step between them:', np.round(np.diff(np.log10(ev_full[-8:])), 2))\n"
    "\n"
    "fig, ax = plt.subplots(1, 3, figsize=(17, 3.8))\n"
    "ax[0].semilogy(range(1, 25), ev_full, 'o-')\n"
    "ax[0].set_title('full eigen spectrum')\n"
    "for a, fun in zip(ax[1:], ['logcosh', 'cube']):\n"
    "    S24 = FastICA(24, whiten='unit-variance', fun=fun, max_iter=20000, tol=1e-6,\n"
    "                  random_state=0).fit_transform(X_all)\n"
    "    ku = np.sort(stats.kurtosis(S24, axis=0))\n"
    "    print(f'ICA-24 ({fun}): {(ku < 0).sum()} of 24 sources sub-Gaussian, median kurtosis {np.median(ku):.2f}')\n"
    "    a.bar(range(24), ku)\n"
    "    a.axhline(-1.2, c='r', ls='--', label='bounded / uniform source')\n"
    "    a.set_title(f'ICA-24 ({fun}) sorted source kurtosis')\n"
    "    a.legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'pipeline_inversion.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> The features have mean 0 and standard deviation 1 on train and test combined, but not on train "
    "alone, so the organisers standardised all 478 records together. Fitting unsupervised transforms "
    "such as PCA on the combined features is therefore consistent with how the data was built.\n\n"
    "> The spectrum decays smoothly with no flat floor, so the projection is full rank and no feature "
    "direction is pure noise. The 24-component ICA finds almost only heavy-tailed sources (median "
    "kurtosis well above 3), not the flat sources that rank and bounded transforms would produce. The "
    "original indicators are correlated with each other, which breaks the independence assumption of "
    "ICA, so the projection cannot be undone and this route is closed."
))

cells.append(md(
    "## Typicality: Why Most Features Show an Inverted U\n\n"
    "Section 3 showed that the score peaks near zero for many features at once. A single mechanism "
    "explains this: the score falls as a record moves away from the centre of the data in any direction. "
    "If that is true, distance-to-centre measures should correlate with the target at least as strongly "
    "as the best single feature. Five distances are computed from the features alone, without the target."
))

cells.append(code(
    "ev_z, V_z = np.linalg.eigh(np.cov(Z_all.T))\n"
    "ev_z, V_z = ev_z[::-1], V_z[:, ::-1]\n"
    "W_pc = Z_all @ V_z / np.sqrt(ev_z)\n"
    "TYP = pd.DataFrame({\n"
    "    'mahalanobis'      : np.linalg.norm(W_pc, axis=1),\n"
    "    'maha_major_pc1_4' : np.linalg.norm(W_pc[:, :4], axis=1),\n"
    "    'maha_minor_pc5_24': np.linalg.norm(W_pc[:, 4:], axis=1),\n"
    "    'euclid'           : np.linalg.norm(Z_all, axis=1),\n"
    "    'max_abs_z'        : np.abs(Z_all).max(1),\n"
    "})\n"
    "print(pd.Series({c: stats.spearmanr(TYP[c][:len(X)], y)[0] for c in TYP}).round(3).to_string())\n"
    "print('best single raw feature |spearman|:', round(max(abs(stats.spearmanr(X[:, j], y)[0]) for j in range(24)), 3))\n"
    "\n"
    "fig, axes = plt.subplots(1, 5, figsize=(22, 3.8))\n"
    "for a, c in zip(axes, TYP):\n"
    "    a.scatter(TYP[c][:len(X)], y, s=6, alpha=0.5)\n"
    "    lw = lowess(y, TYP[c][:len(X)], frac=0.4)\n"
    "    a.plot(lw[:, 0], lw[:, 1], c='r')\n"
    "    a.set_title(c, fontsize=10)\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'typicality.png')\n"
    "plt.show()"
))

cells.append(md(
    "The distances are appended to the inputs of four model families to see whether the relationship "
    "is already captured or adds new information."
))

cells.append(code(
    "X_typ = np.hstack([X, TYP.values[:len(X)]])\n"
    "Xp12_typ = np.hstack([pca_view(12)[0], TYP.values[:len(X)]])\n"
    "typ_models = {\n"
    "    'Ridge'     : (lambda: make_pipeline(StandardScaler(), RidgeCV(np.logspace(-3, 3, 30))), X, X_typ),\n"
    "    'SVR(log)'  : (lambda: log_target(z_target(make_pipeline(StandardScaler(), SVR(C=3, gamma='scale', epsilon=0.05)))),\n"
    "                  pca_view(12)[0], Xp12_typ),\n"
    "    'CatBoost'  : (lambda: CatBoostRegressor(iterations=1500, learning_rate=0.02, depth=5, l2_leaf_reg=5,\n"
    "                                             verbose=0, random_seed=CFG.SEED, allow_writing_files=False), X, X_typ),\n"
    "    'TabICLv2'  : (lambda: TabICLRegressor(device=CFG.DEVICES[0], random_state=CFG.SEED), X, X_typ),\n"
    "}\n"
    "for name, (make, A, B) in typ_models.items():\n"
    "    r0, r1 = run_cv(make, A)[0], run_cv(make, B)[0]\n"
    "    print(f'{name:10s} base={r0:.4f}  +typicality={r1:.4f}  delta={r1 - r0:+.4f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> The hypothesis holds. The Mahalanobis distance has a Spearman correlation of about -0.39 with "
    "the score, close to the strongest raw feature, and the distance inside the minor principal "
    "subspace (PC5 to PC24) is almost as strong. The plots show the relationship is one-sided. Atypical "
    "records (Mahalanobis above about 6) almost always score around 40, while typical records can score "
    "anything, and every score above 80 lies at a moderate distance.\n\n"
    "> Acting on it helps only the weaker models. Ridge and CatBoost improve by about 0.3 and 0.2, the "
    "SVR gets worse because five correlated distances distort its RBF metric, and TabICLv2 does not "
    "change. The foundation model already learns typicality from the raw features, so the distances "
    "are not added to the final pipeline."
))

cells.append(md(
    "## Sufficient Dimension Reduction\n\n"
    "If the score depends on only a few linear combinations of the features, modelling in that small "
    "subspace should beat modelling in 24 dimensions. Sliced inverse regression (SIR) finds directions "
    "with monotone effects and sliced average variance estimation (SAVE) finds directions with symmetric "
    "effects such as the inverted U. Their eigenvalues are compared against a permutation null, and "
    "because both methods use the target, the directions are re-estimated inside every training fold."
))

cells.append(code(
    "def sdr_directions(A, t, method, H=10):\n"
    "    mu = A.mean(0)\n"
    "    ev_a, V_a = np.linalg.eigh(np.cov(A.T))\n"
    "    Wh = V_a @ np.diag(ev_a ** -0.5) @ V_a.T\n"
    "    Zh = (A - mu) @ Wh\n"
    "    slices = np.array_split(np.argsort(t), H)\n"
    "    p = A.shape[1]\n"
    "    if method == 'sir':\n"
    "        M = sum(len(s) / len(t) * np.outer(Zh[s].mean(0), Zh[s].mean(0)) for s in slices)\n"
    "    else:\n"
    "        M = sum(len(s) / len(t) * (np.eye(p) - np.cov(Zh[s].T)) @ (np.eye(p) - np.cov(Zh[s].T)) for s in slices)\n"
    "    ev_m, V_m = np.linalg.eigh(M)\n"
    "    return ev_m[::-1], Wh @ V_m[:, ::-1]\n"
    "\n"
    "\n"
    "rng = np.random.default_rng(CFG.SEED)\n"
    "for method in ['sir', 'save']:\n"
    "    obs = sdr_directions(X, y, method)[0][:6]\n"
    "    null = np.array([sdr_directions(X, rng.permutation(y), method)[0][:6] for _ in range(200)])\n"
    "    print(f'{method.upper():4s} eigenvalues:', np.round(obs, 3))\n"
    "    print(f'     null 95%   :', np.round(np.percentile(null, 95, 0), 3),\n"
    "          '-> significant:', int((obs > np.percentile(null, 95, 0)).sum()))\n"
    "\n"
    "\n"
    "def sdr_svr_cv(d_sir, d_save):\n"
    "    err = []\n"
    "    for tri, vai in FOLDS:\n"
    "        B = np.hstack([sdr_directions(X[tri], y[tri], 'sir')[1][:, :d_sir],\n"
    "                       sdr_directions(X[tri], y[tri], 'save')[1][:, :d_save]])\n"
    "        mu = X[tri].mean(0)\n"
    "        A_tr, A_va = (X[tri] - mu) @ B, (X[vai] - mu) @ B\n"
    "        sd = A_tr.std(0)\n"
    "        m = log_target(z_target(SVR(C=3, gamma=1 / B.shape[1], epsilon=0.05)))\n"
    "        err.append(np.mean((y[vai] - m.fit(A_tr / sd, y[tri]).predict(A_va / sd)) ** 2))\n"
    "    return np.sqrt(np.mean(err))\n"
    "\n"
    "\n"
    "print()\n"
    "for d in [(3, 0), (3, 3), (6, 6)]:\n"
    "    print(f'SVR on {d[0]} SIR + {d[1]} SAVE directions: RMSE={sdr_svr_cv(*d):.4f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> Both methods find about six directions above the permutation null, so the score depends on a "
    "subspace much smaller than 24 dimensions. In theory, modelling there should help.\n\n"
    "> In practice it fails badly. An SVR on the estimated SIR and SAVE directions scores between "
    "about 9.6 and 10.6, against about 8.1 for the same SVR on unsupervised PCA scores. With roughly 300 "
    "training rows per fold, the supervised directions are estimated too noisily, and the error in "
    "the directions costs more than the dimension reduction saves. The route is dropped."
))

# ---------------------------------------------------------------------------
# Section 5: Preprocessing Verification
# ---------------------------------------------------------------------------
cells.append(section("Preprocessing Verification", "5"))

cells.append(md(
    "The EDA suggests several preprocessing ideas that are sound in theory. Each one is applied, "
    "inspected and scored with the same repeated 5-fold split and three different model families, "
    "because a transform that is correct on paper can still damage the evaluation score."
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
    "the multi-dimension SVR is used as the kernel model in Section 6."
))

cells.append(md(
    "## Feature-Space Views\n\n"
    "The 24 features are a private linear projection, and TabICLv2 embeds each column separately, so "
    "it is not rotation invariant. Presenting the same data in another linear basis is the tabular "
    "counterpart of image augmentation: if the organisers' basis is arbitrary, other bases should score "
    "about the same and make partly different errors. The raw basis is compared with PCA scores and two "
    "random orthogonal rotations."
))

cells.append(code(
    "icl_single = lambda: TabICLRegressor(device=CFG.DEVICES[0], random_state=CFG.SEED, norm_methods=CFG.ICL_NORM)\n"
    "ev_v, V_v = np.linalg.eigh(np.cov(Z_all.T))\n"
    "VIEWS = {\n"
    "    'raw'  : X_all,\n"
    "    'pca24': Z_all @ V_v[:, ::-1],\n"
    "    'rot0' : Z_all @ stats.ortho_group.rvs(24, random_state=100),\n"
    "    'rot1' : Z_all @ stats.ortho_group.rvs(24, random_state=101),\n"
    "}\n"
    "view_oof = {}\n"
    "for vn, Vw in VIEWS.items():\n"
    "    score, view_oof[vn], _ = run_cv(icl_single, Vw[:len(X)])\n"
    "    rc = np.corrcoef(y - view_oof['raw'], y - view_oof[vn])[0, 1]\n"
    "    print(f'{vn:6s} RMSE={score:.4f}  residual corr vs raw={rc:.3f}')\n"
    "print(f'average of all views (bagged OOF) RMSE={rmse(y, np.mean(list(view_oof.values()), 0)):.4f}'\n"
    "      f'  | raw alone {rmse(y, view_oof[\"raw\"]):.4f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> The organisers' basis is not arbitrary for the model. In the local run PCA scored 7.72, random "
    "rotations 7.54 to 7.70 and ICA 7.96, against 7.60 for the raw features, and the residuals of "
    "rotated views correlate at about 0.99 with the raw view. Rotations behave like extra random "
    "seeds, not like new information, so the raw basis is kept and variance is reduced by seed "
    "bagging instead."
))

cells.append(md(
    "## Test-Time Jitter\n\n"
    "The features carry a controlled perturbation. Averaging the prediction over 16 jittered copies of "
    "each query row, with Gaussian noise of scale sigma in standardised units, should smooth that "
    "perturbation if sigma matches it. If sigma is too large, the jitter blurs real differences between "
    "records instead."
))

cells.append(code(
    "tta_rng = np.random.default_rng(CFG.SEED)\n"
    "\n"
    "\n"
    "class JitterTTA:\n"
    "    def __init__(self, sigma, k=16):\n"
    "        self.sigma, self.k = sigma, k\n"
    "\n"
    "    def fit(self, A, t):\n"
    "        self.model = icl_single().fit(A, t)\n"
    "        return self\n"
    "\n"
    "    def predict(self, B):\n"
    "        Bj = np.repeat(B, self.k, 0) + tta_rng.normal(0, self.sigma, (len(B) * self.k, B.shape[1])) * X_all.std(0)\n"
    "        return self.model.predict(Bj).reshape(len(B), self.k).mean(1)\n"
    "\n"
    "\n"
    "print(f'no jitter   RMSE={run_cv(icl_single, X)[0]:.4f}')\n"
    "for sigma in [0.03, 0.1, 0.3]:\n"
    "    print(f'sigma={sigma:<5} RMSE={run_cv(lambda s=sigma: JitterTTA(s), X)[0]:.4f}')\n"
    "print(f'jitter norm at sigma=0.1: {0.1 * np.sqrt(24):.2f}  vs median nearest-neighbour distance {np.median(d_trtr):.2f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> Jitter only hurts beyond a negligible scale. Locally, sigma 0.03 changed the RMSE by -0.025 "
    "(inside seed noise), sigma 0.1 added 0.18 and sigma 0.3 added 1.34. A jitter of sigma 0.1 already "
    "has a norm of about 0.5, a large fraction of the typical nearest-neighbour distance, so it blurs "
    "real structure. The organisers' perturbation is small and there is nothing to smooth, so no "
    "test-time jitter is used."
))

cells.append(md(
    "## Target Transform via Predictive Quantiles\n\n"
    "Version 1 found TabICLv2 worse on a log target (about 7.77 against 7.60). The suspected cause is "
    "the back-transform. Exponentiating the predicted mean of log y gives exp(E[log y]), which is "
    "smaller than E[y] by Jensen's inequality, so every prediction is biased low. TabICLv2 also "
    "returns predictive quantiles, and a monotone transform maps quantiles to quantiles, so inverting "
    "each quantile and then averaging gives a proper estimate of E[y]."
))

cells.append(code(
    "ALPHAS = np.linspace(0.005, 0.995, 199)\n"
    "\n"
    "\n"
    "class LogTargetICL:\n"
    "    def __init__(self, via_quantiles):\n"
    "        self.via_quantiles = via_quantiles\n"
    "\n"
    "    def fit(self, A, t):\n"
    "        self.model = icl_single().fit(A, np.log(t - CFG.LOG_SHIFT))\n"
    "        return self\n"
    "\n"
    "    def predict(self, B):\n"
    "        if self.via_quantiles:\n"
    "            q = np.asarray(self.model.predict(B, output_type='quantiles', alphas=ALPHAS))\n"
    "            return (np.exp(q) + CFG.LOG_SHIFT).mean(1)\n"
    "        return np.exp(self.model.predict(B)) + CFG.LOG_SHIFT\n"
    "\n"
    "\n"
    "print(f'raw y                       RMSE={run_cv(icl_single, X)[0]:.4f}')\n"
    "print(f'log y, exp(mean)  (v1 way)  RMSE={run_cv(lambda: LogTargetICL(False), X)[0]:.4f}')\n"
    "print(f'log y, via quantiles        RMSE={run_cv(lambda: LogTargetICL(True), X)[0]:.4f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> The diagnosis is confirmed. Back-transforming through quantiles recovers most of the loss "
    "(from about 7.77 to about 7.62), so the version 1 result was an artefact of Jensen's inequality, "
    "not of the log transform itself. Even done correctly, the log target does not beat the raw target "
    "for TabICLv2, whose own output distribution already handles the skew, so the raw target is kept."
))

# ---------------------------------------------------------------------------
# Section 6: Modelling
# ---------------------------------------------------------------------------
cells.append(section("Modelling", "6"))

cells.append(md(
    "The model pool keeps every family from version 1 and adds the version 2 candidates. Ridge is the "
    "linear baseline, CatBoost represents gradient boosting, the multi-dimension SVR is the verified "
    "kernel model, and four pretrained foundation models cover different priors. TabICLv2 runs on the "
    "first T4 and TabPFN v2 and TabDPT on the second."
))

cells.append(md(
    "## Pretrained Foundation Models\n\n"
    "With 382 rows, a model trained from scratch has very little data to learn a prior from. Tabular "
    "foundation models are transformers pretrained on millions of datasets that predict new rows by "
    "in-context learning, with the training set passed as context. TabICLv2 (Hugging Face "
    "`jingang/TabICL`) is reported to outperform RealTabPFN-2.5 on TabArena and TALENT, TabPFN v2 "
    "(`Prior-Labs/TabPFN-v2-reg`) is its most widely used predecessor, and TabDPT (`Layer6/TabDPT`) is "
    "pretrained on real rather than synthetic datasets, which makes it the most likely to make "
    "different mistakes. LimiX-2 was considered and skipped because it requires flash-attention, which "
    "the T4 architecture does not support. All models are used only as learning algorithms on the "
    "official data, which the competition rules allow."
))

cells.append(md(
    "Two version 2 changes to TabICLv2 come from a local sweep of its internal options (norm methods, "
    "outlier clipping threshold, feature shuffling and ensemble size). All options scored within about "
    "0.05 of each other, but disabling the internal Yeo-Johnson power normalisation was better both with "
    "one random state (7.601 against 7.634) and when each option was averaged over four random states "
    "(7.612 against 7.647). The sweep also showed that the version 1 random state was slightly lucky, so "
    "version 2 averages four random states to make the test prediction less dependent on one draw."
))

cells.append(code(
    "class MultiPCASVR:\n"
    "    def __init__(self, dims=PCA_DIMS, cells=((1.0, 3),), clip=False):\n"
    "        self.dims, self.cells, self.clip = dims, cells, clip\n"
    "        self.pcas = {k: PCA(k, random_state=CFG.SEED).fit(Z_all) for k in dims}\n"
    "        self.sds = {k: self.pcas[k].transform(Z_all).std(0) for k in dims}\n"
    "\n"
    "    def _view(self, k, A):\n"
    "        return self.pcas[k].transform(sc_all.transform(A)) / self.sds[k]\n"
    "\n"
    "    def fit(self, A, t):\n"
    "        self.lo, self.hi = (t.min(), t.max()) if self.clip else (-np.inf, np.inf)\n"
    "        self.models = {\n"
    "            (k, scale, C): log_target(z_target(SVR(C=C, gamma=scale / k, epsilon=0.05))).fit(self._view(k, A), t)\n"
    "            for k in self.dims for scale, C in self.cells\n"
    "        }\n"
    "        return self\n"
    "\n"
    "    def predict(self, A):\n"
    "        return np.mean([np.clip(m.predict(self._view(k, A)), self.lo, self.hi)\n"
    "                        for (k, _, _), m in self.models.items()], 0)\n"
    "\n"
    "\n"
    "class BaggedTabICL:\n"
    "    def __init__(self, seeds=CFG.ICL_SEEDS, device='cpu', **kw):\n"
    "        self.seeds, self.device, self.kw = seeds, device, kw\n"
    "\n"
    "    def fit(self, A, t):\n"
    "        self.models = [TabICLRegressor(device=self.device, random_state=s, **self.kw).fit(A, t)\n"
    "                       for s in self.seeds]\n"
    "        return self\n"
    "\n"
    "    def predict(self, A):\n"
    "        return np.mean([m.predict(A) for m in self.models], 0)\n"
    "\n"
    "\n"
    "class PseudoLabelTabICL:\n"
    "    def __init__(self, n_split=CFG.PL_SPLIT, seeds=CFG.ICL_SEEDS, device='cpu'):\n"
    "        self.n_split, self.seeds, self.device = n_split, seeds, device\n"
    "\n"
    "    def _bag(self, A, t, B):\n"
    "        return BaggedTabICL(self.seeds, self.device, norm_methods=CFG.ICL_NORM).fit(A, t).predict(B)\n"
    "\n"
    "    def fit(self, A, t):\n"
    "        self.A, self.t = A, t\n"
    "        return self\n"
    "\n"
    "    def predict(self, B):\n"
    "        first = self._bag(self.A, self.t, B)\n"
    "        part = np.arange(len(B)) % self.n_split\n"
    "        out = np.empty(len(B))\n"
    "        for k in range(self.n_split):\n"
    "            q, c = part == k, part != k\n"
    "            out[q] = self._bag(np.vstack([self.A, B[c]]), np.r_[self.t, first[c]], B[q])\n"
    "        return out\n"
    "\n"
    "\n"
    "class PseudoLabelTabPFN35(PseudoLabelTabICL):\n"
    "    def __init__(self, version=ModelVersion.V3_5, n_estimators=8, **kw):\n"
    "        super().__init__(**kw)\n"
    "        self.version, self.n_estimators = version, n_estimators\n"
    "\n"
    "    def _bag(self, A, t, B):\n"
    "        return TabPFNRegressor.create_default_for_version(\n"
    "            self.version, device=self.device, n_estimators=self.n_estimators,\n"
    "            random_state=CFG.SEED).fit(A, t).predict(B)\n"
    "\n"
    "\n"
    "class TabDPTWrapper:\n"
    "    def __init__(self, device='cpu'):\n"
    "        self.device = device\n"
    "\n"
    "    def fit(self, A, t):\n"
    "        self.model = TabDPTRegressor(device=self.device, compile=False, use_flash=False, verbose=False)\n"
    "        self.model.fit(A, t)\n"
    "        return self\n"
    "\n"
    "    def predict(self, A):\n"
    "        return self.model.predict(A, n_ensembles=8, seed=CFG.SEED)\n"
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
    "    'SVR-plateau' : lambda: MultiPCASVR(cells=CFG.SVR_CELLS, clip=True),\n"
    "    'TabPFNv2'    : lambda: TabPFNRegressor.create_default_for_version(\n"
    "                        ModelVersion.V2, device=GPU_B, n_estimators=16, random_state=CFG.SEED),\n"
    "    'TabDPT'      : lambda: TabDPTWrapper(device=GPU_B),\n"
    "    'TabICLv2-v1' : lambda: TabICLRegressor(device=GPU_A, random_state=CFG.SEED),\n"
    "    'TabICLv2-v2' : lambda: BaggedTabICL(device=GPU_A, norm_methods=CFG.ICL_NORM),\n"
    "    'TabICLv2-PL' : lambda: PseudoLabelTabICL(device=GPU_A),\n"
    "    'TabPFN35-PL' : lambda: PseudoLabelTabPFN35(device=GPU_B),\n"
    "    'TabICLv2-PL-8': lambda: PseudoLabelTabICL(seeds=CFG.ICL_SEEDS8, device=GPU_A),\n"
    "    'TabPFN35-PL-16': lambda: PseudoLabelTabPFN35(n_estimators=16, device=GPU_B),\n"
    "    'TabPFN3-PL'  : lambda: PseudoLabelTabPFN35(version=ModelVersion.V3, device=GPU_B),\n"
    "}\n"
    "print('TabICLv2 on', GPU_A, '| TabPFNv2 and TabDPT on', GPU_B)"
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
    "FM_MEANS = {\n"
    "    'FM-PL-mean'     : ['TabICLv2-PL', 'TabPFN35-PL'],\n"
    "    'FM-PL-mean-big' : ['TabICLv2-PL-8', 'TabPFN35-PL-16'],\n"
    "    'FM-PL-mean-3'   : ['TabICLv2-PL', 'TabPFN35-PL', 'TabPFN3-PL'],\n"
    "    'FM-PL-mean-big3': ['TabICLv2-PL-8', 'TabPFN35-PL-16', 'TabPFN3-PL'],\n"
    "}\n"
    "for fm_name, members in FM_MEANS.items():\n"
    "    OOF[fm_name] = np.mean([OOF[m] for m in members], 0)\n"
    "    TEST[fm_name] = np.mean([TEST[m] for m in members], 0)\n"
    "print('residual corr TabICLv2-PL vs TabPFN35-PL:',\n"
    "      round(np.corrcoef(y - OOF['TabICLv2-PL'], y - OOF['TabPFN35-PL'])[0, 1], 3))\n"
    "\n"
    "cv_table = pd.DataFrame(rows, columns=['model', 'RMSE', 'MAE (bagged OOF)', 'seconds']).set_index('model')\n"
    "cv_table = cv_table.sort_values('RMSE')\n"
    "print()\n"
    "print(cv_table.round(4).to_string())\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
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
    "## Fine-Tuning TabICLv2\n\n"
    "Fine-tuning adapts the pretrained weights to this dataset with a pinball loss and early stopping "
    "on an internal 10% split of the training fold, so the validation fold stays unseen. In theory it "
    "can lower bias; the risk is that the early-stopping split holds only about 30 rows. Fine-tuning at "
    "two learning rates is compared with the same zero-shot model on the first five folds."
))

cells.append(code(
    "ft_configs = {\n"
    "    'zero-shot'  : lambda: TabICLRegressor(device=GPU_A, random_state=CFG.SEED, norm_methods=CFG.ICL_NORM),\n"
    "    'ft lr=1e-5' : lambda: FinetunedTabICLRegressor(epochs=30, learning_rate=1e-5, device=GPU_A,\n"
    "                                                    random_state=CFG.SEED, norm_methods=CFG.ICL_NORM),\n"
    "    'ft lr=3e-6' : lambda: FinetunedTabICLRegressor(epochs=30, learning_rate=3e-6, device=GPU_A,\n"
    "                                                    random_state=CFG.SEED, norm_methods=CFG.ICL_NORM),\n"
    "}\n"
    "ft_fold = {}\n"
    "for name, make in ft_configs.items():\n"
    "    ft_fold[name] = [rmse(y[vai], make().fit(X[tri], y[tri]).predict(X[vai])) for tri, vai in FOLDS[:5]]\n"
    "ft_fold = pd.DataFrame(ft_fold)\n"
    "print(ft_fold.round(3).to_string())\n"
    "print('\\nmean RMSE:', ft_fold.mean().round(4).to_dict())\n"
    "for name in ['ft lr=1e-5', 'ft lr=3e-6']:\n"
    "    d = ft_fold[name] - ft_fold['zero-shot']\n"
    "    print(f'{name}: mean difference {d.mean():+.4f}, better on {(d < 0).sum()} of 5 folds')"
))

cells.append(md(
    "#### Insights\n\n"
    "> Fine-tuning makes TabICLv2 worse. In the full local run on all 15 folds, a learning rate of "
    "1e-5 raised the RMSE from 7.60 to 7.69 and was better on only 1 of 15 folds, a learning rate of "
    "3e-6 raised it to 7.63, and freezing the column and row encoders still gave 7.65. The smaller the "
    "update, the closer the result is to zero-shot, which means the pretrained prior is better than "
    "anything the weights can learn from 382 rows. The final model uses zero-shot TabICLv2."
))

cells.append(md(
    "## Transductive Pseudo-Labelling\n\n"
    "`TabICLv2-PL` receives the whole query set at prediction time: the validation fold during "
    "cross-validation and the 96 test rows at the end. It first predicts all query rows with the bagged "
    "TabICLv2, splits them into four parts, and predicts each part again with the other three parts and "
    "their pseudo labels added to the context. Three safeguards keep it honest. True labels of query "
    "rows are never used, a query row never sees its own pseudo label (otherwise the model would simply "
    "copy it), and the cross-validation reproduces the test situation closely, with a ratio of pseudo "
    "to labelled context rows of about 0.19 in both cases (58 of 306 per fold, 72 of 382 at test time). "
    "Using unlabeled test features this way is standard semi-supervised learning and does not use any "
    "test target or external data."
))

cells.append(md(
    "## TabPFN Checkpoint Zoo\n\n"
    "Prior Labs publishes several gated regression checkpoints: TabPFN-2.5 (default, small-samples, real "
    "and low-skew variants), 2.6, 3 and 3.5 (September 2026). Each is scored as a plain in-context "
    "learner on the shared folds, and its residual correlation with the pseudo-labelled TabICLv2 shows "
    "whether it could add anything to the blend."
))

cells.append(code(
    "ZOO = {\n"
    "    'TabPFN-2.5'        : 'tabpfn-v2.5-regressor-v2.5_default.ckpt',\n"
    "    'TabPFN-2.5-small'  : 'tabpfn-v2.5-regressor-v2.5_small-samples.ckpt',\n"
    "    'RealTabPFN-2.5'    : 'tabpfn-v2.5-regressor-v2.5_real.ckpt',\n"
    "    'TabPFN-2.5-lowskew': 'tabpfn-v2.5-regressor-v2.5_low-skew.ckpt',\n"
    "    'TabPFN-2.6'        : 'tabpfn-v2.6-regressor-v2.6_default.ckpt',\n"
    "    'TabPFN-3'          : 'tabpfn-v3-regressor-v3_default.ckpt',\n"
    "    'TabPFN-3.5'        : None,\n"
    "}\n"
    "zoo_rows = []\n"
    "for name, ckpt in ZOO.items():\n"
    "    if ckpt is None:\n"
    "        make = lambda: TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, device=GPU_B,\n"
    "                                                                  n_estimators=8, random_state=CFG.SEED)\n"
    "    else:\n"
    "        make = lambda c=ckpt: TabPFNRegressor(model_path=c, device=GPU_B, n_estimators=8, random_state=CFG.SEED)\n"
    "    score, oof_z, _ = run_cv(make, X)\n"
    "    rc = np.corrcoef(y - oof_z, y - OOF['TabICLv2-PL'])[0, 1]\n"
    "    zoo_rows.append((name, score, rc))\n"
    "zoo = pd.DataFrame(zoo_rows, columns=['checkpoint', 'RMSE', 'residual corr vs TabICLv2-PL']).set_index('checkpoint')\n"
    "print(zoo.sort_values('RMSE').round(4).to_string())"
))

cells.append(md(
    "#### Insights\n\n"
    "> Without pseudo-labelling, no TabPFN checkpoint beats TabICLv2. Locally, the 2.5 family scored "
    "about 7.79 to 7.88, version 2.6 about 7.84 and version 3 about 7.70, all with residual "
    "correlations of 0.94 to 0.98 against the pseudo-labelled TabICLv2, and none of them improved the "
    "version 3 blend. TabPFN-3.5 is the exception worth pursuing: at about 7.66 it is almost as "
    "accurate as TabICLv2, so it is also run with the same pseudo-labelling scheme as `TabPFN35-PL`."
))

cells.append(md(
    "## Kernel Choice from the Variogram\n\n"
    "The foundation models all fail on the same rows, so the blend can only improve through its one "
    "member with different errors, the SVR. The target semivariogram rises roughly linearly from the "
    "origin, which in geostatistics indicates a rough process with an exponential covariance (Matern "
    "with smoothness 1/2), whereas the RBF kernel assumes an infinitely smooth function. If that reading "
    "is right, a Laplacian or Matern-1/2 kernel should beat the RBF kernel in the same SVR recipe."
))

cells.append(code(
    "def kernel_fn(name):\n"
    "    if name == 'rbf':\n"
    "        return lambda A, B: np.exp(-cdist(A, B, 'sqeuclidean') / A.shape[1])\n"
    "    if name == 'laplacian':\n"
    "        return lambda A, B: np.exp(-cdist(A, B, 'cityblock') / A.shape[1])\n"
    "    if name == 'matern12':\n"
    "        return lambda A, B: np.exp(-cdist(A, B) / np.sqrt(A.shape[1]))\n"
    "    return lambda A, B: (1 + np.sqrt(3) * cdist(A, B) / np.sqrt(A.shape[1])) * np.exp(\n"
    "        -np.sqrt(3) * cdist(A, B) / np.sqrt(A.shape[1]))\n"
    "\n"
    "\n"
    "class KernelSVR:\n"
    "    def __init__(self, kernel, C=10):\n"
    "        self.K, self.C = kernel_fn(kernel), C\n"
    "\n"
    "    def fit(self, A, t):\n"
    "        self.A = A\n"
    "        self.models = {k: log_target(z_target(SVR(kernel='precomputed', C=self.C, epsilon=0.05))).fit(\n"
    "            self.K(views[k][0][A], views[k][0][A]), t) for k in PCA_DIMS}\n"
    "        return self\n"
    "\n"
    "    def predict(self, B):\n"
    "        return np.mean([m.predict(self.K(views[k][0][B], views[k][0][self.A])) for k, m in self.models.items()], 0)\n"
    "\n"
    "\n"
    "row_index = np.arange(len(y))\n"
    "for kname in ['rbf', 'matern32', 'matern12', 'laplacian']:\n"
    "    print(f'{kname:10s} CV RMSE={run_cv(lambda k=kname: KernelSVR(k), row_index)[0]:.4f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> The rough kernels lose. Locally, the best Laplacian and Matern-1/2 settings scored about 8.51 "
    "and 8.35 against about 8.09 for RBF and 8.04 for Matern-3/2. The variogram was measured in the raw "
    "24-dimensional space, where its apparent roughness is more likely target noise than a rough "
    "function, so the smooth RBF kernel is kept.\n\n"
    "> Other model families were also tried locally as partners with different errors and rejected. "
    "Networks trained from scratch (RealMLP 8.82, TabM 9.94) and the learned-metric kernel machine xRFM "
    "(8.58) were both too inaccurate on about 300 rows and gave no blend gain, and weaker "
    "pseudo-labelled foundation models (TabDPT, TabPFN v2) made the foundation average worse."
))

cells.append(md(
    "## SVR Regularisation and Blend Value\n\n"
    "A blend member is valuable for how different its errors are, not only for its own accuracy. The "
    "SVR penalty C controls how closely it follows the training targets, so a less regularised SVR may "
    "be slightly less accurate alone yet more independent of the foundation models. The SVR is refitted "
    "on a small grid of C and length-scale, and each setting is judged by the nested RMSE of its blend "
    "with the averaged pseudo-labelled foundation models."
))

cells.append(code(
    "def nested_pair(f, s_):\n"
    "    scores = []\n"
    "    for seed in [42, 7, 123, 2024, 99]:\n"
    "        pr = np.zeros(len(y))\n"
    "        for a, b in KFold(10, shuffle=True, random_state=seed).split(f):\n"
    "            w, _ = nnls(np.c_[f[a], s_[a]], y[a])\n"
    "            pr[b] = np.c_[f[b], s_[b]] @ (w / w.sum())\n"
    "        scores.append(rmse(y, pr))\n"
    "    return np.mean(scores)\n"
    "\n"
    "\n"
    "svr_grid = []\n"
    "for scale in [0.7, 1.0, 1.4]:\n"
    "    for C in [3, 10, 30, 100]:\n"
    "        score, oof_s, _ = run_cv(lambda sc=scale, c=C: MultiPCASVR(cells=[(sc, c)]), X)\n"
    "        svr_grid.append((scale, C, score, np.corrcoef(y - oof_s, y - OOF['FM-PL-mean'])[0, 1],\n"
    "                         nested_pair(OOF['FM-PL-mean'], oof_s)))\n"
    "svr_grid = pd.DataFrame(svr_grid, columns=['length-scale', 'C', 'SVR RMSE', 'residual corr vs FM', 'blend RMSE'])\n"
    "print(svr_grid.round(4).to_string(index=False))\n"
    "print(f\"\\nplateau average (SVR-plateau) blend RMSE: {nested_pair(OOF['FM-PL-mean'], OOF['SVR-plateau']):.4f}\"\n"
    "      f\"  | version 5 SVR (C=3): {nested_pair(OOF['FM-PL-mean'], OOF['SVR-multiPCA']):.4f}\")"
))

cells.append(md(
    "#### Insights\n\n"
    "> Blend value and single-model accuracy point in different directions. Raising C from 3 to 30 "
    "leaves the SVR's own RMSE about the same or slightly worse, but lowers its residual correlation "
    "with the foundation models and improves the blend. Locally the blend went from 7.296 at C=3 to "
    "between 7.252 and 7.262 for C of 10 or 30 and a length-scale multiplier of 0.7 or 1.0.\n\n"
    "> Those four settings form a plateau rather than a single sharp optimum, so they are averaged "
    "instead of picking one, following the same rule used for the PCA dimensions. Locally the averaged "
    "SVR improved the version 5 blend in all three repeats (7.251 against 7.296 on the averaged "
    "out-of-fold predictions and 7.411 against 7.443 averaged over single repeats), with a paired "
    "bootstrap probability of about 0.84.\n\n"
    "> One safeguard is added. A weakly regularised SVR on a log target can extrapolate far outside "
    "the possible range once its output is exponentiated. In the first version 6 run, single members "
    "predicted up to 138 for one test record, although the target never exceeds about 100.4, and "
    "cross-validation cannot validate such values because they almost never occur out-of-fold. Each "
    "member of `SVR-plateau` is therefore clipped to the range of its training targets before "
    "averaging. Locally this left the cross-validated blend unchanged (7.255 against 7.251) and "
    "changed two test predictions."
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
    "for name in ['Ridge', 'CatBoost', 'SVR-multiPCA', 'TabICLv2-v1', 'TabICLv2-v2', 'TabICLv2-PL', 'TabPFN35-PL']:\n"
    "    p = np.zeros(len(y))\n"
    "    for tri, vai in GroupKFold(5).split(X, y, sim_groups):\n"
    "        p[vai] = MODELS[name]().fit(X[tri], y[tri]).predict(X[vai])\n"
    "    print(f'{name:13s} GroupKFold RMSE={rmse(y, p):.4f}  |  random KFold RMSE={cv_table.loc[name, \"RMSE\"]:.4f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> Pseudo-labelling gives the largest single-model gain of the whole study. Locally, the "
    "pseudo-labelled TabICLv2 reached about 7.52 against 7.61 for the same four-seed TabICLv2 without "
    "it, and it was better on 12 of the 15 folds. A single model with a different random state (7.61) "
    "rules out a seed effect. Using half of the query rows as context gave 7.55 and three quarters "
    "gave 7.52, so more pseudo-labelled neighbours help.\n\n"
    "> Among the other models, TabICLv2 remains the strongest at about 7.6, followed by TabDPT (about 7.8), TabPFN "
    "v2 (about 7.9), the SVR (about 8.1), CatBoost (about 8.9) and Ridge (about 10.8). The version 2 "
    "TabICLv2 is about 0.02 better than the version 1 configuration, a small but consistent gain.\n\n"
    "> Under similarity-grouped folds every model gets worse, but the order is unchanged and the "
    "version 2 TabICLv2 keeps its lead over version 1 (about 7.90 against 7.95). The choice is not an "
    "artefact of near-duplicate rows leaking across folds."
))

# ---------------------------------------------------------------------------
# Section 7: Ensemble, Selection Protocol and Limits
# ---------------------------------------------------------------------------
cells.append(section("Ensemble, Selection Protocol and Limits", "7"))

cells.append(md(
    "A blend helps only when models make different mistakes. This section measures that, compares "
    "candidate blends with a protocol that does not rely on the public leaderboard, and then asks how "
    "close the chosen blend is to the limit of what the features can explain."
))

cells.append(md(
    "## Residual Correlation\n\n"
    "Residuals are computed from the out-of-fold predictions of each model. A correlation near 1 "
    "means two models fail on the same rows and blending them adds little."
))

cells.append(code(
    "names = list(MODELS)\n"
    "resid = pd.DataFrame({n: y - OOF[n] for n in names})\n"
    "fig, ax = plt.subplots(figsize=(8, 6.5))\n"
    "sns.heatmap(resid.corr(), annot=True, fmt='.2f', cmap='viridis', ax=ax)\n"
    "ax.set_title('out-of-fold residual correlation')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'residual_corr.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> The three foundation models are pretrained on different data (two synthetic priors and one "
    "real-data corpus), yet their residuals correlate at about 0.96 to 0.97. They fail on the same "
    "rows, which points to a limit in the data rather than in any one model. The SVR is the only strong "
    "model with clearly different errors (about 0.84 against TabICLv2), because it sees a PCA view with "
    "a log target."
))

cells.append(md(
    "## Selection Protocol\n\n"
    "Each candidate blend is fitted with non-negative least squares and judged on four criteria. The "
    "nested RMSE refits the weights inside a 10-fold split of the out-of-fold rows under five outer "
    "seeds. A paired bootstrap gives the probability that the candidate has a lower squared error than "
    "the version 6 blend (public 7.837). A leaderboard simulation draws random "
    "29-row and 67-row subsets and counts how often the candidate beats version 6, which shows how much the public score can confirm. Finally, an "
    "adversarially weighted RMSE up-weights training rows that look like test rows. A new candidate "
    "replaces version 6 only if its nested RMSE is lower by more than 0.015, about one standard "
    "deviation of the nested estimate over outer seeds, so that the submission does not chase noise.\n\n"
    "The two pseudo-labelled foundation models are almost collinear (residual correlation about 0.98). "
    "Free NNLS weights between them swing from fold to fold (locally between about 0.2 and 0.75), and "
    "that instability costs more than it gains. The standard remedy for nearly collinear predictors is "
    "to average them with equal weights and fit only the weight of the genuinely different SVR, which "
    "is the `FM-PL-mean` candidate."
))

cells.append(code(
    "def nnls_weights(P, t):\n"
    "    w, _ = nnls(P, t)\n"
    "    return w / w.sum()\n"
    "\n"
    "\n"
    "def nested_blend(cols, seed):\n"
    "    P = np.column_stack([OOF[c] for c in cols])\n"
    "    out = np.zeros(len(y))\n"
    "    for a, b in KFold(10, shuffle=True, random_state=seed).split(P):\n"
    "        out[b] = P[b] @ nnls_weights(P[a], y[a])\n"
    "    return out\n"
    "\n"
    "\n"
    "p_test_like = cross_val_predict(LogisticRegression(C=0.3, max_iter=3000), X_all, is_test,\n"
    "                                cv=skf, method='predict_proba')[:len(X), 1]\n"
    "adv_w = p_test_like / (1 - p_test_like)\n"
    "adv_w = adv_w / adv_w.mean()\n"
    "\n"
    "CANDIDATES = {\n"
    "    'v1: TabICL + SVR'           : ['TabICLv2-v1', 'SVR-multiPCA'],\n"
    "    'v2: TabICL-bag + SVR'       : ['TabICLv2-v2', 'SVR-multiPCA'],\n"
    "    'v3: TabICL-PL + SVR'        : ['TabICLv2-PL', 'SVR-multiPCA'],\n"
    "    'v3 + TabICL-bag'            : ['TabICLv2-PL', 'TabICLv2-v2', 'SVR-multiPCA'],\n"
    "    'v5: ICL-PL + PFN35-PL + SVR': ['TabICLv2-PL', 'TabPFN35-PL', 'SVR-multiPCA'],\n"
    "    'v5: mean(ICL-PL, PFN35-PL) + SVR': ['FM-PL-mean', 'SVR-multiPCA'],\n"
    "    'v6: mean(ICL-PL, PFN35-PL) + SVR-plateau': ['FM-PL-mean', 'SVR-plateau'],\n"
    "    'v7a: bigger ensembles + SVR-plateau'      : ['FM-PL-mean-big', 'SVR-plateau'],\n"
    "    'v7b: + TabPFN3-PL + SVR-plateau'          : ['FM-PL-mean-3', 'SVR-plateau'],\n"
    "    'v7c: bigger + TabPFN3-PL + SVR-plateau'   : ['FM-PL-mean-big3', 'SVR-plateau'],\n"
    "    'v2 + TabDPT'                : ['TabICLv2-v2', 'SVR-multiPCA', 'TabDPT'],\n"
    "    'v2 + TabDPT + TabPFN'       : ['TabICLv2-v2', 'SVR-multiPCA', 'TabDPT', 'TabPFNv2'],\n"
    "    'all models'                 : names,\n"
    "}\n"
    "BASELINE = 'v6: mean(ICL-PL, PFN35-PL) + SVR-plateau'\n"
    "base_nested = nested_blend(CANDIDATES[BASELINE], CFG.SEED)\n"
    "rng = np.random.default_rng(CFG.SEED)\n"
    "sel_rows = []\n"
    "for cname, cols in CANDIDATES.items():\n"
    "    runs = [rmse(y, nested_blend(cols, s)) for s in [42, 7, 123, 2024, 99]]\n"
    "    cand = nested_blend(cols, CFG.SEED)\n"
    "    se_c, se_b = (y - cand) ** 2, (y - base_nested) ** 2\n"
    "    boot = [np.mean((se_c - se_b)[rng.integers(0, len(y), len(y))]) for _ in range(4000)]\n"
    "    win = lambda k: np.mean([se_c[i].mean() < se_b[i].mean()\n"
    "                             for i in (rng.choice(len(y), k, replace=False) for _ in range(4000))])\n"
    "    sel_rows.append((cname, np.mean(runs), np.std(runs), np.sqrt(np.mean(adv_w * se_c)),\n"
    "                     np.mean(np.array(boot) < 0), win(29), win(67)))\n"
    "\n"
    "selection = pd.DataFrame(sel_rows, columns=['candidate', 'nested RMSE', 'sd over seeds', 'adv-weighted RMSE',\n"
    "                                            'P(better) bootstrap', 'P(win) public 29', 'P(win) private 67'])\n"
    "print(selection.round(4).to_string(index=False))\n"
    "\n"
    "best_row = selection.loc[selection['nested RMSE'].idxmin()]\n"
    "base_score = selection.set_index('candidate').loc[BASELINE, 'nested RMSE']\n"
    "CHOSEN = best_row['candidate'] if base_score - best_row['nested RMSE'] > CFG.ADOPT_MIN_GAIN else BASELINE\n"
    "print(f\"\\nbest candidate: {best_row['candidate']} ({best_row['nested RMSE']:.4f}), \"\n"
    "      f\"baseline {base_score:.4f}, required gain {CFG.ADOPT_MIN_GAIN}\")\n"
    "chosen_cols = CANDIDATES[CHOSEN]\n"
    "P_oof  = np.column_stack([OOF[c] for c in chosen_cols])\n"
    "P_test = np.column_stack([TEST[c] for c in chosen_cols])\n"
    "weights = pd.Series(nnls_weights(P_oof, y), index=chosen_cols)\n"
    "blend_oof = P_oof @ weights.values\n"
    "nested = nested_blend(chosen_cols, CFG.SEED)\n"
    "print(f'\\nchosen: {CHOSEN}')\n"
    "print(weights.round(3).to_string())"
))

cells.append(md(
    "#### Insights\n\n"
    "> The equal-weight average of the two pseudo-labelled foundation models, blended with the SVR, is "
    "the version 5 candidate. Locally, on the same 15 folds, it improved the version 3 blend by about "
    "0.02 to 0.025 RMSE (7.296 against 7.316 on the averaged out-of-fold predictions, and 7.443 against "
    "7.468 averaged over single repeats), with a paired bootstrap probability of about 0.88 of being "
    "better. The same three models with free NNLS weights score worse than version 3 because of the "
    "weight instability described above, which is why the averaged form is used.\n\n"
    "> Version 6 keeps that foundation average and replaces the SVR by the plateau-averaged, less "
    "regularised SVR. Locally this improved the nested RMSE in every repeat (7.251 against 7.296 on the "
    "averaged out-of-fold predictions), for a cumulative gain of about 0.065 over version 3.\n\n"
    "> Even a gain of this size is hard to confirm on 29 public rows, as the public-size win "
    "probability shows, while the private-size probability is clearly higher. The decision is made on "
    "cross-validation.\n\n"
    "> The adversarially weighted RMSE is lower than the plain RMSE for every candidate, which means "
    "the training rows that resemble the test set are, if anything, slightly easier. There is no sign "
    "that the test set is harder than the validation folds."
))

cells.append(md(
    "## Calibration: Is the Tail Really Under-Predicted?\n\n"
    "Version 1 reported that rows with a true score above 70 are under-predicted by 13 to 20 points. "
    "Grouping by the true value is misleading, because even a perfect conditional-mean model shows "
    "that pattern (regression to the mean). The fair check groups rows by the prediction and asks "
    "whether the mean target in each group matches the mean prediction."
))

cells.append(code(
    "cal = pd.DataFrame({'pred': blend_oof, 'y': y})\n"
    "cal['bin'] = pd.qcut(cal.pred, 10)\n"
    "cal_tab = cal.groupby('bin', observed=True).agg(mean_pred=('pred', 'mean'), mean_y=('y', 'mean'), n=('y', 'size'))\n"
    "cal_tab['gap'] = cal_tab.mean_y - cal_tab.mean_pred\n"
    "print(cal_tab.round(2).to_string())\n"
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
    "ax[0].plot(cal_tab.mean_pred, cal_tab.mean_y, 'o-')\n"
    "ax[0].plot([35, 80], [35, 80], 'r--')\n"
    "ax[0].set_xlabel('mean prediction in decile')\n"
    "ax[0].set_ylabel('mean target in decile')\n"
    "ax[0].set_title('calibration by prediction decile')\n"
    "ax[1].scatter(blend_oof, y, s=8)\n"
    "ax[1].plot([30, 101], [30, 101], 'r--')\n"
    "ax[1].set_xlabel('blend OOF prediction')\n"
    "ax[1].set_ylabel('target')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'calibration.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> The blend is well calibrated. In most prediction deciles the mean target is within about 1.5 "
    "to 2 points of the mean prediction, with no systematic drift, and the top decile is almost exact. The large "
    "under-prediction of true high scores reported in version 1 is regression to the mean, not a "
    "fixable bias, which is also why the linear stretch does not help. The high scores are rows that "
    "look like ordinary rows in feature space."
))

cells.append(md(
    "## How Close Is the Limit?\n\n"
    "Two tests estimate how much signal is left. The first is a semivariogram borrowed from "
    "geostatistics. For every pair of training rows, half the squared difference of their targets is "
    "plotted against their feature distance. Its value at distance zero (the nugget) is the variance "
    "no smoother could remove, and the distance at which it reaches the blend's MSE shows the "
    "resolution the model effectively achieves. The second test fits three different learners to the "
    "blend's out-of-fold residuals. If none of them reaches a positive cross-validated R2, the "
    "remaining error has no structure these features can explain."
))

cells.append(code(
    "d_pair = pdist(Z_tr)\n"
    "g_pair = 0.5 * pdist(y[:, None], 'sqeuclidean')\n"
    "edges = np.quantile(d_pair, np.linspace(0, 0.25, 26))\n"
    "mid = 0.5 * (edges[1:] + edges[:-1])\n"
    "gam = np.array([g_pair[(d_pair >= lo) & (d_pair < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])\n"
    "coef = np.polyfit(mid[:8], gam[:8], 1)\n"
    "mse_blend = rmse(y, blend_oof) ** 2\n"
    "print(f'nugget (extrapolated to d=0) = {max(coef[1], 0):.1f}  -> irreducible RMSE about {np.sqrt(max(coef[1], 0)):.2f}')\n"
    "print(f'blend MSE = {mse_blend:.1f}, reached by the variogram at distance about {(mse_blend - coef[1]) / coef[0]:.2f}')\n"
    "print(f'median nearest-neighbour distance in train = {np.median(d_trtr):.2f}')\n"
    "\n"
    "res_blend = y - blend_oof\n"
    "cv10 = KFold(10, shuffle=True, random_state=CFG.SEED)\n"
    "res_learners = {\n"
    "    'Ridge'   : make_pipeline(StandardScaler(), RidgeCV(np.logspace(-2, 4, 30))),\n"
    "    'kNN(15)' : make_pipeline(StandardScaler(), KNeighborsRegressor(15)),\n"
    "    'CatBoost': CatBoostRegressor(iterations=400, learning_rate=0.02, depth=3, l2_leaf_reg=10,\n"
    "                                  verbose=0, random_seed=CFG.SEED, allow_writing_files=False),\n"
    "}\n"
    "for name, m in res_learners.items():\n"
    "    p = cross_val_predict(m, np.hstack([X, blend_oof[:, None]]), res_blend, cv=cv10)\n"
    "    print(f'{name:9s} CV R2 on blend residuals = {1 - np.mean((res_blend - p) ** 2) / np.var(res_blend):+.4f}')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "ax.plot(mid, gam, 'o-', label='empirical semivariance')\n"
    "xx = np.linspace(0, mid[7], 20)\n"
    "ax.plot(xx, np.polyval(coef, xx), 'r--', label='extrapolation to d = 0')\n"
    "ax.axhline(mse_blend, c='k', ls=':', label='blend MSE')\n"
    "ax.axhline(y.var(), c='gray', ls=':', label='variance of y')\n"
    "ax.set_xlabel('pair distance (standardised features)')\n"
    "ax.set_ylabel('semivariance of target')\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'variogram.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> No learner finds structure in the residuals. All three reach a cross-validated R2 at or below "
    "zero, even when they are given the blend prediction as an input, so the information these "
    "features carry about the score is used up.\n\n"
    "> The variogram explains why. Its extrapolated nugget is small (an irreducible RMSE of roughly 3), "
    "so the target itself is not very noisy, but the blend's MSE is reached at a pair distance of "
    "about 1, which is the typical nearest-neighbour distance in this dataset. The model is about as "
    "accurate as knowing the score of the nearest training record. The remaining error comes from how "
    "sparsely 382 rows cover a 24-dimensional space, and more data, not a different model, would be "
    "needed to go further."
))

cells.append(md(
    "## Why Pseudo-Labelling Helps\n\n"
    "The variogram says the error is limited by how far each record is from its nearest labelled "
    "neighbour. If pseudo-labelling works by filling those gaps, its gain should be near zero for "
    "records that already have a close training neighbour and largest for records that do not. The "
    "reduction in squared error is grouped by the distance from each validation row to its nearest "
    "training row in the first repeat."
))

cells.append(code(
    "d_nearest = np.zeros(len(y))\n"
    "for tri, vai in FOLDS[:CFG.N_SPLITS]:\n"
    "    d_nearest[vai] = NearestNeighbors(n_neighbors=1).fit(Z_tr[tri]).kneighbors(Z_tr[vai])[0][:, 0]\n"
    "pl_gain = pd.DataFrame({'distance': d_nearest,\n"
    "                        'mse_reduction': (y - OOF['TabICLv2-v2']) ** 2 - (y - OOF['TabICLv2-PL']) ** 2})\n"
    "pl_gain['bin'] = pd.qcut(pl_gain.distance, 4)\n"
    "by_dist = pl_gain.groupby('bin', observed=True).mse_reduction.agg(['mean', 'count'])\n"
    "print(by_dist.round(3).to_string())\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(13, 4))\n"
    "ax[0].scatter(OOF['TabICLv2-v2'], OOF['TabICLv2-PL'], s=8)\n"
    "ax[0].plot([30, 90], [30, 90], 'r--')\n"
    "ax[0].set_xlabel('TabICLv2 bagged OOF')\n"
    "ax[0].set_ylabel('TabICLv2 pseudo-labelled OOF')\n"
    "ax[1].bar(range(len(by_dist)), by_dist['mean'])\n"
    "ax[1].set_xticks(range(len(by_dist)), [str(i) for i in by_dist.index], fontsize=7)\n"
    "ax[1].axhline(0, c='k')\n"
    "ax[1].set_title('MSE reduction by distance to nearest training row')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'pseudo_label_gain.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> The gain follows the density argument. In the full local run, rows whose nearest training "
    "neighbour is closer than about 0.8 gained the least (about 0.02 in squared error), while rows "
    "between about 1.2 and 1.8 from the nearest training row gained the most (about 3.4). Pseudo-labelled test rows act as extra, slightly noisy neighbours "
    "exactly where the labelled data is sparse, which is the limit version 2 identified. It is not a "
    "leak and not a seed effect, and it is the one lever in this study that attacks that limit "
    "directly."
))

cells.append(md(
    "## Which Validation Scheme Looks Like the Test Set?\n\n"
    "The pseudo-labelled TabICLv2 wins under random K-fold but loses under the similarity-grouped "
    "folds. Pseudo-labelling can fail when the query rows are far from labelled data and clustered "
    "together, because their first-pass errors are then correlated and feeding them back reinforces "
    "them. Both conditions can be measured from the features alone: the distance from each query row to "
    "its nearest labelled row, and to its nearest other query row. The test set is compared with the "
    "validation folds of both schemes using the two-sample Kolmogorov-Smirnov statistic."
))

cells.append(code(
    "GROUP_FOLDS = list(GroupKFold(5).split(X, y, sim_groups))\n"
    "\n"
    "\n"
    "def query_geometry(Z_lab, Z_q):\n"
    "    d_lab = NearestNeighbors(n_neighbors=1).fit(Z_lab).kneighbors(Z_q)[0][:, 0]\n"
    "    d_pool = NearestNeighbors(n_neighbors=2).fit(Z_q).kneighbors(Z_q)[0][:, 1]\n"
    "    return d_lab, d_pool\n"
    "\n"
    "\n"
    "geo = {'test': query_geometry(Z_tr, Z_te)}\n"
    "for name, folds in [('random folds', FOLDS[:CFG.N_SPLITS]), ('group folds', GROUP_FOLDS)]:\n"
    "    parts = [query_geometry(Z_tr[tri], Z_tr[vai]) for tri, vai in folds]\n"
    "    geo[name] = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))\n"
    "for name, (d_lab, d_pool) in geo.items():\n"
    "    ks_lab = stats.ks_2samp(d_lab, geo['test'][0]).statistic\n"
    "    ks_pool = stats.ks_2samp(d_pool, geo['test'][1]).statistic\n"
    "    print(f'{name:13s} median distance to labelled={np.median(d_lab):.3f}  to other query={np.median(d_pool):.3f}'\n"
    "          f'  | KS vs test: {ks_lab:.3f} / {ks_pool:.3f}')\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(13, 4))\n"
    "for name, (d_lab, d_pool) in geo.items():\n"
    "    ax[0].hist(d_lab, bins=30, density=True, histtype='step', lw=2, label=name)\n"
    "    ax[1].hist(d_pool, bins=30, density=True, histtype='step', lw=2, label=name)\n"
    "ax[0].set_title('distance to nearest labelled row')\n"
    "ax[1].set_title('distance to nearest other query row')\n"
    "ax[0].legend()\n"
    "ax[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'test_geometry.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> The test set matches the random folds almost exactly (KS statistics of about 0.08 and 0.07) and "
    "differs strongly from the grouped folds (about 0.39 and 0.30). By construction, no grouped "
    "validation row lies within distance 1 of a labelled row, and the grouped query rows sit close to "
    "each other, which is exactly the failure setting for pseudo-labelling. The test rows do not look "
    "like that, so the grouped result is a stress test that does not describe the test set, and the "
    "random-fold gain is the relevant estimate."
))

cells.append(md(
    "## Does the Gain Shrink With More Labelled Data?\n\n"
    "In cross-validation the labelled context has about 306 rows, while the final model uses 382. If "
    "pseudo-labelling mainly helps when labelled data is scarce, its gain could be smaller at test time "
    "than cross-validation suggests. Plain and pseudo-labelled TabICLv2 are compared at several labelled "
    "sizes n, keeping the ratio of pseudo to labelled context rows at its test-time value by using n/4 "
    "query rows. Two random states per model keep the runtime moderate."
))

cells.append(code(
    "LC_SEEDS = [0, 1]\n"
    "lc_rng = np.random.default_rng(CFG.SEED)\n"
    "lc_rows = []\n"
    "for n_lab in [120, 180, 240, 306]:\n"
    "    e_plain, e_pl = [], []\n"
    "    for tri, vai in FOLDS:\n"
    "        tri_n = lc_rng.choice(tri, n_lab, replace=False) if n_lab < len(tri) else tri\n"
    "        vai_q = lc_rng.choice(vai, min(n_lab // 4, len(vai)), replace=False)\n"
    "        p_plain = BaggedTabICL(LC_SEEDS, GPU_A, norm_methods=CFG.ICL_NORM).fit(X[tri_n], y[tri_n]).predict(X[vai_q])\n"
    "        p_pl = PseudoLabelTabICL(seeds=LC_SEEDS, device=GPU_A).fit(X[tri_n], y[tri_n]).predict(X[vai_q])\n"
    "        e_plain.append(np.mean((y[vai_q] - p_plain) ** 2))\n"
    "        e_pl.append(np.mean((y[vai_q] - p_pl) ** 2))\n"
    "    better = np.mean(np.sqrt(e_pl) < np.sqrt(e_plain))\n"
    "    lc_rows.append((n_lab, np.sqrt(np.mean(e_plain)), np.sqrt(np.mean(e_pl)), better))\n"
    "lc = pd.DataFrame(lc_rows, columns=['n_labelled', 'plain', 'pseudo-label', 'folds PL better'])\n"
    "lc['gain'] = lc['plain'] - lc['pseudo-label']\n"
    "print(lc.round(4).to_string(index=False))\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(13, 4))\n"
    "ax[0].plot(lc.n_labelled, lc.plain, 'o-', label='plain')\n"
    "ax[0].plot(lc.n_labelled, lc['pseudo-label'], 'o-', label='pseudo-label')\n"
    "ax[0].set_xlabel('labelled context size')\n"
    "ax[0].set_ylabel('RMSE')\n"
    "ax[0].legend()\n"
    "ax[1].plot(lc.n_labelled, lc.gain, 'o-')\n"
    "ax[1].axhline(0, c='k')\n"
    "ax[1].set_title('pseudo-label gain vs labelled size')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'learning_curve.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> The gain does not shrink with more labelled data. In the local run it was slightly negative at "
    "120 labelled rows (-0.02), then +0.07, +0.12 and +0.08 at 180, 240 and 306 rows. Pseudo-labelling "
    "needs a first pass that is accurate enough to be worth copying, and above about 180 labelled rows "
    "it is. At 382 rows the gain should be at least as large as in cross-validation, so the "
    "cross-validation estimate is not optimistic for this reason. The version 3 public score moved "
    "less than cross-validation predicted (7.902 to 7.888), which is within the noise of 29 rows."
))

cells.append(md(
    "## Can Uncertainty Filtering Make Pseudo-Labelling Robust?\n\n"
    "A standard remedy for confirmation bias is to pseudo-label only the rows the model is confident "
    "about. TabICLv2 returns predictive quantiles, so the inter-quartile range is a per-row uncertainty. "
    "The most certain fraction of the query rows is kept as context, and every variant is scored under "
    "both validation schemes with two random states."
))

cells.append(code(
    "class FilteredPseudoLabelTabICL(PseudoLabelTabICL):\n"
    "    def __init__(self, keep, **kw):\n"
    "        super().__init__(**kw)\n"
    "        self.keep = keep\n"
    "\n"
    "    def predict(self, B):\n"
    "        outs = [TabICLRegressor(device=self.device, random_state=s, norm_methods=CFG.ICL_NORM).fit(self.A, self.t)\n"
    "                .predict(B, output_type=['mean', 'quantiles'], alphas=[0.25, 0.75]) for s in self.seeds]\n"
    "        first = np.mean([o['mean'] for o in outs], 0)\n"
    "        iqr = np.mean([np.asarray(o['quantiles'])[:, 1] - np.asarray(o['quantiles'])[:, 0] for o in outs], 0)\n"
    "        confident = iqr <= np.quantile(iqr, self.keep)\n"
    "        part = np.arange(len(B)) % self.n_split\n"
    "        out = np.empty(len(B))\n"
    "        for k in range(self.n_split):\n"
    "            q, c = part == k, (part != k) & confident\n"
    "            out[q] = self._bag(np.vstack([self.A, B[c]]), np.r_[self.t, first[c]], B[q])\n"
    "        return out\n"
    "\n"
    "\n"
    "def cv_scheme(make, folds):\n"
    "    n_rep = len(folds) // CFG.N_SPLITS\n"
    "    oof = np.zeros((n_rep, len(y)))\n"
    "    for i, (tri, vai) in enumerate(folds):\n"
    "        oof[i // CFG.N_SPLITS, vai] = make().fit(X[tri], y[tri]).predict(X[vai])\n"
    "    return np.mean([rmse(y, o) for o in oof])\n"
    "\n"
    "\n"
    "robust_cfg = {\n"
    "    'plain'       : lambda: BaggedTabICL(LC_SEEDS, GPU_A, norm_methods=CFG.ICL_NORM),\n"
    "    'PL keep=1.0' : lambda: PseudoLabelTabICL(seeds=LC_SEEDS, device=GPU_A),\n"
    "    'PL keep=0.75': lambda: FilteredPseudoLabelTabICL(0.75, seeds=LC_SEEDS, device=GPU_A),\n"
    "    'PL keep=0.5' : lambda: FilteredPseudoLabelTabICL(0.5, seeds=LC_SEEDS, device=GPU_A),\n"
    "}\n"
    "robust = pd.DataFrame({\n"
    "    name: {'random folds': cv_scheme(make, FOLDS), 'group folds': cv_scheme(make, GROUP_FOLDS)}\n"
    "    for name, make in robust_cfg.items()\n"
    "}).T\n"
    "robust['gain random'] = robust.loc['plain', 'random folds'] - robust['random folds']\n"
    "robust['gain group'] = robust.loc['plain', 'group folds'] - robust['group folds']\n"
    "print(robust.round(4).to_string())"
))

cells.append(md(
    "#### Insights\n\n"
    "> Filtering moves the result along a single trade-off line instead of fixing it. Locally, keeping "
    "all pseudo labels gained about 0.10 under random folds and lost about 0.12 under grouped folds, "
    "keeping 75% gained 0.06 and lost 0.03, and keeping 50% gained 0.03 and lost almost nothing. The "
    "uncertainty signal is real (the IQR correlates with the error at about 0.31), but removing "
    "pseudo rows also removes the extra neighbours that produce the gain.\n\n"
    "> Because the test geometry matches the random folds, the full pseudo-labelling configuration is "
    "kept. Two further extensions were tested locally and also rejected, since both changed the nested "
    "blend RMSE by less than the spread over seeds: training the SVR on the pseudo-labelled rows as well "
    "(-0.005) and a second pseudo-labelling round (-0.006)."
))

cells.append(md(
    "## How Much to Trust the Public Leaderboard\n\n"
    "The public leaderboard uses about 30% of the 96 test rows, roughly 29 records. Resampling 29 "
    "out-of-fold residuals many times shows how much the public score can move by chance alone."
))

cells.append(code(
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
    "ax.axvline(7.947, c='r', ls='--', label='v1 public score')\n"
    "ax.axvline(7.888, c='g', ls='--', label='v3 public score')\n"
    "ax.legend()\n"
    "ax.set_title('leaderboard RMSE spread from sampling alone')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / 'lb_noise.png')\n"
    "plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> The same model can score anywhere from about 4.7 to 9.9 on the public leaderboard purely from "
    "which 29 rows are public. The version 1 public score of 7.947 sits well inside that range, so it "
    "is consistent with the cross-validation estimate and says nothing about overfitting either way.\n\n"
    "> Model selection is therefore done by cross-validation on 382 rows. The public leaderboard is "
    "used only to check that the file is valid and the score is inside the expected range. Tuning "
    "toward the public score would fit the noise of 29 rows."
))

# ---------------------------------------------------------------------------
# Section 8: Submission
# ---------------------------------------------------------------------------
cells.append(section("Submission", "8"))

cells.append(md(
    "The weights of the blend chosen in Section 7 are applied to the test predictions of the models refitted on "
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
    "print(f'train OOF blend mean {blend_oof.mean():.2f}  range {blend_oof.min():.2f} to {blend_oof.max():.2f}')\n"
    "\n"
    "for ref in ['v3: TabICL-PL + SVR', 'v5: mean(ICL-PL, PFN35-PL) + SVR', BASELINE]:\n"
    "    ref_pred = np.column_stack([TEST[c] for c in CANDIDATES[ref]]) @ nnls_weights(\n"
    "        np.column_stack([OOF[c] for c in CANDIDATES[ref]]), y)\n"
    "    print(f'vs {ref:22s}: corr={np.corrcoef(pred, ref_pred)[0, 1]:.4f}  '\n"
    "          f'RMS difference={rmse(pred, ref_pred):.3f}  max |difference|={np.abs(pred - ref_pred).max():.2f}')"
))

cells.append(md(
    "#### Insights\n\n"
    "> The test predictions have a mean and range similar to the out-of-fold predictions on the "
    "training set, which is consistent with the mild shift measured in Section 3. The file "
    "`submission.csv` in the output directory is the final submission. Its agreement with the version 1 and version 2 "
    "test predictions is printed so that the size of the change is visible before submitting."
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
