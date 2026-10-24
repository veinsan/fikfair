import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "final.ipynb"
HR = '<hr style="border: 2px solid #8E7B6B; margin-top: 10px;">'


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": uuid4().hex[:8],
        "metadata": {}, "outputs": [], "source": source,
    }


def section(title: str, anchor: str) -> dict:
    return md(f"{HR}\n\n# {title} <a name=\"{anchor}\"></a>\n\n{HR}")



cells = []

# ---------------------------------------------------------------------------
# Banner + Daftar Isi
# ---------------------------------------------------------------------------
cells.append(md(
    f"{HR}\n"
    "<br>\n"
    "<h1 style=\"font-family:verdana; font-size:26px\"><center>~ Data Science Competition FIK FAIR 2026: "
    "Prediksi Anonymous Competition Score ~</center></h1>\n"
    "<h3 style=\"font-family:verdana; font-size:14px; color:gray\"><center>Foundation model tabular, "
    "informasi urutan baris, dan koreksi residual bertingkat</center></h3>\n"
    f"{HR}\n\n"
    "# **NAMA TIM**"
))

cells.append(md(
    f"{HR}\n\n"
    "# Daftar Isi\n\n"
    "1. [**Pendahuluan**](#1)\n"
    "2. [**Inisialisasi**](#2)\n"
    "3. [**Analisis Data Eksploratif**](#3)\n"
    "4. [**Prapemrosesan dan Rekayasa Fitur**](#4)\n"
    "5. [**Skema Validasi**](#5)\n"
    "6. [**Pemodelan**](#6)\n"
    "7. [**Inferensi dan Submisi**](#7)\n"
    "8. [**Dashboard Solusi**](#8)\n"
    "9. [**Pasca-Analisis**](#9)\n"
    "10. [**Daftar Pustaka**](#10)"
))

# ---------------------------------------------------------------------------
# Section 1: Pendahuluan
# ---------------------------------------------------------------------------
cells.append(section("Pendahuluan", "1"))

cells.append(md(
    "## Kompetisi\n\n"
    "Pada **Data Science Competition FIK FAIR 2026**, peserta diminta memprediksi *Anonymous Competition "
    "Score*, sebuah skor komposit yang rumusnya dirahasiakan panitia. Setiap observasi hanya dideskripsikan "
    "oleh 24 fitur numerik anonim yang berasal dari indikator lingkungan dan iklim, lalu ditransformasi "
    "berlapis (normalisasi rank, transformasi terbatas, proyeksi linear privat, standardisasi, perturbasi "
    "terkontrol, dan pembulatan). Akibatnya, fitur tidak memiliki satuan maupun makna fisik yang dapat "
    "ditafsirkan secara langsung.\n\n"
    "- **Tugas**: regresi tabular, satu nilai kontinu per `record_id`\n"
    "- **Data**: 382 baris *train* berlabel, 96 baris *test*, 24 fitur\n"
    "- **Metrik**: *Root Mean Squared Error* (RMSE), MAE sebagai *tie-breaker*\n"
    "- **Leaderboard**: *public* sekitar 30% *test* (+/-29 baris), *private* sekitar 70% (+/-67 baris)\n"
    "- **Platform**: Kaggle Notebook, GPU T4 x2 (hanya untuk mode `FULL`)"
))

cells.append(md(
    "---\n\n"
    "## Dataset\n\n"
    "*Dataset* resmi terdiri dari `train.csv` (fitur dan `target`), `test.csv` (fitur saja), dan "
    "`submission_template.csv`. Kolom `record_id` adalah ID teknis yang diacak dan tidak membawa makna "
    "geografis. Fitur `feature_01` sampai `feature_24` bersifat tanpa dimensi dan disimpan dengan enam "
    "angka desimal. Target bernilai kontinu sekitar 32,6 sampai 100,4.\n\n"
    "Dengan hanya 382 baris berlabel, risiko terbesar kompetisi ini bukan kekurangan model, melainkan "
    "*overfitting*, validasi yang terlalu optimis, dan godaan mengejar skor *public* yang hanya dihitung "
    "dari sekitar 29 baris."
))

cells.append(md(
    "---\n\n"
    "## Metrik\n\n"
    "RMSE mengukur besar galat tipikal dalam satuan target. Karena galat dikuadratkan, sedikit baris "
    "dengan galat besar mendominasi skor, sehingga ekor kanan target (skor di atas 70) menjadi perhatian "
    "utama sepanjang analisis.\n\n"
    "$$\\text{RMSE} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^{n}\\left(y_i - \\hat{y}_i\\right)^2}$$"
))

cells.append(md(
    "---\n\n"
    "## Pendekatan: *Foundation Model* + Urutan Baris + Koreksi Residual Bertingkat\n\n"
    "Solusi akhir tersusun dari tiga lapisan. **Lapisan fitur** memprediksi skor hanya dari 24 fitur "
    "menggunakan *tabular foundation model* TabICLv2 dan TabPFN-3.5 (prediksi lewat *in-context learning*, "
    "tanpa *training* gradien pada data kompetisi) serta SVR RBF pada beberapa *view* PCA. Ketiganya "
    "memakai *pseudo-labelling* transduktif.\n\n"
    "**Lapisan urutan** memanfaatkan temuan EDA bahwa baris pada kedua berkas tersusun menurut target. "
    "Penggunaan urutan baris ini telah dikonfirmasi panitia sebagai bagian dari EDA dan prapemrosesan "
    "peserta (4 Oktober 2026). **Lapisan koreksi** memperbaiki posisi relatif tiap baris *test* dan "
    "memodelkan penyimpangan lokal target dari urutannya melalui empat tahap bersarang.\n\n"
    "```\n"
    "24 fitur mentah\n"
    "    |\n"
    "    +-- TabICLv2 (norm=none, 4 seed) + pseudo-label  --+\n"
    "    +-- TabPFN-3.5 + pseudo-label                     --+-- rata-rata = FM-PL-mean\n"
    "    +-- PCA(10..20) whitened -> SVR plateau + pseudo-label = SVR-plateau-PL\n"
    "         |\n"
    "    prediksi fitur = 0.7 FM-PL-mean + 0.3 SVR-plateau-PL          (lapisan fitur)\n"
    "         |\n"
    "    v9  = rata-rata(peta kuantil posisi, prediksi fitur terurut)  (lapisan urutan)\n"
    "    v10 = 0.5 v9  + 0.5 koreksi rank dari tetangga PCA-12         (lapisan koreksi)\n"
    "    v11 = 0.5 v10 + 0.5 Ridge pada residual fitur ter-detrend\n"
    "    v12 = 0.5 v11 + 0.5 detrend local-linear + interaksi posisi\n"
    "    v13 = 0.5 v12 + 0.5 Kernel Ridge RBF pada residual  -> submisi\n"
    "```\n\n"
    "Seluruh tahap memakai satu skema *repeated 5-fold* (3 *repeat*) yang sama. Setiap pemilihan kandidat "
    "di dalam tahap dilakukan dengan *inner cross-validation* pada baris *train* luar saja, dan bobot 50:50 "
    "ditetapkan sebelum evaluasi, sehingga estimasi RMSE di setiap tahap tetap jujur."
))

cells.append(md(
    "---\n\n"
    "## Perjalanan Solusi\n\n"
    "Solusi dikembangkan dalam 13 iterasi. Tabel berikut merangkum iterasi yang mengubah model. Skor "
    "*public* hanya dicatat sebagai pemeriksaan arah, tidak pernah dipakai untuk memilih parameter.\n\n"
    "| Versi | Perubahan utama | RMSE CV | RMSE *public* |\n"
    "|---|---|---:|---:|\n"
    "| v1 | TabICLv2 + SVR multi-PCA, *blend* NNLS | 7,41 | 7,947 |\n"
    "| v2 | TabICLv2 tanpa normalisasi *power*, *bagging* 4 seed | 7,40 | 7,902 |\n"
    "| v3 | *Pseudo-labelling* transduktif TabICLv2 | 7,32 | 7,888 |\n"
    "| v5 | Rata-rata TabICLv2-PL dan TabPFN-3.5-PL | 7,30 | 7,888 |\n"
    "| v6b | SVR *plateau* dengan *clipping* rentang target | 7,26 | 7,837 |\n"
    "| v8 | *Pseudo-labelling* untuk SVR | 7,23 | 7,900 |\n"
    "| v9 | Pemetaan urutan baris | 3,613 | 2,387 |\n"
    "| v10 | Koreksi rank dari tetangga fitur | 3,401 | 2,107 |\n"
    "| v11 | Ridge pada residual fitur ter-*detrend* | 3,303 | 1,955 |\n"
    "| v12 | *Detrend local-linear* + interaksi posisi | 3,179 | 1,825 |\n"
    "| **v13** | ***Kernel Ridge* pada residual (final)** | **3,016** | - |\n\n"
    "RMSE CV v1 sampai v8 adalah RMSE *blend nested*, sedangkan mulai v9 adalah rata-rata RMSE 15 *fold* "
    "luar. Kedua estimand tidak dapat dibandingkan secara langsung, namun lompatan dari sekitar 7,2 ke 3,6 "
    "jauh melampaui perbedaan estimand. Notebook ini menghitung ulang seluruh angka v9 sampai v13."
))

cells.append(md(
    "---\n\n"
    "## Referensi Utama\n\n"
    "- `docs/eda_findings.md` -- catatan seluruh eksperimen v1 sampai v9, termasuk ide yang ditolak\n"
    "- `docs/v10_order_refinement.md` sampai `docs/v13_deep_analysis.md` -- analisis dan validasi tahap koreksi\n"
    "- `eda/` -- *script* analisis yang mencetak angka dan menyimpan grafik untuk setiap iterasi"
))

# ---------------------------------------------------------------------------
# Section 2: Inisialisasi
# ---------------------------------------------------------------------------
cells.append(section("Inisialisasi", "2"))

cells.append(md("## Environment"))

cells.append(code("!nvidia-smi"))

cells.append(md(
    "> Catatan untuk juri dan panitia: notebook berjalan dalam mode `INFERENCE` secara bawaan. Mode ini "
    "memuat keluaran lapisan fitur (TabICLv2, TabPFN-3.5, SVR) dari *run* GPU Kaggle yang dibekukan pada "
    "dataset `fikfair-artifacts`, lalu menjalankan seluruh analisis, lapisan urutan, dan keempat tahap "
    "koreksi di CPU dalam beberapa menit. Hasilnya diverifikasi identik dengan `submission_v13.csv` yang "
    "disubmit. Mode `FULL` melatih ulang lapisan fitur di dua GPU T4 dan membutuhkan token Prior Labs."
))

cells.append(md("---\n\n## Instalasi"))

cells.append(md(
    "Baris berikut memasang pustaka *foundation model* yang hanya dibutuhkan pada mode `FULL`. Pada mode "
    "`INFERENCE`, kegagalan instalasi (misalnya tanpa internet) tidak memengaruhi hasil."
))

cells.append(code("%pip install -q tabicl tabpfn"))

cells.append(md("---\n\n## Pustaka"))

cells.append(md(
    "Seluruh impor dikelompokkan dalam satu sel agar mudah ditelusuri. NumPy, pandas, SciPy, dan "
    "scikit-learn menjalankan seluruh analisis dan tahap koreksi, sedangkan PyTorch, TabICL, dan TabPFN "
    "hanya diperlukan untuk melatih ulang lapisan fitur sehingga impornya dibuat opsional."
))

cells.append(code(
    "import hashlib\n"
    "import io\n"
    "import os\n"
    "import platform\n"
    "import random\n"
    "import sys\n"
    "import time\n"
    "import warnings\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import scipy\n"
    "import sklearn\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "from matplotlib.gridspec import GridSpec\n"
    "from scipy import stats\n"
    "from scipy.cluster.hierarchy import leaves_list, linkage\n"
    "from scipy.ndimage import gaussian_filter1d\n"
    "from scipy.spatial.distance import cdist, pdist\n"
    "from scipy.special import betainc\n"
    "from sklearn.compose import TransformedTargetRegressor\n"
    "from sklearn.decomposition import PCA\n"
    "from sklearn.ensemble import RandomForestClassifier\n"
    "from sklearn.isotonic import IsotonicRegression\n"
    "from sklearn.kernel_ridge import KernelRidge\n"
    "from sklearn.linear_model import Ridge\n"
    "from sklearn.metrics import roc_auc_score\n"
    "from sklearn.mixture import GaussianMixture\n"
    "from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, cross_val_predict\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.svm import SVR\n"
    "from statsmodels.nonparametric.smoothers_lowess import lowess\n"
    "\n"
    "try:\n"
    "    import torch\n"
    "    from tabicl import TabICLRegressor\n"
    "    from tabpfn import TabPFNRegressor\n"
    "    from tabpfn.constants import ModelVersion\n"
    "except ImportError:\n"
    "    torch = TabICLRegressor = TabPFNRegressor = ModelVersion = None\n"
    "\n"
    "try:\n"
    "    from kaggle_secrets import UserSecretsClient\n"
    "except ImportError:\n"
    "    UserSecretsClient = None\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "sns.set_theme(style='whitegrid', context='notebook')\n"
    "print('Python :', sys.version.split()[0], '|', platform.system(), platform.release())\n"
    "print('numpy', np.__version__, '| pandas', pd.__version__, '| scipy', scipy.__version__, '| scikit-learn', sklearn.__version__)\n"
    "print('foundation models available:', TabICLRegressor is not None)"
))

cells.append(md("---\n\n## Seed"))

cells.append(md(
    "Pengaturan *seed* dilakukan pada semua sumber keacakan. Seluruh tahap CPU bersifat deterministik, "
    "sedangkan *foundation model* di GPU tetap dapat menghasilkan sedikit galat numerik antar eksekusi."
))

cells.append(code(
    "def seed_everything(seed: int = 42):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "    if torch is not None:\n"
    "        torch.manual_seed(seed)\n"
    "        torch.cuda.manual_seed_all(seed)\n"
    "\n"
    "seed_everything(42)"
))

cells.append(md("---\n\n## Settings"))

cells.append(md(
    "Kelas `Settings` memusatkan seluruh konfigurasi: mode eksekusi, jalur data dan artefak, "
    "*hyperparameter* lapisan fitur, serta *seed* pemilihan untuk setiap tahap koreksi. *Seed* tahap "
    "sama persis dengan yang dipakai saat pengembangan agar submisi dapat direproduksi. Ubah `RUN_MODE` "
    "menjadi `'FULL'` untuk melatih ulang lapisan fitur di GPU."
))

cells.append(code(
    "class Settings:\n"
    "    SEED        = 42\n"
    "\n"
    "    # Mode ('INFERENCE' memuat artefak lapisan fitur, 'FULL' melatih ulang di GPU)\n"
    "    RUN_MODE    = 'INFERENCE'\n"
    "\n"
    "    # Validation\n"
    "    N_SPLITS    = 5\n"
    "    N_REPEATS   = 3\n"
    "\n"
    "    # Feature layer\n"
    "    LOG_SHIFT   = 30.0\n"
    "    PCA_DIMS    = [10, 12, 14, 16, 20]\n"
    "    SVR_CELLS   = [(0.7, 10), (0.7, 30), (1.0, 10), (1.0, 30)]\n"
    "    ICL_SEEDS   = [0, 1, 2, 3]\n"
    "    ICL_NORM    = ['none']\n"
    "    PL_SPLIT    = 4\n"
    "    SVR_PL_SPLITS = (2, 4, 8)\n"
    "    FEAT_W      = np.array([0.7, 0.3])\n"
    "\n"
    "    # Refinement stages: (inner seed offset, final selection seed, development choice)\n"
    "    STAGES = {\n"
    "        'v10': (100, 4900, 'k10_a1.0_s0.005'),\n"
    "        'v11': (5200, 5299, 'ridge10_w1.0'),\n"
    "        'v12': (560100, 560199, 'local_linear_interaction1_ridge10'),\n"
    "        'v13': (6000, 6099, 'kernel_a0.1_g2'),\n"
    "    }\n"
    "\n"
    "    # Visualisation\n"
    "    C_TRAIN, C_TEST, C_ACCENT, C_GREY = '#8E7B6B', '#2e86de', '#c0392b', '#7f8c8d'\n"
    "    STAGE_COLORS = dict(zip(['v9', 'v10', 'v11', 'v12', 'v13'], sns.color_palette('flare', 5)))\n"
    "\n"
    "    # Paths\n"
    "    _ON_KAGGLE  = Path('/kaggle/input').exists()\n"
    "    if _ON_KAGGLE:\n"
    "        DATA_DIR     = next(p.parent for p in Path('/kaggle/input').rglob('train.csv')\n"
    "                            if (p.parent / 'submission_template.csv').exists())\n"
    "        ARTIFACT_DIR = next(Path('/kaggle/input').rglob('feature_oof.npz')).parent\n"
    "        OUTPUT_DIR   = Path('/kaggle/working')\n"
    "    else:\n"
    "        DATA_DIR     = Path('../data')\n"
    "        ARTIFACT_DIR = Path('../artifacts')\n"
    "        OUTPUT_DIR   = Path('../outputs/fikfair_final')\n"
    "    FIG_DIR     = OUTPUT_DIR / 'figures'\n"
    "    SUBMISSION_CSV = OUTPUT_DIR / 'submission.csv'\n"
    "\n"
    "\n"
    "CFG = Settings()\n"
    "CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
    "\n"
    "print(f'RUN_MODE     : {CFG.RUN_MODE}')\n"
    "print(f'DATA_DIR     : {CFG.DATA_DIR}  exists={CFG.DATA_DIR.exists()}')\n"
    "print(f'ARTIFACT_DIR : {CFG.ARTIFACT_DIR}  exists={CFG.ARTIFACT_DIR.exists()}')\n"
    "print(f'OUTPUT_DIR   : {CFG.OUTPUT_DIR}')"
))

cells.append(md("---\n\n## Token Prior Labs"))

cells.append(md(
    "Checkpoint TabPFN-3.5 bersifat *gated* oleh Prior Labs. Lisensi perlu diterima sekali di "
    "https://ux.priorlabs.ai, lalu token disimpan sebagai Kaggle Secret bernama `TABPFN_TOKEN`. Token hanya "
    "dibaca pada mode `FULL`."
))

cells.append(code(
    "if CFG.RUN_MODE == 'FULL' and UserSecretsClient is not None and 'TABPFN_TOKEN' not in os.environ:\n"
    "    os.environ['TABPFN_TOKEN'] = UserSecretsClient().get_secret('TABPFN_TOKEN')\n"
    "print('TabPFN token set:', bool(os.environ.get('TABPFN_TOKEN')))"
))

cells.append(md("---\n\n## Muat Data"))

cells.append(md(
    "Tiga berkas resmi dimuat **tanpa mengubah urutan baris**, karena urutan tersebut merupakan sumber "
    "informasi penting yang dianalisis pada bagian EDA. Fitur gabungan *train* dan *test* hanya dipakai "
    "untuk statistik tanpa label."
))

cells.append(code(
    "train    = pd.read_csv(CFG.DATA_DIR / 'train.csv')\n"
    "test     = pd.read_csv(CFG.DATA_DIR / 'test.csv')\n"
    "template = pd.read_csv(CFG.DATA_DIR / 'submission_template.csv')\n"
    "\n"
    "FEATURES = [c for c in train.columns if c.startswith('feature_')]\n"
    "X, y     = train[FEATURES].to_numpy(), train['target'].to_numpy()\n"
    "X_te     = test[FEATURES].to_numpy()\n"
    "X_all    = np.vstack([X, X_te])\n"
    "pos_tr, pos_te = np.arange(len(X)), np.arange(len(X_te))\n"
    "\n"
    "print(f'Train: {train.shape}  Test: {test.shape}  Template: {template.shape}  Features: {len(FEATURES)}')"
))

cells.append(md("---\n\n## Muat Artefak"))

cells.append(md(
    "Artefak berikut adalah keluaran *run* GPU yang dibekukan. `feature_oof.npz` berisi prediksi "
    "*out-of-fold* lapisan fitur pada 15 *fold* yang sama (3 *repeat* x 382 baris), `submission_v9.csv` "
    "berisi prediksi *test* lapisan urutan versi 9, `benchmark_oof.csv` berisi prediksi *out-of-fold* "
    "model pembanding dari benchmark awal, dan `submission_v13.csv` adalah berkas yang disubmit, dipakai "
    "hanya untuk verifikasi di akhir notebook."
))

cells.append(code(
    "feature_art = np.load(CFG.ARTIFACT_DIR / 'feature_oof.npz')\n"
    "benchmark   = pd.read_csv(CFG.ARTIFACT_DIR / 'benchmark_oof.csv')\n"
    "v9_file     = pd.read_csv(CFG.ARTIFACT_DIR / 'submission_v9.csv')\n"
    "\n"
    "assert benchmark['record_id'].tolist() == train['record_id'].tolist()\n"
    "assert feature_art['FM_PL_mean'].shape == (CFG.N_REPEATS, len(y))\n"
    "assert set(v9_file['record_id']) == set(test['record_id']) and v9_file['record_id'].is_unique\n"
    "\n"
    "for name in feature_art.files:\n"
    "    print(f'{name:15s} shape={feature_art[name].shape}')\n"
    "print('benchmark models:', [c for c in benchmark.columns if c != 'record_id'])"
))

# ---------------------------------------------------------------------------
# Section 3: Analisis Data Eksploratif
# ---------------------------------------------------------------------------
cells.append(section("Analisis Data Eksploratif", "3"))

cells.append(md(
    "Bagian EDA menggunakan berkas mentah dengan urutan baris asli. Analisis bergerak dari kualitas data, "
    "distribusi target dan fitur, kesamaan distribusi *train* dan *test*, struktur antarfitur, hubungan "
    "fitur dengan target, kedekatan antarbaris, hingga temuan kunci tentang urutan baris. Setiap temuan "
    "diterjemahkan menjadi keputusan prapemrosesan atau pemodelan."
))

cells.append(md("---\n\n## Ringkasan dan Kualitas Data"))

cells.append(md(
    "Pemeriksaan kualitas mencakup nilai hilang, duplikat persis (di dalam *train*, di dalam *test*, dan "
    "lintas keduanya), keunikan `record_id`, serta presisi penyimpanan fitur. Duplikat lintas *train* dan "
    "*test* adalah bentuk kebocoran paling sederhana sehingga diperiksa lebih dulu."
))

cells.append(code(
    "def rmse(a, b):\n"
    "    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))\n"
    "\n"
    "\n"
    "def n_decimals_ok(A, k=6):\n"
    "    return bool(np.allclose(A * 10 ** k, np.round(A * 10 ** k), atol=1e-6))\n"
    "\n"
    "\n"
    "cross_dup = len(train[FEATURES].merge(test[FEATURES], how='inner'))\n"
    "quality = pd.DataFrame({\n"
    "    'train': [len(train), int(train[FEATURES].isna().sum().sum()), int(train['target'].isna().sum()),\n"
    "              int(train.duplicated(FEATURES).sum()), train['record_id'].is_unique, n_decimals_ok(X)],\n"
    "    'test' : [len(test), int(test[FEATURES].isna().sum().sum()), '-',\n"
    "              int(test.duplicated(FEATURES).sum()), test['record_id'].is_unique, n_decimals_ok(X_te)],\n"
    "}, index=['jumlah baris', 'nilai hilang (fitur)', 'nilai hilang (target)', 'duplikat persis fitur',\n"
    "          'record_id unik', 'fitur tersimpan 6 desimal'])\n"
    "display(quality)\n"
    "print(f'duplikat persis lintas train-test : {cross_dup}')\n"
    "print(f'record_id beririsan train-test    : {len(set(train.record_id) & set(test.record_id))}')\n"
    "\n"
    "summary = pd.DataFrame({\n"
    "    'mean': train[FEATURES].mean(), 'std': train[FEATURES].std(),\n"
    "    'min': train[FEATURES].min(), 'max': train[FEATURES].max(),\n"
    "    'skew': train[FEATURES].skew(), 'kurtosis': train[FEATURES].kurt(),\n"
    "    'n_unique': train[FEATURES].nunique(),\n"
    "})\n"
    "display(summary.round(3).T)"
))

cells.append(md("> ### Insights\n>\n> Data bersih secara struktural: tidak ada nilai hilang, tidak ada duplikat persis di dalam maupun lintas *train* dan *test*, dan `record_id` unik serta tidak beririsan. Setiap fitur memiliki 382 nilai unik dan tersimpan tepat enam desimal, konsisten dengan tahap pembulatan yang disebut panitia.\n>\n> Rata-rata setiap fitur mendekati 0 dan simpangan bakunya mendekati 1, sehingga fitur sudah distandardisasi oleh panitia. Sisa transformasi terbatas masih terlihat dari *skewness* hingga -0,89 (`feature_08`) dan kurtosis 1,81 (`feature_14`). **Keputusan:** tidak ada imputasi maupun deduplikasi, dan standardisasi ulang tetap dilakukan di dalam setiap *fold* untuk model berbasis jarak."))

cells.append(md("---\n\n## Distribusi Target"))

cells.append(md(
    "Bentuk distribusi target menentukan transformasi target dan bagian data mana yang paling menentukan "
    "RMSE. *Gaussian Mixture* dua komponen dipakai untuk menguji apakah target memiliki dua mode, dan kurva "
    "ECDF menunjukkan proporsi baris pada ekor kanan."
))

cells.append(code(
    "gmm = GaussianMixture(2, random_state=CFG.SEED, n_init=10).fit(y[:, None])\n"
    "order_g = np.argsort(gmm.means_.ravel())\n"
    "feature_oof = CFG.FEAT_W[0] * feature_art['FM_PL_mean'] + CFG.FEAT_W[1] * feature_art['SVR_plateau_PL']\n"
    "se_feature = ((feature_oof - y) ** 2).mean(0)\n"
    "tail = y > 70\n"
    "\n"
    "fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))\n"
    "grid = np.linspace(y.min() - 3, y.max() + 3, 400)\n"
    "ax[0].hist(y, bins=40, density=True, color=CFG.C_TRAIN, alpha=0.6, edgecolor='white')\n"
    "for k in order_g:\n"
    "    m, s, w = gmm.means_[k, 0], np.sqrt(gmm.covariances_[k].ravel()[0]), gmm.weights_[k]\n"
    "    ax[0].plot(grid, w * stats.norm.pdf(grid, m, s), lw=2, label=f'mode {m:.1f} (bobot {w:.2f})')\n"
    "ax[0].set(title='Histogram target dan dua komponen GMM', xlabel='target', ylabel='densitas')\n"
    "ax[0].legend()\n"
    "sns.violinplot(x=y, ax=ax[1], color=CFG.C_TRAIN, inner='quartile')\n"
    "ax[1].set(title='Sebaran target', xlabel='target')\n"
    "ax[2].plot(np.sort(y), np.arange(1, len(y) + 1) / len(y), color=CFG.C_TRAIN, lw=2)\n"
    "ax[2].axvline(70, color=CFG.C_ACCENT, ls='--', label='target = 70')\n"
    "ax[2].set(title='ECDF target', xlabel='target', ylabel='proporsi kumulatif')\n"
    "ax[2].legend()\n"
    "fig.suptitle('Distribusi Target', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '01_target_distribution.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "print(f'mean {y.mean():.2f} | median {np.median(y):.2f} | std {y.std():.2f} | skew {stats.skew(y):.2f} | '\n"
    "      f'min {y.min():.2f} | max {y.max():.2f}')\n"
    "print(f'baris dengan target > 70     : {tail.sum()} ({tail.mean():.1%})')\n"
    "print(f'porsi squared error model fitur dari baris > 70: {se_feature[tail].sum() / se_feature.sum():.1%}')\n"
    "print('mode GMM:', ', '.join(f'{gmm.means_[k, 0]:.1f} (bobot {gmm.weights_[k]:.2f})' for k in order_g))"
))

cells.append(md("> ### Insights\n>\n> Target bimodal dengan dua mode yang jelas (lihat keluaran GMM) dan batas bawah tajam di 32,6, sedangkan ekor kanannya memanjang sampai 100,4 (*skewness* 0,75). Bentuk ini tidak cocok dengan asumsi galat simetris, sehingga SVR memakai target log(y - 30) agar ekor kanan tidak mendominasi *kernel*.\n>\n> Hanya 26 baris (6,8%) yang bernilai di atas 70, namun baris ini menyumbang 40,1% *squared error* model fitur terbaik. Artinya, perbaikan RMSE paling besar hanya bisa datang dari baris ekor kanan, dan setiap evaluasi selanjutnya dipecah per posisi untuk melihat apakah ekor tersebut benar-benar membaik."))

cells.append(md("---\n\n## Distribusi Fitur: *Train* vs *Test*"))

cells.append(md(
    "Jika distribusi *test* berbeda dari *train*, validasi silang acak akan menyesatkan. Setiap fitur "
    "dibandingkan dengan uji Kolmogorov-Smirnov dua sampel, dan *adversarial validation* menguji apakah "
    "sebuah *classifier* mampu membedakan baris *train* dari *test* menggunakan seluruh fitur sekaligus."
))

cells.append(code(
    "ks = pd.DataFrame([stats.ks_2samp(X[:, j], X_te[:, j]) for j in range(len(FEATURES))],\n"
    "                  index=FEATURES)[['statistic', 'pvalue']]\n"
    "ks['shift_sd'] = (X_te.mean(0) - X.mean(0)) / X_all.std(0)\n"
    "\n"
    "fig, axes = plt.subplots(4, 6, figsize=(20, 11))\n"
    "for j, ax in enumerate(axes.ravel()):\n"
    "    sns.kdeplot(X[:, j], ax=ax, color=CFG.C_TRAIN, fill=True, alpha=0.3, label='train')\n"
    "    sns.kdeplot(X_te[:, j], ax=ax, color=CFG.C_TEST, fill=True, alpha=0.3, label='test')\n"
    "    flag = ' *' if ks['pvalue'].iloc[j] < 0.05 else ''\n"
    "    ax.set_title(f'{FEATURES[j]} (p={ks[\"pvalue\"].iloc[j]:.2f}){flag}', fontsize=9)\n"
    "    ax.set(xlabel='', ylabel='')\n"
    "axes[0, 0].legend(fontsize=8)\n"
    "fig.suptitle('Distribusi Fitur Train vs Test (* = KS p < 0.05)', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '02_feature_distributions.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "is_test = np.r_[np.zeros(len(X)), np.ones(len(X_te))]\n"
    "adv_auc = []\n"
    "for seed in range(5):\n"
    "    proba = cross_val_predict(RandomForestClassifier(300, min_samples_leaf=3, random_state=seed, n_jobs=-1),\n"
    "                              X_all, is_test, cv=StratifiedKFold(5, shuffle=True, random_state=seed),\n"
    "                              method='predict_proba')[:, 1]\n"
    "    adv_auc.append(roc_auc_score(is_test, proba))\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 4))\n"
    "colors = [CFG.C_ACCENT if p < 0.05 else CFG.C_GREY for p in ks['pvalue']]\n"
    "ax.bar(range(len(FEATURES)), ks['shift_sd'], color=colors)\n"
    "ax.axhline(0, color='black', lw=0.8)\n"
    "ax.set_xticks(range(len(FEATURES)), [f[-2:] for f in FEATURES])\n"
    "ax.set(title=f'Pergeseran rata-rata test terhadap train (satuan SD); merah = KS p < 0.05 | '\n"
    "             f'adversarial AUC = {np.mean(adv_auc):.3f}', xlabel='fitur', ylabel='selisih rata-rata / SD')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '03_train_test_shift.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "print(f'fitur dengan KS p < 0.05 : {(ks.pvalue < 0.05).sum()} dari {len(FEATURES)}')\n"
    "print(f'pergeseran terbesar      : {ks.shift_sd.abs().max():.2f} SD ({ks.shift_sd.abs().idxmax()})')\n"
    "print(f'adversarial AUC (5 seed) : {np.mean(adv_auc):.3f} (rentang {min(adv_auc):.3f} - {max(adv_auc):.3f})')"
))

cells.append(md("> ### Insights\n>\n> Sembilan dari 24 fitur lolos uji KS pada p < 0,05, tetapi pergeserannya kecil: maksimum 0,31 SD pada `feature_02`. Dengan 24 uji sekaligus, beberapa hasil signifikan memang diharapkan muncul secara kebetulan.\n>\n> *Adversarial validation* lebih meyakinkan: *Random Forest* hanya mencapai AUC 0,543 (rentang 0,525 sampai 0,573 untuk 5 *seed*), nyaris sama dengan tebakan acak 0,5. **Keputusan:** *train* dan *test* berasal dari distribusi fitur yang sama, sehingga *random K-fold* representatif untuk *test* dan tidak diperlukan *reweighting*."))

cells.append(md("---\n\n## Korelasi Antarfitur dan Struktur PCA"))

cells.append(md(
    "Panitia menyebut adanya proyeksi linear privat, sehingga fitur kemungkinan merupakan kombinasi dari "
    "sedikit sinyal laten. Matriks korelasi yang diurutkan dengan *hierarchical clustering* memperlihatkan "
    "kelompok fitur, spektrum PCA dibandingkan dengan batas *noise* Marchenko-Pastur, dan RMSE regresi "
    "linear pada k komponen pertama menunjukkan di komponen mana sinyal target berada."
))

cells.append(code(
    "corr = train[FEATURES].corr()\n"
    "leaf = leaves_list(linkage(pdist(1 - corr.abs().to_numpy()), method='average'))\n"
    "corr_ord = corr.iloc[leaf, leaf]\n"
    "upper = corr.abs().to_numpy()[np.triu_indices(len(FEATURES), 1)]\n"
    "\n"
    "Z_all_std = StandardScaler().fit_transform(X_all)\n"
    "pca_full = PCA().fit(Z_all_std)\n"
    "eig = pca_full.explained_variance_\n"
    "mp_edge = (1 + np.sqrt(len(FEATURES) / len(X_all))) ** 2\n"
    "\n"
    "pcs_tr = pca_full.transform(Z_all_std)[:len(X)]\n"
    "lin_rmse = []\n"
    "for k in range(1, len(FEATURES) + 1):\n"
    "    p = cross_val_predict(Ridge(alpha=1e-3), pcs_tr[:, :k], y, cv=KFold(5, shuffle=True, random_state=CFG.SEED))\n"
    "    lin_rmse.append(np.sqrt(np.mean((p - y) ** 2)))\n"
    "\n"
    "fig = plt.figure(figsize=(19, 5.5))\n"
    "gs = GridSpec(1, 3, width_ratios=[1.25, 1, 1], figure=fig)\n"
    "ax0, ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])\n"
    "sns.heatmap(corr_ord, cmap='vlag', center=0, vmin=-1, vmax=1, ax=ax0, cbar_kws={'shrink': 0.8},\n"
    "            xticklabels=[f[-2:] for f in corr_ord.columns], yticklabels=[f[-2:] for f in corr_ord.index])\n"
    "ax0.set_title('Korelasi Pearson (diurutkan per klaster)')\n"
    "ax1.bar(range(1, len(eig) + 1), eig, color=CFG.C_TRAIN)\n"
    "ax1.axhline(mp_edge, color=CFG.C_ACCENT, ls='--', label=f'batas Marchenko-Pastur {mp_edge:.2f}')\n"
    "ax1b = ax1.twinx()\n"
    "ax1b.plot(range(1, len(eig) + 1), np.cumsum(pca_full.explained_variance_ratio_), color='black', marker='o', ms=3)\n"
    "ax1b.set_ylabel('varians kumulatif')\n"
    "ax1b.grid(False)\n"
    "ax1.set(title='Spektrum PCA', xlabel='komponen', ylabel='eigenvalue')\n"
    "ax1.legend(loc='center right')\n"
    "ax2.plot(range(1, len(FEATURES) + 1), lin_rmse, marker='o', color=CFG.C_TRAIN)\n"
    "ax2.axhline(y.std(), color=CFG.C_GREY, ls=':', label='prediksi rata-rata')\n"
    "ax2.set(title='RMSE regresi linear pada k PC pertama', xlabel='k', ylabel='RMSE (5-fold)')\n"
    "ax2.legend()\n"
    "fig.suptitle('Struktur Antarfitur', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '04_correlation_pca.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "print(f'pasangan |r| > 0.7 : {(upper > 0.7).sum()} | |r| maksimum {upper.max():.2f}')\n"
    "print(f'eigenvalue di atas batas noise : {(eig > mp_edge).sum()} | varians 4 PC pertama {pca_full.explained_variance_ratio_[:4].sum():.1%}')\n"
    "print(f'RMSE linear: 4 PC {lin_rmse[3]:.2f} | 12 PC {lin_rmse[11]:.2f} | 24 PC {lin_rmse[-1]:.2f} | rata-rata {y.std():.2f}')"
))

cells.append(md("> ### Insights\n>\n> Fitur sangat kolinear: 43 pasang memiliki |r| > 0,7 dengan maksimum 0,93, dan hanya empat *eigenvalue* yang berada di atas batas *noise* Marchenko-Pastur. Keempatnya menjelaskan 91,3% varians, sesuai dengan dugaan bahwa 24 fitur adalah proyeksi linear dari sekitar empat sinyal laten ditambah perturbasi.\n>\n> Namun empat komponen terbesar itu hampir tidak membawa informasi target: regresi linear pada 4 PC mencapai RMSE 12,07, hampir sama dengan memprediksi rata-rata (12,89). RMSE baru turun ke 10,68 ketika seluruh 24 PC dipakai. Sinyal target justru tersimpan di komponen bervarians kecil. **Keputusan:** tidak ada reduksi dimensi agresif; SVR memakai *view* PCA 10 sampai 20 dimensi yang di-*whitening* agar komponen kecil berbobot setara."))

cells.append(md("---\n\n## Hubungan Fitur dengan Target"))

cells.append(md(
    "Korelasi Spearman merangkum hubungan monoton tiap fitur dengan target, sedangkan kurva LOWESS "
    "memperlihatkan bentuk hubungan yang tidak monoton. Selisih R-kuadrat LOWESS dan linear dipakai sebagai "
    "ukuran sederhana seberapa nonlinear hubungan sebuah fitur."
))

cells.append(code(
    "rel = []\n"
    "for j, f in enumerate(FEATURES):\n"
    "    fit = lowess(y, X[:, j], frac=0.4, return_sorted=False)\n"
    "    r_lin = np.corrcoef(X[:, j], y)[0, 1] ** 2\n"
    "    r_low = 1 - np.mean((y - fit) ** 2) / y.var()\n"
    "    rel.append((f, stats.spearmanr(X[:, j], y).statistic, r_lin, r_low))\n"
    "rel = pd.DataFrame(rel, columns=['feature', 'spearman', 'r2_linear', 'r2_lowess']).set_index('feature')\n"
    "rel['nonlinear_gain'] = rel['r2_lowess'] - rel['r2_linear']\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 4))\n"
    "srt = rel['spearman'].sort_values()\n"
    "ax.barh([f[-2:] for f in srt.index], srt, color=[CFG.C_ACCENT if v < 0 else CFG.C_TEST for v in srt])\n"
    "ax.set(title='Korelasi Spearman fitur dengan target', xlabel='Spearman rho', ylabel='fitur')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '05_spearman.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "top = rel['r2_lowess'].sort_values(ascending=False).index[:8]\n"
    "fig, axes = plt.subplots(2, 4, figsize=(19, 8))\n"
    "for ax, f in zip(axes.ravel(), top):\n"
    "    j = FEATURES.index(f)\n"
    "    sm = lowess(y, X[:, j], frac=0.4)\n"
    "    ax.scatter(X[:, j], y, s=10, alpha=0.45, c=y, cmap='flare')\n"
    "    ax.plot(sm[:, 0], sm[:, 1], color='black', lw=2)\n"
    "    ax.set_title(f'{f}: R2 linear {rel.loc[f, \"r2_linear\"]:.2f} | LOWESS {rel.loc[f, \"r2_lowess\"]:.2f}', fontsize=10)\n"
    "fig.suptitle('Delapan fitur paling informatif (kurva LOWESS)', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '06_feature_target.png', dpi=110, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "display(rel.sort_values('r2_lowess', ascending=False).head(10).round(3))"
))

cells.append(md("> ### Insights\n>\n> Tidak ada fitur yang kuat secara individual. Korelasi Spearman tertinggi hanya 0,42 (`feature_09`) dan R-kuadrat LOWESS tertinggi 0,22 (`feature_20`). Target merupakan komposit yang tersebar di banyak fitur.\n>\n> Beberapa fitur jelas nonlinear. `feature_20`, `feature_06`, `feature_14`, dan `feature_24` memperoleh tambahan R-kuadrat 0,10 sampai 0,16 dari kurva LOWESS dibanding garis lurus. `feature_06` bahkan tidak berkorelasi linear sama sekali (R-kuadrat 0,000) tetapi memiliki R-kuadrat LOWESS 0,14, dengan pola U terbalik. **Keputusan:** model harus nonlinear dan memakai seluruh fitur bersama-sama, yaitu *kernel* dan *foundation model*, bukan seleksi fitur."))

cells.append(md("---\n\n## Kedekatan Antarbaris dan Duplikat Dekat"))

cells.append(md(
    "Duplikat persis tidak ditemukan, namun baris yang sangat mirip tetap dapat membuat validasi terlalu "
    "optimis apabila satu baris berada di *fold train* dan pasangannya di *fold* validasi. Jarak tetangga "
    "terdekat dihitung pada fitur terstandardisasi, lalu dibandingkan antara *train* ke *train* dan *test* "
    "ke *train*. Pasangan terdekat diperiksa apakah targetnya juga mirip."
))

cells.append(code(
    "Z_tr, Z_te = Z_all_std[:len(X)], Z_all_std[len(X):]\n"
    "D_tt = cdist(Z_tr, Z_tr)\n"
    "np.fill_diagonal(D_tt, np.inf)\n"
    "nn_tt, nn_et = D_tt.min(1), cdist(Z_te, Z_tr).min(1)\n"
    "\n"
    "iu = np.triu_indices(len(X), 1)\n"
    "pair_d, pair_dy = D_tt[iu], np.abs(y[iu[0]] - y[iu[1]])\n"
    "closest = np.argsort(pair_d)[:8]\n"
    "near_pairs = pd.DataFrame({\n"
    "    'record_a': train.record_id.to_numpy()[iu[0][closest]], 'record_b': train.record_id.to_numpy()[iu[1][closest]],\n"
    "    'feature_distance': pair_d[closest], 'abs_target_diff': pair_dy[closest],\n"
    "    'row_gap': np.abs(iu[0][closest] - iu[1][closest]),\n"
    "})\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))\n"
    "bins = np.linspace(0, max(nn_tt.max(), nn_et.max()), 35)\n"
    "ax[0].hist(nn_tt, bins=bins, alpha=0.6, density=True, color=CFG.C_TRAIN, label='train ke train')\n"
    "ax[0].hist(nn_et, bins=bins, alpha=0.6, density=True, color=CFG.C_TEST, label='test ke train')\n"
    "ax[0].set(title='Jarak tetangga terdekat (fitur terstandardisasi)', xlabel='jarak', ylabel='densitas')\n"
    "ax[0].legend()\n"
    "close = pair_d < np.quantile(pair_d, 0.02)\n"
    "ax[1].scatter(pair_d[close], pair_dy[close], s=8, alpha=0.5, color=CFG.C_TRAIN)\n"
    "ax[1].set(title='2% pasangan terdekat: jarak fitur vs selisih target', xlabel='jarak fitur', ylabel='|selisih target|')\n"
    "fig.suptitle('Kedekatan Antarbaris', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '07_proximity.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "print(f'median jarak NN: train ke train {np.median(nn_tt):.3f} | test ke train {np.median(nn_et):.3f}')\n"
    "print(f'jarak NN minimum: train ke train {nn_tt.min():.3f} | test ke train {nn_et.min():.3f}')\n"
    "print(f'korelasi Spearman jarak vs selisih target (2% terdekat): {stats.spearmanr(pair_d[close], pair_dy[close]).statistic:.3f}')\n"
    "display(near_pairs.round(3))"
))

cells.append(md("> ### Insights\n>\n> Baris *test* tidak lebih dekat ke *train* dibanding baris *train* satu sama lain (median jarak tetangga terdekat 1,242 vs 1,151). Dengan demikian *random K-fold* tidak memberi keuntungan kedekatan yang tidak dimiliki *test*.\n>\n> Kemiripan fitur tidak menjamin kemiripan target. Pasangan terdekat D0250/D0183 (jarak 0,203) berbeda 2,74 poin, sedangkan D0271/D0116 (jarak 0,331) berbeda 9,87 poin. Korelasi antara jarak dan selisih target pada 2% pasangan terdekat hanya 0,17. **Keputusan:** baris yang mirip tidak dihapus karena tidak terbukti sebagai label salah. Uji *grouped split* saat pengembangan menunjukkan keunggulan tahap akhir tetap bertahan ketika baris yang berdekatan dipaksa berada di sisi split yang sama."))

cells.append(md("---\n\n## Temuan Kunci: Urutan Baris Berkas"))

cells.append(md(
    "Pemeriksaan sebelumnya hanya menguji angka di dalam `record_id`, yang memang acak. Bagian ini "
    "menguji hal yang berbeda, yaitu **posisi baris di dalam berkas**. Untuk *train*, posisi langsung "
    "dibandingkan dengan target. Untuk *test*, target tidak tersedia sehingga urutannya diuji secara tidak "
    "langsung dengan dua cara: (1) apakah setiap fitur bergeser sepanjang berkas *test* dengan arah yang "
    "sama seperti pada *train*, dan (2) apakah posisi yang ditaksir dari tetangga fitur di *train* "
    "berkorelasi dengan posisi baris *test*. Penggunaan urutan ini telah dikonfirmasi panitia sebagai bagian "
    "dari EDA dan prapemrosesan peserta."
))

cells.append(code(
    "def unit_rank(n):\n"
    "    return (np.arange(n) + 0.5) / n\n"
    "\n"
    "\n"
    "def neighbor_signals(A, B, k=10, dim=12):\n"
    "    mu, sd = A.mean(0), A.std(0)\n"
    "    pc = PCA(n_components=dim, svd_solver='full').fit((A - mu) / sd)\n"
    "    ZA = pc.transform((A - mu) / sd) / np.sqrt(pc.explained_variance_)\n"
    "    ZB = pc.transform((B - mu) / sd) / np.sqrt(pc.explained_variance_)\n"
    "    DA = cdist(ZA, ZA)\n"
    "    np.fill_diagonal(DA, np.inf)\n"
    "    DB = cdist(ZB, ZA)\n"
    "    qa = unit_rank(len(A))\n"
    "\n"
    "    def local(D):\n"
    "        ix = np.argpartition(D, k - 1, axis=1)[:, :k]\n"
    "        weights = 1 / np.maximum(np.take_along_axis(D, ix, axis=1), 0.05) ** 2\n"
    "        return (weights * qa[ix]).sum(1) / weights.sum(1)\n"
    "\n"
    "    return local(DA), local(DB)\n"
    "\n"
    "\n"
    "id_num = train['record_id'].str[1:].astype(int).to_numpy()\n"
    "iso_curve = IsotonicRegression(increasing=False).fit_transform(unit_rank(len(y)), y)\n"
    "drift_tr = np.array([stats.spearmanr(pos_tr, X[:, j]).statistic for j in range(len(FEATURES))])\n"
    "drift_te = np.array([stats.spearmanr(pos_te, X_te[:, j]).statistic for j in range(len(FEATURES))])\n"
    "est_tr, est_te = neighbor_signals(X, X_te)\n"
    "\n"
    "rho_id   = stats.spearmanr(id_num, y).statistic\n"
    "rho_pos  = stats.spearmanr(pos_tr, y).statistic\n"
    "rho_drift = np.corrcoef(drift_tr, drift_te)[0, 1]\n"
    "rho_est_tr = stats.spearmanr(pos_tr, est_tr).statistic\n"
    "rho_est_te = stats.spearmanr(pos_te, est_te).statistic\n"
    "\n"
    "fig, ax = plt.subplots(2, 3, figsize=(19, 9))\n"
    "ax[0, 0].scatter(id_num, y, s=9, alpha=0.6, color=CFG.C_GREY)\n"
    "ax[0, 0].set(title=f'Nomor record_id vs target (rho = {rho_id:+.3f})', xlabel='nomor record_id', ylabel='target')\n"
    "ax[0, 1].scatter(pos_tr, y, s=9, alpha=0.6, color=CFG.C_TRAIN, label='target')\n"
    "ax[0, 1].plot(pos_tr, iso_curve, color=CFG.C_ACCENT, lw=2, label='kurva isotonic menurun')\n"
    "ax[0, 1].set(title=f'Posisi baris train vs target (rho = {rho_pos:+.3f})', xlabel='posisi di train.csv', ylabel='target')\n"
    "ax[0, 1].legend()\n"
    "ax[0, 2].scatter(pos_tr, y - iso_curve, s=9, alpha=0.6, color=CFG.C_TRAIN)\n"
    "ax[0, 2].plot(pos_tr, pd.Series(np.abs(y - iso_curve)).rolling(25, center=True, min_periods=5).mean(),\n"
    "              color='black', lw=2, label='rata-rata bergulir |deviasi|')\n"
    "ax[0, 2].set(title='Deviasi lokal target dari kurva isotonic', xlabel='posisi di train.csv', ylabel='target - kurva')\n"
    "ax[0, 2].legend()\n"
    "ax[1, 0].scatter(drift_tr, drift_te, s=30, color=CFG.C_TEST)\n"
    "for j in range(len(FEATURES)):\n"
    "    ax[1, 0].annotate(FEATURES[j][-2:], (drift_tr[j], drift_te[j]), fontsize=7)\n"
    "ax[1, 0].axline((0, 0), slope=1, color='black', ls='--', lw=1)\n"
    "ax[1, 0].set(title=f'Drift fitur sepanjang berkas (r = {rho_drift:.3f})', xlabel='Spearman(posisi, fitur) train',\n"
    "             ylabel='Spearman(posisi, fitur) test')\n"
    "ax[1, 1].scatter(pos_tr / len(y), est_tr, s=9, alpha=0.6, color=CFG.C_TRAIN)\n"
    "ax[1, 1].set(title=f'Train: posisi taksiran tetangga (rho = {rho_est_tr:.3f})', xlabel='posisi relatif sebenarnya',\n"
    "             ylabel='posisi taksiran dari 10 tetangga')\n"
    "ax[1, 2].scatter(pos_te / len(X_te), est_te, s=14, alpha=0.7, color=CFG.C_TEST)\n"
    "ax[1, 2].set(title=f'Test: posisi taksiran tetangga (rho = {rho_est_te:.3f})', xlabel='posisi relatif di test.csv',\n"
    "             ylabel='posisi taksiran dari tetangga train')\n"
    "fig.suptitle('Urutan Baris Berkas Membawa Informasi Target', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '08_row_order.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "steps_up = np.mean(np.diff(y) > 0)\n"
    "print(f'Spearman(nomor record_id, target)          : {rho_id:+.3f}')\n"
    "print(f'Spearman(posisi baris train, target)        : {rho_pos:+.3f}')\n"
    "print(f'proporsi baris berurutan yang targetnya naik: {steps_up:.1%}')\n"
    "print(f'RMSE kurva isotonic posisi (in-sample)      : {np.sqrt(np.mean((y - iso_curve) ** 2)):.3f}')\n"
    "print(f'korelasi drift fitur train vs test          : {rho_drift:.3f}')\n"
    "print(f'Spearman posisi vs taksiran tetangga, train : {rho_est_tr:.3f} | test {rho_est_te:.3f}')"
))

cells.append(md("> ### Insights\n>\n> Ini temuan paling penting dari seluruh EDA. Nomor `record_id` sama sekali tidak berkorelasi dengan target (rho +0,049), tetapi **posisi baris** di `train.csv` berkorelasi -0,981. Kurva isotonic dari posisi saja mencapai RMSE *in-sample* 2,356, jauh di bawah sekitar 7,2 yang dicapai model fitur terbaik setelah delapan iterasi.\n>\n> Urutan tersebut hanya kasar. Pada dua baris yang berurutan, target naik di 49,6% kasus, yaitu sama seperti acak. Data tampaknya diurutkan per blok, dan di dalam blok urutannya acak. Deviasi lokal paling besar terjadi pada sekitar 50 baris pertama (rata-rata |deviasi| sekitar 4 sampai 5 poin), tepat di ekor kanan yang menyumbang 40% galat.\n>\n> Untuk *test*, dua bukti tidak langsung menunjuk arah yang sama: arah *drift* fitur sepanjang berkas *test* berkorelasi 0,873 dengan arah *drift* pada *train*, dan posisi yang ditaksir dari tetangga fitur berkorelasi 0,736 dengan posisi baris *test*, setara dengan 0,785 pada *train*. **Keputusan:** urutan baris menjadi lapisan kedua solusi. Peta posisi ke target (v9) menangkap tren global, taksiran tetangga mengoreksi posisi (v10), dan model residual menjelaskan deviasi lokal yang tidak dapat dijelaskan oleh urutan (v11 sampai v13)."))

cells.append(md("---\n\n## Dashboard EDA"))

cells.append(md(
    "Dashboard berikut merangkum temuan EDA dalam satu halaman: indikator kualitas data di baris atas, "
    "lalu empat grafik yang menjadi dasar keputusan pemodelan."
))

cells.append(code(
    "def kpi(ax, value, label, color=CFG.C_TRAIN):\n"
    "    ax.axis('off')\n"
    "    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.9, transform=ax.transAxes, color=color, alpha=0.12))\n"
    "    ax.text(0.5, 0.6, value, ha='center', va='center', fontsize=22, fontweight='bold', color=color, transform=ax.transAxes)\n"
    "    ax.text(0.5, 0.22, label, ha='center', va='center', fontsize=10, transform=ax.transAxes)\n"
    "\n"
    "\n"
    "fig = plt.figure(figsize=(20, 11))\n"
    "gs = GridSpec(3, 4, height_ratios=[0.45, 1, 1], figure=fig)\n"
    "kpi(fig.add_subplot(gs[0, 0]), f'{len(train)} / {len(test)}', 'baris train / test')\n"
    "kpi(fig.add_subplot(gs[0, 1]), f'{int(quality.loc[\"nilai hilang (fitur)\", \"train\"]) + cross_dup}', 'nilai hilang + duplikat lintas')\n"
    "kpi(fig.add_subplot(gs[0, 2]), f'{np.mean(adv_auc):.2f}', 'adversarial AUC (0.5 = identik)', CFG.C_TEST)\n"
    "kpi(fig.add_subplot(gs[0, 3]), f'{rho_pos:+.3f}', 'Spearman posisi baris vs target', CFG.C_ACCENT)\n"
    "\n"
    "a = fig.add_subplot(gs[1, 0])\n"
    "a.hist(y, bins=40, color=CFG.C_TRAIN, alpha=0.7, edgecolor='white')\n"
    "a.axvline(70, color=CFG.C_ACCENT, ls='--')\n"
    "a.set(title=f'Target bimodal, {tail.mean():.0%} baris > 70', xlabel='target')\n"
    "a = fig.add_subplot(gs[1, 1])\n"
    "a.plot(range(1, len(FEATURES) + 1), lin_rmse, marker='o', color=CFG.C_TRAIN)\n"
    "a.set(title='Sinyal linear tersebar di banyak PC', xlabel='jumlah PC', ylabel='RMSE linear')\n"
    "a = fig.add_subplot(gs[1, 2:])\n"
    "a.scatter(pos_tr, y, s=8, alpha=0.6, color=CFG.C_TRAIN)\n"
    "a.plot(pos_tr, iso_curve, color=CFG.C_ACCENT, lw=2)\n"
    "a.set(title='Train tersusun menurun menurut target', xlabel='posisi baris di train.csv', ylabel='target')\n"
    "a = fig.add_subplot(gs[2, 0])\n"
    "a.bar(range(len(FEATURES)), ks['shift_sd'], color=[CFG.C_ACCENT if p < 0.05 else CFG.C_GREY for p in ks['pvalue']])\n"
    "a.set(title='Pergeseran train-test kecil', xlabel='fitur', ylabel='selisih / SD')\n"
    "a = fig.add_subplot(gs[2, 1])\n"
    "a.hist(nn_tt, bins=30, alpha=0.6, density=True, color=CFG.C_TRAIN, label='train-train')\n"
    "a.hist(nn_et, bins=30, alpha=0.6, density=True, color=CFG.C_TEST, label='test-train')\n"
    "a.set(title='Jarak tetangga terdekat setara', xlabel='jarak')\n"
    "a.legend()\n"
    "a = fig.add_subplot(gs[2, 2])\n"
    "a.scatter(drift_tr, drift_te, s=25, color=CFG.C_TEST)\n"
    "a.axline((0, 0), slope=1, color='black', ls='--', lw=1)\n"
    "a.set(title=f'Drift fitur train = test (r = {rho_drift:.2f})', xlabel='train', ylabel='test')\n"
    "a = fig.add_subplot(gs[2, 3])\n"
    "a.scatter(pos_te / len(X_te), est_te, s=12, color=CFG.C_TEST)\n"
    "a.set(title=f'Test juga terurut (rho = {rho_est_te:.2f})', xlabel='posisi di test.csv', ylabel='posisi taksiran')\n"
    "fig.suptitle('Dashboard EDA - FIK FAIR 2026', fontsize=16, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '09_eda_dashboard.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(md("> ### Insights\n>\n> Cerita data dapat diringkas dalam satu kalimat: fitur bersih dan stabil antara *train* dan *test* tetapi lemah secara individual, sedangkan informasi terkuat tentang target tersimpan pada urutan baris berkas.\n>\n> Ketiga lapisan solusi berasal langsung dari dashboard ini. Panel distribusi dan PCA membentuk lapisan fitur yang nonlinear dan memakai seluruh komponen, panel urutan membentuk lapisan urutan, dan deviasi lokal di bagian atas berkas membentuk lapisan koreksi."))


# ---------------------------------------------------------------------------
# Section 4: Prapemrosesan dan Rekayasa Fitur
# ---------------------------------------------------------------------------
cells.append(section("Prapemrosesan dan Rekayasa Fitur", "4"))

cells.append(md(
    "Setiap langkah prapemrosesan pada bagian ini berasal dari temuan EDA dan hanya di-*fit* pada "
    "partisi *train* di dalam setiap *fold*. Informasi yang dipakai hanya fitur, target *train*, dan "
    "posisi baris. Label *test* tidak pernah tersedia maupun ditebak."
))

cells.append(md("---\n\n## Posisi Relatif dan Peta Kuantil"))

cells.append(md(
    "Berkas *train* dan *test* memiliki panjang berbeda, sehingga posisi dinormalisasi menjadi posisi "
    "relatif (i + 0,5) / n. `curve` memetakan posisi relatif ke kurva isotonic menurun dari target *train*, "
    "dengan opsi penghalusan Gaussian. `beta_curve` adalah varian yang mengintegrasikan kurva terhadap "
    "distribusi Beta dari setiap *order statistic* berkas *query*, karena baris ke-j dari 96 baris *test* "
    "tidak tepat berada di persentil j/96 data penuh."
))

cells.append(code(
    "def curve(t, u, smooth=0):\n"
    "    q = unit_rank(len(t))\n"
    "    ys = IsotonicRegression(increasing=False).fit_transform(q, t)\n"
    "    if smooth:\n"
    "        ys = gaussian_filter1d(ys, smooth * len(t), mode='nearest')\n"
    "    return np.interp(u, q, ys)\n"
    "\n"
    "\n"
    "def beta_curve(t, m):\n"
    "    q = unit_rank(len(t))\n"
    "    ys = IsotonicRegression(increasing=False).fit_transform(q, t)\n"
    "    edges = np.r_[0, (q[1:] + q[:-1]) / 2, 1]\n"
    "    j = np.arange(1, m + 1)[:, None]\n"
    "    w = np.diff(betainc(j, m + 1 - j, edges[None, :]), axis=1)\n"
    "    return w @ ys\n"
    "\n"
    "\n"
    "u_test = unit_rank(len(X_te))\n"
    "fig, ax = plt.subplots(1, 2, figsize=(17, 4.5))\n"
    "ax[0].scatter(unit_rank(len(y)), y, s=8, alpha=0.4, color=CFG.C_GREY, label='target train')\n"
    "ax[0].plot(unit_rank(len(y)), curve(y, unit_rank(len(y))), color=CFG.C_TRAIN, lw=2, label='isotonic')\n"
    "ax[0].plot(unit_rank(len(y)), curve(y, unit_rank(len(y)), 0.015), color=CFG.C_ACCENT, lw=2, label='isotonic + smoothing 0.015')\n"
    "ax[0].set(title='Kurva target terhadap posisi relatif', xlabel='posisi relatif', ylabel='target')\n"
    "ax[0].legend()\n"
    "ax[1].plot(u_test, curve(y, u_test), color=CFG.C_TRAIN, lw=2, marker='o', ms=3, label='peta kuantil')\n"
    "ax[1].plot(u_test, beta_curve(y, len(X_te)), color=CFG.C_TEST, lw=2, label='integrasi Beta')\n"
    "ax[1].set(title='Peta posisi ke target untuk 96 baris test', xlabel='posisi relatif di test.csv', ylabel='prediksi')\n"
    "ax[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '10_quantile_map.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "print(f'peta kuantil test: maks {curve(y, u_test).max():.2f} | min {curve(y, u_test).min():.2f}')\n"
    "print(f'integrasi Beta   : maks {beta_curve(y, len(X_te)).max():.2f} | min {beta_curve(y, len(X_te)).min():.2f}')"
))

cells.append(md("> ### Insights\n>\n> Peta kuantil saja sudah merupakan prediktor yang sangat kuat. Untuk 96 baris *test*, prediksinya bergerak dari 93,40 di baris pertama sampai 33,34 di baris terakhir, mencakup ekor kanan yang tidak pernah dapat dijangkau model fitur. Peta ini menjadi separuh dari prediksi versi 9.\n>\n> Integrasi Beta lebih konservatif di ujung atas (90,88 dibanding 93,40) karena memperhitungkan bahwa posisi baris teratas *test* pada data penuh tidak pasti. Kedua varian dimasukkan ke katalog versi 10, dan *inner CV* yang menentukan mana yang lebih baik."))

cells.append(md("---\n\n## Koreksi Posisi dari Tetangga Fitur"))

cells.append(md(
    "Posisi relatif sebuah baris di dalam berkasnya sendiri hanya taksiran posisinya pada data penuh. "
    "Fitur dapat memperbaiki taksiran ini karena baris yang fiturnya mirip cenderung berada di posisi yang "
    "mirip (EDA: rho 0,785). `neighbor_warp` (versi 10) dan `warp` (versi 11 ke atas) menaksir posisi "
    "setiap baris *query* dari 10 tetangga terdekat di ruang PCA-12 *whitened*, mengurangi bias tren yang "
    "diukur secara *leave-one-out* pada *train*, menghaluskan koreksi sepanjang berkas, mengecilkannya di "
    "kedua ujung, dan menjaga hasilnya tetap monoton. Varian `bridge` memakai *prior* Gaussian dengan "
    "kovarians *Brownian bridge*, limit dari proses rank empiris.\n\n"
    "Grafik di bawah menguji ide ini pada satu split ilustrasi. Karena posisi sebenarnya baris validasi "
    "pada data *train* penuh diketahui, kesalahan taksiran posisi dapat diukur secara langsung."
))

cells.append(code(
    "def neighbor_warp(A, B, k=20, alpha=0.5):\n"
    "    mu, sd = A.mean(0), A.std(0)\n"
    "    pc = PCA(n_components=12).fit((A - mu) / sd)\n"
    "    ZA = pc.transform((A - mu) / sd) / np.sqrt(pc.explained_variance_)\n"
    "    ZB = pc.transform((B - mu) / sd) / np.sqrt(pc.explained_variance_)\n"
    "    DAA = cdist(ZA, ZA)\n"
    "    np.fill_diagonal(DAA, np.inf)\n"
    "    DBA = cdist(ZB, ZA)\n"
    "    qa, qb = unit_rank(len(A)), unit_rank(len(B))\n"
    "\n"
    "    def local(D):\n"
    "        idx = np.argpartition(D, k - 1, axis=1)[:, :k]\n"
    "        d = np.take_along_axis(D, idx, axis=1)\n"
    "        w = 1 / np.maximum(d, 0.05) ** 2\n"
    "        return np.sum(w * qa[idx], axis=1) / w.sum(1)\n"
    "\n"
    "    ref, query = local(DAA), local(DBA)\n"
    "    ref_trend = gaussian_filter1d(ref, max(1, 0.05 * len(A)), mode='nearest')\n"
    "    delta = query - np.interp(qb, qa, ref_trend)\n"
    "    delta = gaussian_filter1d(delta, max(1, 0.10 * len(B)), mode='nearest')\n"
    "    newq = np.clip(qb + alpha * 4 * qb * (1 - qb) * delta, 0, 1)\n"
    "    return IsotonicRegression(increasing=True).fit_transform(qb, newq)\n"
    "\n"
    "\n"
    "def warp(A, B, k=10, dim=12, kind='gaussian', strength=1.0):\n"
    "    ref, query = neighbor_signals(A, B, k, dim)\n"
    "    qa, qb = unit_rank(len(A)), unit_rank(len(B))\n"
    "    trend = gaussian_filter1d(ref, 0.05 * len(A), mode='nearest')\n"
    "    error = query - np.interp(qb, qa, trend)\n"
    "    if kind == 'bridge':\n"
    "        # Brownian bridge covariance: min(u, v) - u * v\n"
    "        K = (np.minimum.outer(qb, qb) - np.outer(qb, qb)) * (1 / len(A) + 1 / len(B)) * strength\n"
    "        derivative = np.clip(np.interp(qb, qa, np.gradient(trend, qa)), 0.1, 3.0)\n"
    "        sigma2 = max(float(np.mean((ref - trend) ** 2)), 1e-4)\n"
    "        cov = K * derivative[:, None] * derivative[None, :] + np.eye(len(B)) * sigma2\n"
    "        correction = (K * derivative[None, :]) @ np.linalg.solve(cov, error)\n"
    "    else:\n"
    "        correction = strength * 4 * qb * (1 - qb) * gaussian_filter1d(error, max(1, 0.1 * len(B)), mode='nearest')\n"
    "    return IsotonicRegression(increasing=True).fit_transform(qb, np.clip(qb + correction, 0, 1))\n"
    "\n"
    "\n"
    "demo_tr, demo_va = map(np.sort, next(KFold(5, shuffle=True, random_state=CFG.SEED).split(X)))\n"
    "u_true  = (demo_va + 0.5) / len(y)\n"
    "u_naive = unit_rank(len(demo_va))\n"
    "u_warp  = warp(X[demo_tr], X[demo_va])\n"
    "pred_naive, pred_warp = curve(y[demo_tr], u_naive), curve(y[demo_tr], u_warp)\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(17, 4.5))\n"
    "ax[0].plot(u_naive, u_naive - u_true, color=CFG.C_GREY, lw=2, label='posisi relatif di berkas validasi')\n"
    "ax[0].plot(u_naive, u_warp - u_true, color=CFG.C_ACCENT, lw=2, label='setelah koreksi tetangga')\n"
    "ax[0].axhline(0, color='black', lw=0.8)\n"
    "ax[0].set(title='Galat taksiran posisi pada split ilustrasi', xlabel='posisi relatif di berkas validasi',\n"
    "          ylabel='taksiran - posisi sebenarnya')\n"
    "ax[0].legend()\n"
    "ax[1].scatter(u_true, y[demo_va], s=12, color=CFG.C_GREY, label='target validasi')\n"
    "ax[1].plot(u_true, pred_naive, color=CFG.C_TRAIN, lw=2, label=f'peta kuantil (RMSE {rmse(y[demo_va], pred_naive):.2f})')\n"
    "ax[1].plot(u_true, pred_warp, color=CFG.C_ACCENT, lw=2, label=f'+ koreksi posisi (RMSE {rmse(y[demo_va], pred_warp):.2f})')\n"
    "ax[1].set(title='Dampaknya pada prediksi target', xlabel='posisi sebenarnya', ylabel='target')\n"
    "ax[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '11_rank_correction.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "print(f'RMSE posisi: tanpa koreksi {rmse(u_naive, u_true):.4f} | dengan koreksi {rmse(u_warp, u_true):.4f}')\n"
    "print(f'RMSE target: peta kuantil {rmse(y[demo_va], pred_naive):.3f} | + koreksi posisi {rmse(y[demo_va], pred_warp):.3f}')"
))

cells.append(md("> ### Insights\n>\n> Pada split ilustrasi, posisi relatif di berkas validasi meleset secara sistematis hingga +0,14 di bagian tengah. Galatnya tidak acak per baris, melainkan berkorelasi sepanjang berkas, sesuai pola *Brownian bridge* dari proses rank empiris: nol di kedua ujung dan terbesar di tengah. Pola inilah yang dapat dikoreksi oleh tren halus.\n>\n> Koreksi dari tetangga fitur menurunkan RMSE posisi dari 0,0822 ke 0,0666 dan RMSE target dari 5,28 ke 4,14 pada split ini. Satu split belum cukup sebagai bukti; validasi pada 15 *fold* dengan seleksi *nested* ada di bagian Pemodelan (versi 10)."))

cells.append(md("---\n\n## *Detrending* Sepanjang Berkas"))

cells.append(md(
    "Kurva posisi menangkap tren global, sedangkan penyimpangan lokal target dari kurva tersebut "
    "(EDA: sampai sekitar 8 poin di bagian atas berkas) mungkin dijelaskan oleh fitur. Untuk memisahkan "
    "kedua komponen, tren sepanjang berkas dikurangkan dari setiap fitur dan dari target, lalu residual "
    "target dimodelkan dari residual fitur. `local_linear` adalah *smoother* lokal-linear berbobot Gaussian. "
    "Dibandingkan *filter* Gaussian biasa, *smoother* ini tidak tertarik ke tengah di ujung berkas, "
    "tempat target paling tinggi berada."
))

cells.append(code(
    "def local_linear(z, h=0.05):\n"
    "    q = unit_rank(len(z))\n"
    "    dx = q[None, :] - q[:, None]\n"
    "    w = np.exp(-0.5 * (dx / h) ** 2)\n"
    "    s0, s1, s2 = w.sum(1), (w * dx).sum(1), (w * dx ** 2).sum(1)\n"
    "    S = w * (s2[:, None] - s1[:, None] * dx) / (s0 * s2 - s1 * s1)[:, None]\n"
    "    return S @ z\n"
    "\n"
    "\n"
    "Zs = (X - X.mean(0)) / X.std(0)\n"
    "trend_y_gauss, trend_y_ll = gaussian_filter1d(y, 0.05 * len(y), mode='nearest'), local_linear(y)\n"
    "show_feat = [FEATURES.index(f) for f in ['feature_09', 'feature_15']]\n"
    "\n"
    "fig, ax = plt.subplots(1, 3, figsize=(19, 4.5))\n"
    "ax[0].scatter(pos_tr, y, s=8, alpha=0.4, color=CFG.C_GREY)\n"
    "ax[0].plot(pos_tr, trend_y_gauss, color=CFG.C_TEST, lw=2, label='tren Gaussian')\n"
    "ax[0].plot(pos_tr, trend_y_ll, color=CFG.C_ACCENT, lw=2, label='tren local-linear')\n"
    "ax[0].set(title='Tren target sepanjang berkas', xlabel='posisi', ylabel='target', xlim=(-5, 120))\n"
    "ax[0].legend()\n"
    "for a, j in zip(ax[1:], show_feat):\n"
    "    tr_j = local_linear(Zs[:, j])\n"
    "    a.scatter(pos_tr, Zs[:, j], s=8, alpha=0.35, color=CFG.C_GREY, label='fitur')\n"
    "    a.plot(pos_tr, tr_j, color=CFG.C_ACCENT, lw=2, label='tren local-linear')\n"
    "    a.scatter(pos_tr, Zs[:, j] - tr_j - 4, s=8, alpha=0.35, color=CFG.C_TEST, label='residual (digeser -4)')\n"
    "    a.set(title=f'{FEATURES[j]}: sebelum dan sesudah detrending', xlabel='posisi', ylabel='nilai terstandardisasi')\n"
    "    a.legend(fontsize=8)\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '12_detrending.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "top10 = slice(0, 20)\n"
    "print(f'20 baris teratas: target rata-rata {y[top10].mean():.2f} | tren Gaussian {trend_y_gauss[top10].mean():.2f} | '\n"
    "      f'tren local-linear {trend_y_ll[top10].mean():.2f}')\n"
    "resid_corr = [abs(stats.spearmanr(pos_tr, Zs[:, j] - local_linear(Zs[:, j])).statistic) for j in range(len(FEATURES))]\n"
    "print(f'|Spearman(posisi, fitur)| rata-rata: sebelum {np.mean(np.abs(drift_tr)):.3f} | sesudah detrending {np.mean(resid_corr):.3f}')"
))

cells.append(md("> ### Insights\n>\n> Pada 20 baris teratas (target rata-rata 80,56), tren Gaussian memberi 83,04, bias +2,5 poin karena *padding* nilai tepi, sedangkan tren *local-linear* memberi 81,13, bias +0,6 poin. Karena galat terbesar ada di ujung atas berkas, pengurangan bias di ujung ini relevan langsung terhadap RMSE.\n>\n> Setelah *detrending*, rata-rata |Spearman| antara posisi dan fitur turun dari 0,230 ke 0,025. Residual fitur hampir bebas dari tren urutan, sehingga model residual memakai informasi yang berbeda dari peta posisi dan keduanya saling melengkapi, bukan mengulang sinyal yang sama."))

cells.append(md("---\n\n## Interaksi Residual dan Posisi"))

cells.append(md(
    "Efek sebuah fitur terhadap deviasi target belum tentu sama di bagian atas dan bawah berkas. Karena itu "
    "residual fitur R juga dikalikan dengan posisi tersentralisasi (2u - 1), menghasilkan matriks input "
    "[R, R x (2u - 1)] berukuran 48 kolom. Koefisien model residual dengan demikian dapat berubah secara "
    "linear sepanjang berkas. Pada versi 13, hubungan nonlinear antar-residual ditangkap oleh *Kernel Ridge* "
    "RBF dengan lebar *kernel* yang diskalakan terhadap jumlah input (gamma = g / 48)."
))

cells.append(md("---\n\n## Ringkasan Keputusan Prapemrosesan"))

cells.append(md(
    "Tabel berikut merangkum langkah yang dipakai beserta bukti pendukungnya, serta ide yang diuji dan "
    "ditolak. Ide yang ditolak tetap dicantumkan agar proses seleksi transparan.\n\n"
    "| Langkah | Dipakai pada | Bukti |\n"
    "|---|---|---|\n"
    "| Standardisasi + PCA *whitened* 10/12/14/16/20 dimensi | SVR | Sinyal target ada di PC kecil; rata-rata beberapa dimensi lebih stabil |\n"
    "| Target log(y - 30) terstandardisasi | SVR | SVR 8,83 -> 8,35 |\n"
    "| *Clipping* prediksi SVR ke rentang target *train* | SVR | Tanpa *clip* satu baris diprediksi 138 |\n"
    "| `norm_methods=['none']` | TabICLv2 | 7,634 -> 7,601 |\n"
    "| *Pseudo-labelling* transduktif | TabICLv2, TabPFN-3.5, SVR | TabICLv2 7,61 -> 7,52 (12 dari 15 *fold* membaik) |\n"
    "| Posisi relatif + peta kuantil | Lapisan urutan | RMSE CV 7,23 -> 3,61 |\n"
    "| Koreksi posisi dari tetangga PCA-12 | v10 | 3,613 -> 3,401 |\n"
    "| *Detrending* + residual Ridge | v11 | 3,401 -> 3,303 |\n"
    "| *Local-linear* + interaksi posisi | v12 | 3,303 -> 3,179 |\n"
    "| *Kernel Ridge* RBF pada residual | v13 | 3,179 -> 3,016 |\n\n"
    "| Ide yang ditolak | Hasil |\n"
    "|---|---|\n"
    "| *Quantile transform*, PCA-*denoise*, ICA | Tidak ada perbaikan; sumber ICA *heavy-tailed* |\n"
    "| Target log untuk *foundation model* | Lebih buruk (7,45 -> 7,63) |\n"
    "| *Fine-tuning* TabICLv2 | Lebih buruk (7,60 -> 7,69) |\n"
    "| *Test-time jitter*, *subagging context* | Tidak membaik atau memburuk |\n"
    "| Bobot sampel ekor, *classifier* ekor, model dua mode | Tidak membaik |\n"
    "| *Clipping* fitur, *detrending* lebih lebar (v13) | Memburuk (3,517 -> 3,518; 3,544) |"
))

# ---------------------------------------------------------------------------
# Section 5: Skema Validasi
# ---------------------------------------------------------------------------
cells.append(section("Skema Validasi", "5"))

cells.append(md(
    "Dengan 382 baris dan *public leaderboard* sekitar 29 baris, keputusan model hanya dapat dipercaya bila "
    "skema validasinya meniru situasi *test* dengan setia dan tidak membocorkan label. Bagian ini "
    "mendefinisikan *fold* yang dipakai bersama oleh seluruh model dan tahap."
))

cells.append(md("---\n\n## *Repeated K-Fold* dengan Urutan Terjaga"))

cells.append(md(
    "Seluruh model memakai `RepeatedKFold` 5 *fold* x 3 *repeat* dengan *seed* 42, sehingga setiap "
    "perbandingan antarversi bersifat berpasangan. Indeks di setiap *fold* diurutkan, sehingga baris "
    "validasi membentuk berkas kecil yang **tetap tersusun dalam urutan aslinya**, persis seperti "
    "`test.csv` terhadap `train.csv`. Posisi relatif dihitung ulang di dalam setiap partisi."
))

cells.append(code(
    "FOLDS = list(RepeatedKFold(n_splits=CFG.N_SPLITS, n_repeats=CFG.N_REPEATS, random_state=CFG.SEED).split(X))\n"
    "\n"
    "\n"
    "def fold_rmse(p):\n"
    "    return np.array([rmse(y[vai], p[i // CFG.N_SPLITS, vai]) for i, (_, vai) in enumerate(FOLDS)])\n"
    "\n"
    "\n"
    "membership = np.zeros((CFG.N_REPEATS, len(y)))\n"
    "for i, (_, vai) in enumerate(FOLDS):\n"
    "    membership[i // CFG.N_SPLITS, vai] = i % CFG.N_SPLITS\n"
    "    assert np.all(np.diff(np.sort(vai)) > 0)\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(18, 3.8), gridspec_kw={'width_ratios': [2.2, 1]})\n"
    "sns.heatmap(membership, cmap=sns.color_palette('flare', CFG.N_SPLITS), cbar_kws={'label': 'fold validasi'},\n"
    "            ax=ax[0], xticklabels=50, yticklabels=[f'repeat {r + 1}' for r in range(CFG.N_REPEATS)])\n"
    "ax[0].set(title='Pembagian fold sepanjang posisi train.csv', xlabel='posisi baris')\n"
    "val_tops = [y[vai].max() for _, vai in FOLDS]\n"
    "ax[1].bar(range(1, len(FOLDS) + 1), [len(v) for _, v in FOLDS], color=CFG.C_TRAIN)\n"
    "ax[1].set(title='Ukuran fold validasi', xlabel='fold', ylabel='jumlah baris')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '13_folds.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "print(f'{len(FOLDS)} fold | ukuran validasi {min(len(v) for _, v in FOLDS)}-{max(len(v) for _, v in FOLDS)} baris | '\n"
    "      f'target maksimum per fold validasi {min(val_tops):.1f}-{max(val_tops):.1f}')"
))

cells.append(md("> ### Insights\n>\n> Kelima belas *fold* berukuran 76 sampai 77 baris dan tersebar merata di sepanjang berkas pada setiap *repeat*, sehingga setiap *fold* validasi menyerupai berkas *test* kecil yang terurut.\n>\n> Target maksimum di *fold* validasi bervariasi dari 81,8 sampai 100,4. *Fold* yang memuat baris ekor teratas akan memiliki RMSE jauh lebih tinggi, sehingga perbandingan antarversi selalu dilakukan berpasangan per *fold* dan sebarannya ikut dilaporkan, bukan hanya rata-ratanya."))

cells.append(md("---\n\n## Seleksi *Nested* dan Bobot Tetap"))

cells.append(md(
    "Setiap tahap koreksi memiliki katalog kandidat. Pemilihan kandidat dilakukan sebagai berikut:\n\n"
    "1. Untuk setiap *fold* luar, *inner* 5-*fold* CV **hanya pada baris *train* luar** memberi skor "
    "pada seluruh kandidat dan memilih yang terbaik (*seed inner* = offset + indeks *fold*).\n"
    "2. Kandidat terpilih memprediksi baris validasi luar. Hasil ini menjadi prediksi *out-of-fold* yang "
    "jujur karena label validasi tidak pernah dilihat saat pemilihan.\n"
    "3. Prediksi tahap = 0,5 x versi sebelumnya + 0,5 x kandidat terpilih. Bobot 50:50 ditetapkan "
    "sebelum evaluasi, bukan di-*fit*, karena bobot yang di-*fit* pada *fold* lain akan membawa informasi "
    "target *fold* yang sedang dievaluasi melalui peta kuantil.\n"
    "4. Untuk *test*, pemilihan diulang pada seluruh 382 baris dengan *seed* final, lalu kandidat terpilih "
    "memprediksi 96 baris *test*.\n\n"
    "Selain itu, pada saat pengembangan setiap tahap juga diuji dengan *bootstrap* berpasangan per "
    "*record*, ratusan blok *query* 96 baris yang dibagi 29/67 meniru *public*/*private*, *grouped split* "
    "untuk baris yang berdekatan, dan *learning curve*."
))

# ---------------------------------------------------------------------------
# Section 6: Pemodelan
# ---------------------------------------------------------------------------
cells.append(section("Pemodelan", "6"))

cells.append(md(
    "Pemodelan disusun mengikuti tiga lapisan pada Pendahuluan. Lapisan fitur menghasilkan prediksi dari "
    "24 fitur, lapisan urutan menggabungkannya dengan posisi baris, dan empat tahap koreksi memperbaiki "
    "hasilnya secara bertahap. Setiap lapisan dievaluasi pada 15 *fold* yang sama."
))

cells.append(md("---\n\n## Lapisan Fitur: *Foundation Model* dan SVR dengan *Pseudo-Labelling*"))

cells.append(md(
    "Dengan 382 baris, model yang dilatih dari nol hanya memiliki sedikit data untuk mempelajari *prior*. "
    "*Tabular foundation model* adalah *transformer* yang dilatih awal pada jutaan dataset sintetis dan "
    "memprediksi baris baru lewat *in-context learning*, dengan data *train* sebagai konteks. **TabICLv2** "
    "dipakai tanpa normalisasi *power* internal dan dirata-ratakan atas 4 *seed*. **TabPFN-3.5** adalah "
    "satu-satunya checkpoint Prior Labs yang kompetitif dari versi 2.5, 2.6, 3, dan 3.5. **SVR-plateau** "
    "adalah SVR RBF pada lima *view* PCA *whitened* dengan target log(y - 30), dirata-ratakan atas empat "
    "setting C dan skala yang membentuk dataran performa.\n\n"
    "**Pseudo-labelling transduktif.** Baris *query* (*fold* validasi, atau 96 baris *test*) diberi label "
    "awal oleh model, kemudian dibagi ke beberapa bagian. Setiap bagian diprediksi ulang dengan bagian lain "
    "yang sudah ber-*pseudo-label* ditambahkan ke konteks. Tidak ada baris yang melihat *pseudo-label*-nya "
    "sendiri, dan tidak ada label *test* yang dipakai. Cara ini memadatkan konteks di sekitar setiap "
    "*query*.\n\n"
    "Kode di bawah dijalankan penuh pada mode `FULL` (GPU T4 x2). Pada mode `INFERENCE`, prediksi "
    "*out-of-fold* yang sama dimuat dari artefak."
))

cells.append(code(
    "def log_target(model):\n"
    "    return TransformedTargetRegressor(model, func=lambda t: np.log(t - CFG.LOG_SHIFT),\n"
    "                                      inverse_func=lambda p: np.exp(p) + CFG.LOG_SHIFT, check_inverse=False)\n"
    "\n"
    "\n"
    "def z_target(model):\n"
    "    return TransformedTargetRegressor(model, transformer=StandardScaler())\n"
    "\n"
    "\n"
    "sc_all = StandardScaler().fit(X_all)\n"
    "Z_all = sc_all.transform(X_all)\n"
    "\n"
    "\n"
    "class MultiPCASVR:\n"
    "    def __init__(self, cells=CFG.SVR_CELLS, clip=True):\n"
    "        self.cells, self.clip = cells, clip\n"
    "        self.pcas = {k: PCA(k, random_state=CFG.SEED).fit(Z_all) for k in CFG.PCA_DIMS}\n"
    "        self.sds = {k: self.pcas[k].transform(Z_all).std(0) for k in CFG.PCA_DIMS}\n"
    "\n"
    "    def _view(self, k, A):\n"
    "        return self.pcas[k].transform(sc_all.transform(A)) / self.sds[k]\n"
    "\n"
    "    def fit(self, A, t):\n"
    "        self.lo, self.hi = (t.min(), t.max()) if self.clip else (-np.inf, np.inf)\n"
    "        self.models = {(k, scale, C): log_target(z_target(SVR(C=C, gamma=scale / k, epsilon=0.05))).fit(self._view(k, A), t)\n"
    "                       for k in CFG.PCA_DIMS for scale, C in self.cells}\n"
    "        return self\n"
    "\n"
    "    def predict(self, A):\n"
    "        return np.mean([np.clip(m.predict(self._view(k, A)), self.lo, self.hi)\n"
    "                        for (k, _, _), m in self.models.items()], 0)\n"
    "\n"
    "\n"
    "class PseudoLabelTabICL:\n"
    "    def __init__(self, device='cpu'):\n"
    "        self.device = device\n"
    "\n"
    "    def _bag(self, A, t, B):\n"
    "        return np.mean([TabICLRegressor(device=self.device, random_state=s, norm_methods=CFG.ICL_NORM).fit(A, t).predict(B)\n"
    "                        for s in CFG.ICL_SEEDS], 0)\n"
    "\n"
    "    def fit(self, A, t):\n"
    "        self.A, self.t = A, t\n"
    "        return self\n"
    "\n"
    "    def predict(self, B):\n"
    "        first = self._bag(self.A, self.t, B)\n"
    "        part = np.arange(len(B)) % CFG.PL_SPLIT\n"
    "        out = np.empty(len(B))\n"
    "        for k in range(CFG.PL_SPLIT):\n"
    "            q, c = part == k, part != k\n"
    "            out[q] = self._bag(np.vstack([self.A, B[c]]), np.r_[self.t, first[c]], B[q])\n"
    "        return out\n"
    "\n"
    "\n"
    "class PseudoLabelTabPFN35(PseudoLabelTabICL):\n"
    "    def _bag(self, A, t, B):\n"
    "        return TabPFNRegressor.create_default_for_version(\n"
    "            ModelVersion.V3_5, device=self.device, n_estimators=8, random_state=CFG.SEED).fit(A, t).predict(B)\n"
    "\n"
    "\n"
    "def svr_with_pseudo(A, t, B, labels_B):\n"
    "    preds = []\n"
    "    for n_split in CFG.SVR_PL_SPLITS:\n"
    "        part = np.arange(len(B)) % n_split\n"
    "        out = np.empty(len(B))\n"
    "        for k in range(n_split):\n"
    "            q, c = part == k, part != k\n"
    "            out[q] = MultiPCASVR().fit(np.vstack([A, B[c]]), np.r_[t, labels_B[c]]).predict(B[q])\n"
    "        preds.append(out)\n"
    "    return np.mean(preds, 0)\n"
    "\n"
    "\n"
    "def foundation_mean(A, t, B, devices):\n"
    "    icl = PseudoLabelTabICL(device=devices[0]).fit(A, t).predict(B)\n"
    "    pfn = PseudoLabelTabPFN35(device=devices[-1]).fit(A, t).predict(B)\n"
    "    return 0.5 * (icl + pfn)"
))

cells.append(code(
    "if CFG.RUN_MODE == 'FULL':\n"
    "    start = time.time()\n"
    "    devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())] or ['cpu']\n"
    "    fm_oof, svr_oof = np.zeros((CFG.N_REPEATS, len(y))), np.zeros((CFG.N_REPEATS, len(y)))\n"
    "    for i, (tri, vai) in enumerate(FOLDS):\n"
    "        r = i // CFG.N_SPLITS\n"
    "        fm_oof[r, vai] = foundation_mean(X[tri], y[tri], X[vai], devices)\n"
    "        svr_oof[r, vai] = svr_with_pseudo(X[tri], y[tri], X[vai], fm_oof[r, vai])\n"
    "        print(f'fold {i + 1}/{len(FOLDS)} selesai ({time.time() - start:.0f} s)')\n"
    "    fm_test = foundation_mean(X, y, X_te, devices)\n"
    "    feat_test = CFG.FEAT_W[0] * fm_test + CFG.FEAT_W[1] * svr_with_pseudo(X, y, X_te, fm_test)\n"
    "    np.savez(CFG.OUTPUT_DIR / 'feature_oof.npz', FM_PL_mean=fm_oof, SVR_plateau_PL=svr_oof)\n"
    "else:\n"
    "    fm_oof, svr_oof = feature_art['FM_PL_mean'], feature_art['SVR_plateau_PL']\n"
    "    feat_test = None\n"
    "\n"
    "feature_oof = CFG.FEAT_W[0] * fm_oof + CFG.FEAT_W[1] * svr_oof\n"
    "layer = {'FM-PL-mean': fm_oof, 'SVR-plateau-PL': svr_oof, 'Prediksi fitur (0.7/0.3)': feature_oof}\n"
    "for name, p in layer.items():\n"
    "    reps = [rmse(y, pp) for pp in p]\n"
    "    print(f'{name:26s} RMSE per repeat {np.round(reps, 3)} | rata-rata {np.mean(reps):.3f} | fold {fold_rmse(p).mean():.3f}')\n"
    "print(f'korelasi residual FM vs SVR: {np.corrcoef((y - fm_oof).ravel(), (y - svr_oof).ravel())[0, 1]:.3f}')"
))

cells.append(md(
    "Grafik berikut membandingkan lapisan fitur dengan model pembanding dari benchmark awal (RMSE pada "
    "prediksi *out-of-fold* yang dirata-ratakan atas 3 *repeat*), lalu memeriksa kalibrasinya per rentang "
    "target untuk melihat di mana model fitur gagal."
))

cells.append(code(
    "pooled = {c: rmse(y, benchmark[c]) for c in benchmark.columns if c != 'record_id'}\n"
    "pooled.update({name: rmse(y, p.mean(0)) for name, p in layer.items()})\n"
    "pooled['Prediksi rata-rata'] = float(y.std())\n"
    "pooled = pd.Series(pooled).sort_values()\n"
    "\n"
    "bins = pd.cut(y, [30, 40, 50, 60, 70, 101])\n"
    "calib = pd.DataFrame({'y': y, 'p': feature_oof.mean(0), 'bin': bins}).groupby('bin', observed=True).agg(\n"
    "    n=('y', 'size'), target=('y', 'mean'), prediksi=('p', 'mean'))\n"
    "calib['bias'] = calib['target'] - calib['prediksi']\n"
    "\n"
    "fig, ax = plt.subplots(1, 3, figsize=(20, 5))\n"
    "colors = [CFG.C_ACCENT if n in layer else CFG.C_GREY for n in pooled.index]\n"
    "ax[0].barh(pooled.index, pooled.values, color=colors)\n"
    "ax[0].invert_yaxis()\n"
    "ax[0].set(title='RMSE out-of-fold (merah = lapisan fitur)', xlabel='RMSE', xlim=(6, 13.5))\n"
    "for i, v in enumerate(pooled.values):\n"
    "    ax[0].text(v + 0.05, i, f'{v:.2f}', va='center', fontsize=9)\n"
    "ax[1].scatter(y, feature_oof.mean(0), s=10, alpha=0.6, c=y, cmap='flare')\n"
    "ax[1].plot([30, 101], [30, 101], color='black', ls='--', lw=1)\n"
    "ax[1].set(title='Prediksi fitur vs target', xlabel='target', ylabel='prediksi out-of-fold')\n"
    "ax[2].bar(calib.index.astype(str), calib['bias'], color=CFG.C_TRAIN)\n"
    "ax[2].axhline(0, color='black', lw=0.8)\n"
    "ax[2].set(title='Bias (target - prediksi) per rentang target', xlabel='rentang target', ylabel='bias')\n"
    "fig.suptitle('Lapisan Fitur', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '14_feature_layer.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "display(calib.round(2))"
))

cells.append(md("> ### Insights\n>\n> Lapisan fitur adalah model terbaik yang dapat dibangun dari 24 fitur saja. Rata-rata *pseudo-labelled foundation model* (RMSE 7,480 per *repeat*) mengungguli SVR (7,901), dan gabungan 0,7/0,3 turun ke 7,353 karena korelasi residual keduanya hanya 0,850. Seluruh benchmark klasik tertinggal: *gradient boosting* sekitar 8,7 sampai 8,8 dan Ridge 10,7.\n>\n> Grafik kalibrasi memperlihatkan batas pendekatan berbasis fitur. Prediksi menyusut ke tengah: 26 baris dengan target di atas 70 (rata-rata 80,49) hanya diprediksi 64,89, bias +15,60 poin, sedangkan baris 30 sampai 40 diprediksi 3,51 poin terlalu tinggi. Selama delapan iterasi, tidak satu pun ide berbasis fitur mampu menutup bias ekor ini. Informasi yang hilang itu justru tersedia pada urutan baris."))

cells.append(md("---\n\n## Lapisan Urutan (Versi 9)"))

cells.append(md(
    "Untuk setiap berkas *query*, empat kandidat dibangun dari posisi relatif: peta kuantil, proyeksi "
    "isotonic dari prediksi fitur, prediksi fitur yang diurutkan ulang sesuai urutan berkas, dan prediksi "
    "fitur mentah. Tujuh aturan rata-rata ditetapkan sebelum evaluasi, tanpa bobot yang di-*fit*. Versi 9 "
    "di Kaggle memilih **rata-rata(peta kuantil, prediksi fitur terurut)**, yang dipertahankan di sini agar "
    "rantai prediksi identik dengan submisi."
))

cells.append(code(
    "def order_candidates(y_ref, f_query):\n"
    "    q = unit_rank(len(f_query))\n"
    "    quantile_map = curve(y_ref, q)\n"
    "    isotonic_features = IsotonicRegression(increasing=False).fit_transform(q, f_query)\n"
    "    sorted_features = np.sort(f_query)[::-1]\n"
    "    return np.c_[quantile_map, isotonic_features, sorted_features, f_query]\n"
    "\n"
    "\n"
    "ORDER_RULES = {\n"
    "    'peta kuantil'                    : [1, 0, 0, 0],\n"
    "    'isotonic fitur'                  : [0, 1, 0, 0],\n"
    "    'fitur terurut'                   : [0, 0, 1, 0],\n"
    "    'fitur mentah'                    : [0, 0, 0, 1],\n"
    "    'rata-rata(kuantil, isotonic)'    : [1 / 2, 1 / 2, 0, 0],\n"
    "    'rata-rata(kuantil, terurut)'     : [1 / 2, 0, 1 / 2, 0],\n"
    "    'rata-rata(kuantil, iso, mentah)' : [1 / 3, 1 / 3, 0, 1 / 3],\n"
    "}\n"
    "V9_RULE = 'rata-rata(kuantil, terurut)'\n"
    "\n"
    "order_oof = {k: np.zeros((CFG.N_REPEATS, len(y))) for k in ORDER_RULES}\n"
    "for i, (tri, vai) in enumerate(FOLDS):\n"
    "    tri, vai = np.sort(tri), np.sort(vai)\n"
    "    P = order_candidates(y[tri], feature_oof[i // CFG.N_SPLITS, vai])\n"
    "    for name, w in ORDER_RULES.items():\n"
    "        order_oof[name][i // CFG.N_SPLITS, vai] = P @ np.array(w, dtype=float)\n"
    "rule_tab = pd.DataFrame({name: fold_rmse(p) for name, p in order_oof.items()})\n"
    "v9_oof = order_oof[V9_RULE]\n"
    "\n"
    "if CFG.RUN_MODE == 'FULL':\n"
    "    v9_test = order_candidates(y, feat_test) @ np.array(ORDER_RULES[V9_RULE], dtype=float)\n"
    "else:\n"
    "    v9_test = test[['record_id']].merge(v9_file, on='record_id', validate='one_to_one')['prediction'].to_numpy()\n"
    "feat_sorted_test = 2 * v9_test - curve(y, u_test)\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(19, 4.8), gridspec_kw={'width_ratios': [1.3, 1]})\n"
    "rule_tab.plot.box(ax=ax[0], rot=15, color=CFG.C_TRAIN)\n"
    "ax[0].axhline(fold_rmse(feature_oof).mean(), color=CFG.C_GREY, ls=':', label='lapisan fitur saja')\n"
    "ax[0].set(title='Tujuh aturan urutan pada 15 fold', ylabel='RMSE fold')\n"
    "ax[0].legend()\n"
    "ax[1].plot(pos_te, curve(y, u_test), color=CFG.C_TRAIN, lw=2, label='peta kuantil')\n"
    "ax[1].plot(pos_te, feat_sorted_test, color=CFG.C_TEST, lw=2, label='prediksi fitur terurut')\n"
    "ax[1].plot(pos_te, v9_test, color=CFG.C_ACCENT, lw=2.5, label='v9 = rata-rata keduanya')\n"
    "ax[1].set(title='Prediksi test versi 9', xlabel='posisi di test.csv', ylabel='prediksi')\n"
    "ax[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '15_order_layer.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "print(rule_tab.mean().sort_values().round(4).to_string())\n"
    "print(f'\\naturan terbaik: {rule_tab.mean().idxmin()} | aturan v9: {V9_RULE} ({rule_tab[V9_RULE].mean():.4f})')\n"
    "print(f'prediksi fitur terurut di test monoton: {bool(np.all(np.diff(feat_sorted_test) <= 1e-9))}')"
))

cells.append(md("> ### Insights\n>\n> Urutan baris mengubah skala masalah. Hanya dengan mengurutkan ulang prediksi fitur sesuai urutan berkas, RMSE *fold* turun dari 7,322 ke 4,359, dan peta kuantil dari posisi saja mencapai 3,648. Aturan rata-rata(kuantil, terurut) yang dipakai versi 9 juga merupakan aturan terbaik dari tujuh aturan yang ditetapkan sebelumnya (3,6132), sama persis dengan angka validasi versi 9.\n>\n> Prediksi fitur terurut yang direkonstruksi dari berkas versi 9 monoton sempurna, sesuai cara versi 9 dibangun. Pada grafik kanan terlihat peran kedua komponen: peta kuantil memberi rentang yang benar sampai lebih dari 90, sedangkan prediksi fitur terurut meredam lonjakan tajam di baris-baris teratas."))

cells.append(md("---\n\n## Fungsi Tahap Koreksi dan *Runner Nested*"))

cells.append(md(
    "`predict` adalah inti versi 12 dan 13. Fungsi ini mengoreksi posisi *query* dengan `warp`, "
    "menghilangkan tren *local-linear* dari fitur dan target, membentuk residual fitur beserta interaksinya "
    "dengan posisi, lalu memprediksi residual target dengan Ridge atau *Kernel Ridge* dan menambahkannya ke "
    "kurva posisi. `gaussian_ridge` adalah versi 11 dengan tren Gaussian. Setiap tahap memiliki fungsi "
    "katalog kandidat, `select_rule` memilih kandidat dengan *inner CV*, dan `run_stage` menjalankan seleksi "
    "*nested* pada 15 *fold* serta prediksi *test*. Implementasi ini identik dengan kode pengembangan setiap "
    "versi. Pada pengembangan, setiap versi membaca versi sebelumnya dari berkas submisi CSV-nya, sehingga "
    "`as_submitted` meniru langkah tulis dan baca CSV tersebut agar hasil akhir identik per *byte*."
))

cells.append(code(
    "def gaussian_ridge(A, t, B, alpha):\n"
    "    qa, u = unit_rank(len(A)), warp(A, B)\n"
    "    mu, sd = A.mean(0), A.std(0)\n"
    "    Z, Q = (A - mu) / sd, (B - mu) / sd\n"
    "    tx = gaussian_filter1d(Z, 0.05 * len(A), axis=0, mode='nearest')\n"
    "    ty = gaussian_filter1d(t, 0.05 * len(A), mode='nearest')\n"
    "    RQ = Q - np.column_stack([np.interp(u, qa, tx[:, j]) for j in range(Z.shape[1])])\n"
    "    return curve(t, u, 0.005), Ridge(alpha=alpha).fit(Z - tx, t - ty).predict(RQ)\n"
    "\n"
    "\n"
    "def predict(A, t, B, config):\n"
    "    qa, u = unit_rank(len(A)), warp(A, B)\n"
    "    mu, sd = A.mean(0), np.maximum(A.std(0), 1e-12)\n"
    "    Z, Q = (A - mu) / sd, (B - mu) / sd\n"
    "    if config.get('clip'):\n"
    "        lo, hi = np.quantile(Z, [0.01, 0.99], axis=0)\n"
    "        Z, Q = np.clip(Z, lo, hi), np.clip(Q, lo, hi)\n"
    "    target = np.log(t) if config.get('log') else t\n"
    "    h = config.get('h', 0.05)\n"
    "    tx, ty = local_linear(Z, h), local_linear(target, h)\n"
    "    RX = Z - tx\n"
    "    RQ = Q - np.column_stack([np.interp(u, qa, tx[:, j]) for j in range(Z.shape[1])])\n"
    "    degree = config.get('degree', 1)\n"
    "    RA = np.column_stack([RX * (2 * qa[:, None] - 1) ** d for d in range(degree + 1)])\n"
    "    RB = np.column_stack([RQ * (2 * u[:, None] - 1) ** d for d in range(degree + 1)])\n"
    "    resid = target - ty\n"
    "    scale, query_scale = np.ones(len(t)), np.ones(len(B))\n"
    "    if config.get('scaled'):\n"
    "        scale = np.sqrt(np.maximum(local_linear(resid ** 2, 0.1), 1.0))\n"
    "        query_scale = np.interp(u, qa, scale)\n"
    "    if config.get('kernel'):\n"
    "        model = KernelRidge(alpha=config.get('kernel_alpha', 1), kernel='rbf',\n"
    "                            gamma=config.get('gamma_scale', 1) / RA.shape[1])\n"
    "    else:\n"
    "        model = Ridge(alpha=config.get('alpha', 10))\n"
    "    residual = model.fit(RA, resid / scale).predict(RB) * query_scale\n"
    "    if config.get('linear_curve'):\n"
    "        base = np.interp(u, qa, ty)\n"
    "    else:\n"
    "        base = curve(target, u, config.get('smooth', 0.005))\n"
    "    pred = base + residual\n"
    "    if config.get('log'):\n"
    "        pred = np.exp(pred)\n"
    "    return pred\n"
    "\n"
    "\n"
    "def stage10_candidates(A, t, B):\n"
    "    out = {'quantile': curve(t, unit_rank(len(B))), 'beta': beta_curve(t, len(B))}\n"
    "    for k in [10, 30]:\n"
    "        for alpha in [0.6, 1.0, 1.5]:\n"
    "            u = neighbor_warp(A, B, k, alpha)\n"
    "            for smooth in [0, 0.005]:\n"
    "                out[f'k{k}_a{alpha}_s{smooth}'] = curve(t, u, smooth)\n"
    "    return out\n"
    "\n"
    "\n"
    "def stage11_candidates(A, t, B):\n"
    "    dims = {d: curve(t, warp(A, B, dim=d), 0.005) for d in [8, 12, 16, 20]}\n"
    "    out = {'gaussian_dim8': dims[8], 'gaussian_dim12': dims[12],\n"
    "           'dimension_mean': np.mean(list(dims.values()), axis=0),\n"
    "           'bridge_0.5': curve(t, warp(A, B, kind='bridge', strength=0.5), 0.005)}\n"
    "    rng = np.random.default_rng(5100)\n"
    "    bags = []\n"
    "    for _ in range(16):\n"
    "        a = np.sort(rng.choice(len(A), int(0.9 * len(A)), replace=False))\n"
    "        bags.append(curve(t[a], warp(A[a], B), 0.005))\n"
    "    out['subbag16'] = np.mean(bags, axis=0)\n"
    "    for alpha in [10, 100]:\n"
    "        base, residual = gaussian_ridge(A, t, B, alpha)\n"
    "        for weight in [0.5, 1.0]:\n"
    "            out[f'ridge{alpha}_w{weight}'] = base + weight * residual\n"
    "    return out\n"
    "\n"
    "\n"
    "def stage12_candidates(A, t, B):\n"
    "    base, residual = gaussian_ridge(A, t, B, 10)\n"
    "    return {'gaussian_interaction0_ridge10': base + residual,\n"
    "            'local_linear_interaction1_ridge10': predict(A, t, B, {})}\n"
    "\n"
    "\n"
    "STAGE13_CONFIGS = {\n"
    "    'baseline': {}, 'narrow': {'h': 0.025}, 'wide': {'h': 0.10},\n"
    "    'ridge3': {'alpha': 3}, 'ridge30': {'alpha': 30}, 'quadratic_rank': {'degree': 2},\n"
    "    'log_target': {'log': True}, 'residual_scaled': {'scaled': True},\n"
    "    'kernel': {'kernel': True}, 'smooth_curve': {'smooth': 0.015},\n"
    "    'linear_target_curve': {'linear_curve': True}, 'clip_features': {'clip': True},\n"
    "}\n"
    "STAGE13_CONFIGS.update({f'kernel_a{alpha}_g{gamma}': {'kernel': True, 'kernel_alpha': alpha, 'gamma_scale': gamma}\n"
    "                        for alpha in [0.1, 0.3, 1] for gamma in [1, 2, 4] if (alpha, gamma) != (1, 1)})\n"
    "\n"
    "\n"
    "def stage13_candidates(A, t, B):\n"
    "    return {k: predict(A, t, B, c) for k, c in STAGE13_CONFIGS.items()}\n"
    "\n"
    "\n"
    "def select_rule(candidates, A, t, seed):\n"
    "    losses = {}\n"
    "    for a, b in KFold(5, shuffle=True, random_state=seed).split(A):\n"
    "        a, b = np.sort(a), np.sort(b)\n"
    "        for key, p in candidates(A[a], t[a], A[b]).items():\n"
    "            losses[key] = losses.get(key, 0.0) + np.sum((p - t[b]) ** 2)\n"
    "    return min(losses, key=losses.get)\n"
    "\n"
    "\n"
    "def as_submitted(pred):\n"
    "    return pd.read_csv(io.StringIO(pd.DataFrame({'prediction': pred}).to_csv(index=False)))['prediction'].to_numpy()\n"
    "\n"
    "\n"
    "STAGE_LOG, COMPONENT_OOF, COMPONENT_TEST, FOLD_CHOICES = [], {}, {}, {}\n"
    "\n"
    "\n"
    "def run_stage(name, candidates, base_oof, base_test):\n"
    "    inner_seed, final_seed, expected = CFG.STAGES[name]\n"
    "    start = time.time()\n"
    "    new_oof, chosen = np.zeros_like(base_oof), []\n"
    "    for i, (tri, vai) in enumerate(FOLDS):\n"
    "        tri, vai = np.sort(tri), np.sort(vai)\n"
    "        key = select_rule(candidates, X[tri], y[tri], inner_seed + i)\n"
    "        new_oof[i // CFG.N_SPLITS, vai] = candidates(X[tri], y[tri], X[vai])[key]\n"
    "        chosen.append(key)\n"
    "    stage_oof = 0.5 * (base_oof + new_oof)\n"
    "    key = select_rule(candidates, X, y, final_seed)\n"
    "    new_test = candidates(X, y, X_te)[key]\n"
    "    stage_test = 0.5 * (base_test + new_test)\n"
    "\n"
    "    base_f, comp_f, stage_f = fold_rmse(base_oof), fold_rmse(new_oof), fold_rmse(stage_oof)\n"
    "    COMPONENT_OOF[name], COMPONENT_TEST[name], FOLD_CHOICES[name] = new_oof, new_test, chosen\n"
    "    STAGE_LOG.append({'tahap': name, 'RMSE sebelumnya': base_f.mean(), 'RMSE komponen': comp_f.mean(),\n"
    "                      'RMSE tahap': stage_f.mean(), 'fold membaik': f'{(stage_f < base_f).sum()}/{len(FOLDS)}',\n"
    "                      'pilihan final': key, 'sesuai pengembangan': key == expected})\n"
    "    print(f'pilihan inner per fold : {pd.Series(chosen).value_counts().to_dict()}')\n"
    "    print(f'RMSE fold rata-rata    : sebelumnya {base_f.mean():.4f} | komponen {comp_f.mean():.4f} | '\n"
    "          f'tahap 50:50 {stage_f.mean():.4f}')\n"
    "    print(f'fold membaik           : {(stage_f < base_f).sum()}/{len(FOLDS)} | repeat membaik: '\n"
    "          f'{sum(rmse(y, s) < rmse(y, b) for s, b in zip(stage_oof, base_oof))}/{CFG.N_REPEATS}')\n"
    "    print(f'pilihan pada 382 baris : {key} (pengembangan: {expected}, sesuai: {key == expected})')\n"
    "    print(f'perubahan prediksi test: RMS {rmse(stage_test, base_test):.4f} | waktu {time.time() - start:.0f} s')\n"
    "    return stage_oof, stage_test"
))

cells.append(md("---\n\n## Versi 10: Koreksi Rank dari Tetangga Fitur"))

cells.append(md(
    "Katalog versi 10 berisi peta kuantil biasa, integrasi Beta, dan koreksi tetangga dengan 10 atau 30 "
    "tetangga, tiga kekuatan koreksi (0,6; 1,0; 1,5), serta dua tingkat penghalusan kurva."
))

cells.append(code("v10_oof, v10_test = run_stage('v10', stage10_candidates, v9_oof, v9_test)"))

cells.append(md("> ### Insights\n>\n> Koreksi rank menurunkan RMSE dari 3,6132 ke 3,4008, membaik di 11 dari 15 *fold* dan di ketiga *repeat*. Interval *bootstrap* perubahan RMSE [-0,356; -0,093] seluruhnya berada di bawah nol.\n>\n> Pilihan *inner* berbeda antar-*fold* (integrasi Beta di 4 *fold*, koreksi 10 tetangga di 9 *fold*), yang wajar untuk data sekecil ini. Pada 382 baris, seleksi memilih `k10_a1.0_s0.005`, sama dengan pengembangan. Prediksi *test* bergeser dengan RMS 0,72 dari versi 9."))

cells.append(md("---\n\n## Versi 11: Ridge pada Residual Fitur"))

cells.append(md(
    "Versi 11 menambahkan model residual. Tren Gaussian sepanjang berkas dikurangkan dari fitur dan target, "
    "lalu Ridge memprediksi deviasi target dari residual fitur. Katalog yang sama juga memuat alternatif "
    "stabilitas posisi: dimensi PCA berbeda, rata-ratanya, *prior Brownian bridge*, dan 16 *subsample* 90%."
))

cells.append(code("v11_oof, v11_test = run_stage('v11', stage11_candidates, v10_oof, as_submitted(v10_test))"))

cells.append(md("> ### Insights\n>\n> Residual Ridge menurunkan RMSE dari 3,4008 ke 3,3031, membaik di 12 dari 15 *fold* dan di ketiga *repeat*, dengan interval *bootstrap* [-0,178; -0,021]. Kandidat residual Ridge alpha 10 terpilih di 9 dari 15 *fold*, sedangkan alternatif stabilitas posisi jarang menang. Artinya, gain datang dari fitur yang menjelaskan deviasi lokal, bukan dari taksiran posisi yang lebih halus.\n>\n> Mulai tahap ini prediksi tidak lagi dipaksa monoton sepanjang berkas, sesuai temuan EDA bahwa target hanya terurut secara kasar dan naik di separuh pasangan baris yang berurutan."))

cells.append(md("---\n\n## Versi 12: *Local-Linear* dan Interaksi Posisi"))

cells.append(md(
    "Versi 12 membandingkan residual Gaussian versi 11 dengan residual *local-linear* yang memiliki bias "
    "lebih kecil di ujung berkas, ditambah interaksi residual x posisi. Pada pengembangan, pilihan ini "
    "diuji pada 100 split *query* 96 baris yang masing-masing dibagi 50 kali menjadi bagian 29/67 yang "
    "meniru *public* dan *private*, agar yang dinilai bukan hanya rata-rata galat tetapi juga ekor atasnya."
))

cells.append(code("v12_oof, v12_test = run_stage('v12', stage12_candidates, v11_oof, as_submitted(v11_test))"))

cells.append(md("> ### Insights\n>\n> Kandidat *local-linear* dengan interaksi posisi terpilih di **seluruh 15 *fold***. Ini sinyal paling konsisten di antara semua tahap. RMSE turun dari 3,3031 ke 3,1789, membaik di 14 dari 15 *fold*, dengan interval *bootstrap* [-0,188; -0,056].\n>\n> Gain ini sesuai dengan analisis *detrending*: tren *local-linear* hampir tidak bias di ujung atas berkas, tempat galat terbesar berada, dan interaksi memungkinkan efek fitur berubah sepanjang berkas."))

cells.append(md("---\n\n## Versi 13: *Kernel Ridge* pada Residual"))

cells.append(md(
    "Dua belas ablasi satu-perubahan-per-langkah diuji pada 120 blok *query*: lebar tren, penalti Ridge, "
    "interaksi kuadratik, target log, *scaling* varians residual, *clipping* fitur, dan model residual "
    "nonlinear. Hanya *Kernel Ridge* RBF yang memberi perbaikan jelas. Studi sensitivitas 30 setting (alpha x "
    "gamma) menemukan wilayah yang baik di sekitar alpha 0,1 sampai 0,3, sehingga delapan setting di sekitarnya "
    "ditambahkan ke katalog yang berisi 20 kandidat."
))

cells.append(code("v13_oof, v13_test = run_stage('v13', stage13_candidates, v12_oof, as_submitted(v12_test))"))

cells.append(md("> ### Insights\n>\n> *Kernel Ridge* memberi perbaikan terbesar di antara tahap koreksi: RMSE turun dari 3,1789 ke 3,0156, membaik di **15 dari 15 *fold*** dan di ketiga *repeat*, dengan interval *bootstrap* [-0,240; -0,072]. Komponen *kernel* sendiri sudah mencapai 3,0437 sebelum dirata-ratakan dengan versi 12.\n>\n> Varian *kernel* terpilih di 14 dari 15 *fold*. Varian yang sengaja memperburuk struktur (*clipping* fitur, tren lebih lebar) tidak pernah terpilih. Pada 382 baris, seleksi memilih alpha 0,1 dan gamma 2/48, sama dengan pengembangan, dan prediksi *test* bergeser dengan RMS 0,76 dari versi 12."))

cells.append(md("---\n\n## Ringkasan Validasi"))

cells.append(md(
    "Tabel dan grafik berikut merangkum seluruh lapisan pada 15 *fold* yang sama: sebaran RMSE per *fold*, "
    "perubahan berpasangan setiap *fold*, dan letak galat sepanjang berkas. Interval *bootstrap* dihitung "
    "dengan me-*resample* *record* (ketiga prediksi *repeat* dari satu *record* tetap bersama) sebanyak 2.000 kali."
))

cells.append(code(
    "VERSIONS = {'fitur': feature_oof, 'v9': v9_oof, 'v10': v10_oof, 'v11': v11_oof, 'v12': v12_oof, 'v13': v13_oof}\n"
    "stage_tab = pd.DataFrame(STAGE_LOG).set_index('tahap')\n"
    "display(stage_tab.round(4))\n"
    "\n"
    "boot_rng = np.random.default_rng(CFG.SEED)\n"
    "boot_idx = boot_rng.integers(0, len(y), size=(2000, len(y)))\n"
    "\n"
    "\n"
    "def boot_delta(new, old):\n"
    "    se_new, se_old = ((new - y) ** 2).mean(0), ((old - y) ** 2).mean(0)\n"
    "    d = np.sqrt(se_new[boot_idx].mean(1)) - np.sqrt(se_old[boot_idx].mean(1))\n"
    "    return np.percentile(d, [2.5, 97.5]), (d < 0).mean()\n"
    "\n"
    "\n"
    "names = list(VERSIONS)\n"
    "for prev, cur in zip(names[1:-1], names[2:]):\n"
    "    ci, share = boot_delta(VERSIONS[cur], VERSIONS[prev])\n"
    "    print(f'{prev:>4s} -> {cur:<4s}: interval 95% perubahan RMSE [{ci[0]:+.3f}, {ci[1]:+.3f}] | membaik di {share:.1%} bootstrap')\n"
    "\n"
    "fold_tab = pd.DataFrame({k: fold_rmse(v) for k, v in VERSIONS.items()})\n"
    "deciles = np.minimum((pos_tr * 10) // len(y), 9)\n"
    "dec_tab = pd.DataFrame({k: [rmse(y[deciles == d], v.mean(0)[deciles == d]) for d in range(10)]\n"
    "                        for k, v in VERSIONS.items()}, index=[f'{10 * d}-{10 * d + 10}%' for d in range(10)])\n"
    "\n"
    "fig, ax = plt.subplots(1, 3, figsize=(21, 5.2))\n"
    "sns.boxplot(data=fold_tab, ax=ax[0], palette=[CFG.C_GREY] + list(CFG.STAGE_COLORS.values()))\n"
    "ax[0].set(title='RMSE per fold untuk setiap lapisan', ylabel='RMSE fold')\n"
    "for i in range(len(FOLDS)):\n"
    "    ax[1].plot(names[1:], fold_tab.iloc[i, 1:], color=CFG.C_TRAIN, alpha=0.4, marker='o', ms=3)\n"
    "ax[1].plot(names[1:], fold_tab.iloc[:, 1:].mean(), color=CFG.C_ACCENT, lw=3, marker='o', label='rata-rata')\n"
    "ax[1].set(title='Perubahan berpasangan 15 fold', ylabel='RMSE fold')\n"
    "ax[1].legend()\n"
    "sns.heatmap(dec_tab.T, annot=True, fmt='.1f', cmap='flare', ax=ax[2], cbar_kws={'label': 'RMSE'})\n"
    "ax[2].set(title='RMSE per desil posisi di train.csv', xlabel='desil posisi (atas ke bawah)')\n"
    "fig.suptitle('Ringkasan Validasi Nested', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '16_validation_summary.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "print(fold_tab.mean().round(4).to_string())"
))

cells.append(md("> ### Insights\n>\n> Setiap tahap menurunkan RMSE rata-rata, dan interval *bootstrap* setiap transisi berada sepenuhnya di bawah nol. Dari lapisan fitur sampai versi 13, RMSE *fold* turun dari 7,32 ke 3,02. Sebaran per *fold* juga menyempit: *fold* terburuk turun dari sekitar 4,8 (v9) ke 4,2 (v13).\n>\n> *Heatmap* desil menunjukkan di mana perbaikan terjadi. Pada 10% baris teratas, RMSE turun dari 14,8 (fitur saja) ke 7,4 (v9) dan akhirnya 4,8 (v13), sedangkan bagian bawah berkas sudah berada di bawah 2 sejak v9. Desil teratas tetap menjadi sumber galat terbesar, sejalan dengan EDA: ekor kanan target dan deviasi lokal paling besar terletak di sana."))

# ---------------------------------------------------------------------------
# Section 7: Inferensi dan Submisi
# ---------------------------------------------------------------------------
cells.append(section("Inferensi dan Submisi", "7"))

cells.append(md(
    "Prediksi *test* sudah dihasilkan di setiap tahap oleh `run_stage`, menggunakan pilihan kandidat pada "
    "seluruh 382 baris *train*. Bagian ini menelusuri rantai prediksi tersebut, menulis berkas submisi, "
    "dan memverifikasinya terhadap berkas yang disubmit."
))

cells.append(md("---\n\n## Rantai Prediksi *Test*"))

cells.append(md(
    "Karena setiap tahap adalah rata-rata 50:50, prediksi akhir dapat diuraikan menjadi bobot tetap: 1/16 "
    "versi 9, 1/16 koreksi rank, 1/8 komponen versi 11, 1/4 komponen versi 12, dan 1/2 komponen *kernel*. "
    "Penguraian ini dihitung ulang di bawah sebagai pemeriksaan konsistensi."
))

cells.append(code(
    "chain = pd.DataFrame({'record_id': test['record_id'], 'posisi': pos_te, 'v9': v9_test,\n"
    "                      **{f'komponen {k}': COMPONENT_TEST[k] for k in COMPONENT_TEST}, 'v13': v13_test})\n"
    "weights = {'v9': 1 / 16, 'komponen v10': 1 / 16, 'komponen v11': 1 / 8, 'komponen v12': 1 / 4, 'komponen v13': 1 / 2}\n"
    "recomposed = sum(w * chain[c] for c, w in weights.items())\n"
    "print(f'selisih maksimum penguraian bobot: {np.abs(recomposed - v13_test).max():.2e}')\n"
    "display(chain.head(8).round(3))\n"
    "\n"
    "fig = plt.figure(figsize=(20, 5))\n"
    "gs = GridSpec(1, 3, width_ratios=[1.6, 1, 0.8], figure=fig)\n"
    "a = fig.add_subplot(gs[0])\n"
    "a.plot(pos_te, v9_test, color=CFG.STAGE_COLORS['v9'], lw=2, label='v9 (lapisan urutan)')\n"
    "a.plot(pos_te, COMPONENT_TEST['v13'], color=CFG.C_GREY, lw=1.2, alpha=0.8, label='komponen kernel v13')\n"
    "a.plot(pos_te, v13_test, color=CFG.STAGE_COLORS['v13'], lw=2.5, label='v13 (final)')\n"
    "a.set(title='Prediksi sepanjang test.csv', xlabel='posisi di test.csv', ylabel='prediksi')\n"
    "a.legend()\n"
    "a = fig.add_subplot(gs[1])\n"
    "a.hist(y, bins=30, density=True, alpha=0.5, color=CFG.C_TRAIN, label='target train')\n"
    "a.hist(v13_test, bins=20, density=True, alpha=0.6, color=CFG.C_TEST, label='prediksi test v13')\n"
    "a.set(title='Distribusi prediksi vs target train', xlabel='nilai')\n"
    "a.legend()\n"
    "a = fig.add_subplot(gs[2])\n"
    "a.pie(list(weights.values()), autopct='%1.1f%%', pctdistance=0.75, colors=list(CFG.STAGE_COLORS.values()),\n"
    "      startangle=90, wedgeprops={'edgecolor': 'white'})\n"
    "a.legend(list(weights), loc='center left', bbox_to_anchor=(0.95, 0.5), fontsize=9)\n"
    "a.set_title('Bobot efektif setiap komponen')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '17_test_chain.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "print(f'prediksi test: rata-rata {v13_test.mean():.2f} | rentang {v13_test.min():.2f} - {v13_test.max():.2f}')\n"
    "print(f'target train : rata-rata {y.mean():.2f} | rentang {y.min():.2f} - {y.max():.2f}')\n"
    "print(f'Spearman(posisi test, prediksi v13): {stats.spearmanr(pos_te, v13_test).statistic:.3f} | '\n"
    "      f'langkah naik antar-baris berurutan: {np.mean(np.diff(v13_test) > 0):.1%}')"
))

cells.append(md("> ### Insights\n>\n> Penguraian bobot mereproduksi prediksi akhir sampai galat pembulatan (sekitar 1e-14, berasal dari langkah baca-tulis CSV antarversi). Prediksi akhir benar-benar merupakan kombinasi tetap dari lima komponen, tanpa bobot tersembunyi.\n>\n> Prediksi *test* (rata-rata 50,27, rentang 32,86 sampai 95,79) menutupi hampir seluruh rentang target *train* (32,59 sampai 100,44), berbeda dengan model fitur yang menyusut ke tengah. Prediksi mengikuti urutan berkas (Spearman -0,994), tetapi naik di 36,8% pasangan baris berurutan. Koreksi residual mempertahankan variasi lokal yang dalam EDA terbukti ada pada target, alih-alih memaksa kurva monoton.\n>\n> Pada baris-baris teratas, komponen *kernel* berani menyimpang dari versi 9 (misalnya D0261: 68,60 menjadi 76,73), sedangkan bobot 50:50 meredam setiap penyimpangan menjadi separuhnya. Ini adalah bentuk *shrinkage* yang disengaja terhadap komponen paling fleksibel."))

cells.append(md("---\n\n## Pembuatan Berkas Submisi"))

cells.append(md(
    "Berkas submisi mengikuti urutan dan kolom `submission_template.csv`, dengan tepat satu prediksi untuk "
    "setiap `record_id`. Seluruh syarat format diperiksa dengan *assertion* sebelum berkas ditulis, dan "
    "*hash* SHA256 dicetak sebagai identitas berkas."
))

cells.append(code(
    "submission = template[['record_id']].merge(\n"
    "    pd.DataFrame({'record_id': test['record_id'], 'prediction': v13_test}),\n"
    "    on='record_id', how='left', validate='one_to_one')\n"
    "\n"
    "assert list(submission.columns) == ['record_id', 'prediction']\n"
    "assert len(submission) == len(template) == len(test)\n"
    "assert submission['record_id'].is_unique and set(submission['record_id']) == set(test['record_id'])\n"
    "assert np.isfinite(submission['prediction']).all()\n"
    "\n"
    "csv_text = submission.to_csv(index=False)\n"
    "CFG.SUBMISSION_CSV.write_text(csv_text)\n"
    "submission_sha = hashlib.sha256(CFG.SUBMISSION_CSV.read_bytes()).hexdigest()\n"
    "print(f'berkas : {CFG.SUBMISSION_CSV}')\n"
    "print(f'SHA256 : {submission_sha}')\n"
    "display(submission.head())"
))

cells.append(md("---\n\n## Verifikasi terhadap Submisi"))

cells.append(md(
    "Berkas yang baru ditulis dibandingkan dengan `submission_v13.csv` yang disubmit ke Kaggle, baik per "
    "*byte* maupun per nilai. Pada mode `INFERENCE`, notebook berhenti dengan *error* bila selisihnya "
    "melebihi 1e-6, sehingga reproduksi tidak dapat lolos secara diam-diam. Versi pustaka yang berbeda dapat "
    "menggeser digit terakhir representasi desimal, sehingga kesamaan per *byte* hanya dijamin pada "
    "lingkungan yang sama."
))

cells.append(code(
    "reference_path = CFG.ARTIFACT_DIR / 'submission_v13.csv'\n"
    "reference = pd.read_csv(reference_path, float_precision='round_trip')\n"
    "ref_pred = submission[['record_id']].merge(reference, on='record_id', validate='one_to_one')['prediction'].to_numpy()\n"
    "max_diff = float(np.abs(submission['prediction'].to_numpy() - ref_pred).max())\n"
    "byte_identical = CFG.SUBMISSION_CSV.read_bytes() == reference_path.read_bytes()\n"
    "\n"
    "print(f'SHA256 referensi       : {hashlib.sha256(reference_path.read_bytes()).hexdigest()}')\n"
    "print(f'identik per byte       : {byte_identical}')\n"
    "print(f'selisih nilai maksimum : {max_diff:.2e}')\n"
    "if CFG.RUN_MODE == 'INFERENCE':\n"
    "    assert max_diff < 1e-6, 'submisi tidak mereproduksi submission_v13.csv'\n"
    "    print('Reproduksi submission_v13.csv: BERHASIL')"
))

cells.append(md("> ### Insights\n>\n> Berkas yang dihasilkan notebook identik per *byte* dengan `submission_v13.csv` yang disubmit: SHA256 keduanya `0bd8ba26...af14` dan selisih nilainya tepat nol. Seluruh rantai dari artefak lapisan fitur sampai submisi dapat direproduksi tanpa GPU dan tanpa token.\n>\n> Pada mode `FULL`, lapisan fitur dihitung ulang di GPU dan dapat berbeda sangat sedikit antar-*run*. Lapisan fitur hanya masuk ke prediksi akhir melalui versi 9 dengan bobot 1/16, sehingga perbedaan tersebut teredam enam belas kali."))

# ---------------------------------------------------------------------------
# Section 8: Dashboard Solusi
# ---------------------------------------------------------------------------
cells.append(section("Dashboard Solusi", "8"))

cells.append(md(
    "Dashboard berikut merangkum seluruh solusi dalam satu halaman: indikator utama di baris atas, "
    "perjalanan skor CV dan *public* dari v1 sampai v13, sebaran galat per *fold*, kalibrasi per rentang "
    "target, galat per desil posisi, dan prediksi akhir sepanjang berkas *test*. Skor *public* adalah skor "
    "yang tercatat saat submisi dan tidak dipakai dalam pemodelan."
))

cells.append(code(
    "JOURNEY = pd.DataFrame({\n"
    "    'versi' : ['v1', 'v2', 'v3', 'v5', 'v6b', 'v8', 'v9', 'v10', 'v11', 'v12', 'v13'],\n"
    "    'cv'    : [7.41, 7.40, 7.32, 7.30, 7.26, 7.23] + [s['RMSE sebelumnya'] for s in STAGE_LOG] + [STAGE_LOG[-1]['RMSE tahap']],\n"
    "    'public': [7.947, 7.902, 7.888, 7.888, 7.837, 7.900, 2.387, 2.107, 1.955, 1.825, np.nan],\n"
    "})\n"
    "calib_v13 = pd.DataFrame({'y': y, 'fitur': feature_oof.mean(0), 'v13': v13_oof.mean(0), 'bin': bins}).groupby(\n"
    "    'bin', observed=True).apply(lambda d: pd.Series({'fitur': (d.y - d.fitur).mean(), 'v13': (d.y - d.v13).mean()}))\n"
    "\n"
    "fig = plt.figure(figsize=(22, 13))\n"
    "gs = GridSpec(3, 4, height_ratios=[0.42, 1, 1], figure=fig)\n"
    "kpi(fig.add_subplot(gs[0, 0]), f'{fold_tab[\"fitur\"].mean():.2f} -> {fold_tab[\"v13\"].mean():.2f}', 'RMSE CV: fitur saja -> v13', CFG.C_ACCENT)\n"
    "kpi(fig.add_subplot(gs[0, 1]), STAGE_LOG[-1]['fold membaik'], 'fold membaik pada tahap v13', CFG.C_TEST)\n"
    "kpi(fig.add_subplot(gs[0, 2]), '7.947 -> 1.825', 'RMSE public: v1 -> v12')\n"
    "kpi(fig.add_subplot(gs[0, 3]), 'IDENTIK' if max_diff < 1e-6 else 'BERBEDA', 'reproduksi submission_v13.csv',\n"
    "    CFG.C_TEST if max_diff < 1e-6 else CFG.C_ACCENT)\n"
    "\n"
    "a = fig.add_subplot(gs[1, :2])\n"
    "a.plot(JOURNEY['versi'], JOURNEY['cv'], marker='o', lw=2.5, color=CFG.C_ACCENT, label='RMSE CV')\n"
    "a.plot(JOURNEY['versi'], JOURNEY['public'], marker='s', lw=2.5, color=CFG.C_TEST, label='RMSE public')\n"
    "a.axvspan(5.5, 10.5, color=CFG.C_TRAIN, alpha=0.08)\n"
    "a.text(8, 6.5, 'memakai urutan baris', ha='center', fontsize=11, color=CFG.C_TRAIN)\n"
    "for xv, v in zip(JOURNEY['versi'], JOURNEY['cv']):\n"
    "    a.annotate(f'{v:.2f}', (xv, v), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)\n"
    "a.set(title='Perjalanan skor dari v1 sampai v13', ylabel='RMSE')\n"
    "a.legend()\n"
    "a = fig.add_subplot(gs[1, 2])\n"
    "sns.boxplot(data=fold_tab, ax=a, palette=[CFG.C_GREY] + list(CFG.STAGE_COLORS.values()))\n"
    "a.set(title='RMSE per fold', ylabel='RMSE')\n"
    "a = fig.add_subplot(gs[1, 3])\n"
    "calib_v13.plot.bar(ax=a, color=[CFG.C_GREY, CFG.STAGE_COLORS['v13']], rot=20)\n"
    "a.axhline(0, color='black', lw=0.8)\n"
    "a.set(title='Bias per rentang target', xlabel='rentang target', ylabel='target - prediksi')\n"
    "a = fig.add_subplot(gs[2, :2])\n"
    "sns.heatmap(dec_tab[['fitur', 'v9', 'v13']].T, annot=True, fmt='.1f', cmap='flare', ax=a, cbar=False)\n"
    "a.set(title='RMSE per desil posisi train.csv', xlabel='desil posisi (atas ke bawah)')\n"
    "a = fig.add_subplot(gs[2, 2:])\n"
    "a.plot(pos_te, v9_test, color=CFG.STAGE_COLORS['v9'], lw=1.5, label='v9')\n"
    "a.plot(pos_te, v13_test, color=CFG.STAGE_COLORS['v13'], lw=2.5, label='v13 (submisi)')\n"
    "a.set(title='Prediksi akhir sepanjang test.csv', xlabel='posisi di test.csv', ylabel='prediksi')\n"
    "a.legend()\n"
    "fig.suptitle('Dashboard Solusi - FIK FAIR 2026', fontsize=17, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '18_solution_dashboard.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "display(calib_v13.round(2))"
))

cells.append(md("> ### Insights\n>\n> Dashboard merangkum cerita solusi. Delapan iterasi berbasis fitur hanya menurunkan RMSE *public* dari 7,947 ke 7,837, sedangkan temuan urutan baris menurunkannya ke 2,387. Empat tahap koreksi berikutnya menurunkan RMSE secara konsisten, baik di CV (3,61 ke 3,02) maupun di *public* (2,387 ke 1,825). Arah perubahan CV dan *public* selalu sama sejak v9, padahal skor *public* tidak pernah dipakai untuk memilih apa pun.\n>\n> Panel kalibrasi menunjukkan masalah utama lapisan fitur yang terselesaikan. Bias pada baris dengan target di atas 70 turun dari +15,60 menjadi +3,89 poin, dan bias pada rentang 30 sampai 40 turun dari -3,51 menjadi +0,09. Model akhir tidak lagi menyusut ke tengah."))

# ---------------------------------------------------------------------------
# Section 9: Pasca-Analisis
# ---------------------------------------------------------------------------
cells.append(section("Pasca-Analisis", "9"))

cells.append(md("## Analisis Kesalahan"))

cells.append(md(
    "Galat *out-of-fold* versi 13 dianalisis per *record* untuk melihat baris mana yang masih sulit, "
    "apakah galatnya memiliki arah yang sistematis, dan bagaimana galat tersebar sepanjang berkas."
))

cells.append(code(
    "oof13 = v13_oof.mean(0)\n"
    "err = y - oof13\n"
    "se = err ** 2\n"
    "worst = pd.DataFrame({'record_id': train['record_id'], 'posisi': pos_tr, 'target': y, 'prediksi': oof13,\n"
    "                      'galat': err}).reindex(np.argsort(-np.abs(err))).head(10)\n"
    "display(worst.round(2))\n"
    "\n"
    "fig, ax = plt.subplots(1, 3, figsize=(20, 4.8))\n"
    "ax[0].scatter(pos_tr, err, s=10, alpha=0.6, c=np.abs(err), cmap='flare')\n"
    "ax[0].axhline(0, color='black', lw=0.8)\n"
    "ax[0].set(title='Galat v13 sepanjang train.csv', xlabel='posisi', ylabel='target - prediksi')\n"
    "ax[1].scatter(y, err, s=10, alpha=0.6, color=CFG.C_TRAIN)\n"
    "ax[1].axhline(0, color='black', lw=0.8)\n"
    "ax[1].set(title='Galat terhadap nilai target', xlabel='target', ylabel='target - prediksi')\n"
    "ax[2].plot(np.arange(1, len(y) + 1) / len(y), np.cumsum(np.sort(se)[::-1]) / se.sum(), color=CFG.C_ACCENT, lw=2)\n"
    "ax[2].set(title='Kurva konsentrasi squared error', xlabel='proporsi record (galat terbesar dulu)',\n"
    "          ylabel='proporsi squared error kumulatif')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '19_error_analysis.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "\n"
    "print(f'38 baris teratas (10%) menyumbang {se[:38].sum() / se.sum():.1%} squared error')\n"
    "print(f'10 record terburuk menyumbang {np.sort(se)[::-1][:10].sum() / se.sum():.1%} squared error')\n"
    "print(f'galat positif (under-predict) pada 38 baris teratas: {(err[:38] > 0).sum()} | negatif: {(err[:38] < 0).sum()}')"
))

cells.append(md("> ### Insights\n>\n> Galat masih terkonsentrasi. 38 baris teratas berkas (10%) menyumbang 35,5% *squared error*, dan 10 *record* terburuk saja menyumbang 28,4%. Seluruh *record* terburuk berada di 70 baris pertama berkas.\n>\n> Galat di bagian atas berkas memiliki dua arah: 21 *under-predict* dan 17 *over-predict*. Contohnya D0414 (target 75,85, diprediksi 65,83) dan sebaliknya D0203 (target 57,66, diprediksi 65,54). Koreksi sepihak seperti menaikkan semua prediksi teratas tidak dibenarkan. Yang sulit adalah baris yang targetnya menyimpang dari tetangga di urutan berkas, yaitu deviasi lokal yang sudah terlihat pada EDA."))

cells.append(md("---\n\n## Simulasi *Public* vs *Private*"))

cells.append(md(
    "*Public leaderboard* hanya dihitung dari sekitar 29 baris. Untuk memahami seberapa besar skor dapat "
    "berubah hanya karena pemilihan baris, galat *out-of-fold* versi 13 dipakai untuk mensimulasikan 5.000 "
    "berkas *test* berukuran 96 baris, masing-masing dibagi acak menjadi 29 baris *public* dan 67 baris "
    "*private* yang saling lepas."
))

cells.append(code(
    "sim_rng = np.random.default_rng(CFG.SEED)\n"
    "sim = []\n"
    "for _ in range(5000):\n"
    "    r = sim_rng.integers(CFG.N_REPEATS)\n"
    "    rows = sim_rng.choice(len(y), 96, replace=False)\n"
    "    e2 = (v13_oof[r, rows] - y[rows]) ** 2\n"
    "    sim.append((np.sqrt(e2[:29].mean()), np.sqrt(e2[29:].mean())))\n"
    "sim = pd.DataFrame(sim, columns=['public', 'private'])\n"
    "sim['gap'] = sim['public'] - sim['private']\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(16, 4.5))\n"
    "ax[0].hist(sim['public'], bins=60, alpha=0.6, density=True, color=CFG.C_TEST, label='public (29 baris)')\n"
    "ax[0].hist(sim['private'], bins=60, alpha=0.6, density=True, color=CFG.C_TRAIN, label='private (67 baris)')\n"
    "ax[0].set(title='Sebaran RMSE simulasi', xlabel='RMSE', ylabel='densitas')\n"
    "ax[0].legend()\n"
    "ax[1].hist(sim['gap'], bins=60, color=CFG.C_ACCENT, alpha=0.7)\n"
    "ax[1].axvline(0, color='black', lw=0.8)\n"
    "ax[1].set(title='Selisih public - private', xlabel='selisih RMSE')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.FIG_DIR / '20_public_private.png', dpi=120, bbox_inches='tight')\n"
    "plt.show()\n"
    "print(sim.describe(percentiles=[0.05, 0.5, 0.95]).T[['mean', '5%', '50%', '95%']].round(3).to_string())\n"
    "print(f'P(|public - private| > 0.5): {(sim.gap.abs() > 0.5).mean():.1%}')"
))

cells.append(md("> ### Insights\n>\n> Hanya karena pemilihan baris, RMSE 29 baris *public* bervariasi dari 2,06 sampai 4,22 (persentil 5 sampai 95), jauh lebih lebar daripada 67 baris *private* (2,40 sampai 3,84). Separuh simulasi memiliki selisih *public* dan *private* lebih dari 0,5. Dengan variasi sebesar ini, mengoptimalkan skor *public* sama dengan mengoptimalkan *noise*. Karena itu seluruh keputusan diambil dari CV *nested*.\n>\n> Skor *public* aktual v12 (1,825) berada di bawah rentang simulasi. Ini wajar karena model final dilatih pada seluruh 382 baris, sedangkan simulasi memakai model *out-of-fold* yang hanya melihat 306 baris, dan *learning curve* menunjukkan galat masih turun seiring bertambahnya data. Simulasi lebih tepat dibaca sebagai ukuran ketidakpastian, bukan ramalan skor *private*."))

cells.append(md(
    "---\n\n"
    "### Kelebihan Pendekatan\n\n"
    "Solusi ini memadukan dua sumber informasi yang saling melengkapi. *Foundation model* tabular memberi "
    "prediksi terbaik yang dapat diperoleh dari fitur dengan hanya 382 baris, tanpa *training* gradien yang "
    "rawan *overfitting*. Urutan baris memberi informasi tentang ekor kanan yang tidak terlihat dari fitur. "
    "Setiap tahap koreksi bersifat ringan (Ridge dan *Kernel Ridge* di CPU), dipilih melalui seleksi *nested*, "
    "dan digabungkan dengan bobot tetap, sehingga peluang *overfitting* terhadap validasi tetap kecil. "
    "Mode `INFERENCE` mereproduksi submisi secara identik dalam beberapa menit, dan seluruh angka validasi "
    "dihitung ulang di dalam notebook.\n\n"
    "---\n\n"
    "### Keterbatasan yang Diketahui\n\n"
    "Lapisan urutan mengasumsikan bahwa `test.csv` disusun dengan cara yang sama seperti `train.csv`. "
    "Asumsi ini didukung kuat oleh dua bukti tidak langsung dan penggunaannya telah dikonfirmasi panitia, "
    "namun tidak dapat dibuktikan tanpa label *test*, dan metode ini tidak berlaku untuk data masa depan "
    "yang tidak terurut. Tanpa urutan, solusi terbaik berada di sekitar RMSE 7,3. Seleksi *nested* "
    "menghilangkan kebocoran pemilihan parameter di dalam *fold*, tetapi katalog kandidat dirancang setelah "
    "mempelajari 382 baris yang sama, dan tidak ada data *holdout* lokal yang benar-benar belum tersentuh. "
    "Lapisan fitur memakai *pseudo-labelling* transduktif pada fitur *test* (tanpa label), sehingga "
    "keseluruhan pipeline tidak bebas *pseudo-label*.\n\n"
    "---\n\n"
    "### Arah Pengembangan Berikutnya\n\n"
    "Galat terbesar terkonsentrasi pada sekitar 10% baris teratas berkas. Arah yang paling menjanjikan "
    "adalah model residual yang khusus menangani bagian atas berkas, misalnya *heteroscedastic kernel* "
    "dengan varians yang bergantung pada posisi, serta taksiran posisi yang memanfaatkan *foundation model* "
    "sebagai pengukur kemiripan, bukan hanya PCA-12. *Learning curve* komponen *kernel* yang masih turun "
    "(3,73 pada 143 baris menjadi 3,12 pada 286 baris saat pengembangan) menunjukkan tambahan data berlabel "
    "masih akan membantu.\n\n"
    "---\n\n"
    "### Kepatuhan terhadap Aturan\n\n"
    "Hanya dataset resmi yang dipakai. Tidak ada data eksternal, identifikasi entitas asli, rekonstruksi "
    "target *test*, maupun *probing leaderboard*. Skor *public* hanya dicatat dan tidak pernah dipakai "
    "untuk memilih parameter atau mengubah prediksi per baris. `record_id` tidak diubah. Penggunaan urutan "
    "baris diungkapkan secara terbuka dan telah dikonfirmasi panitia pada 4 Oktober 2026. AI/LLM dipakai "
    "sebagai alat bantu *coding*, analisis, dan dokumentasi sesuai aturan kompetisi."
))

# ---------------------------------------------------------------------------
# Section 10: Daftar Pustaka
# ---------------------------------------------------------------------------
cells.append(section("Daftar Pustaka", "10"))

cells.append(md(
    "Barlow, R. E., Bartholomew, D. J., Bremner, J. M., & Brunk, H. D. (1972). *Statistical Inference under "
    "Order Restrictions: The Theory and Application of Isotonic Regression*. Wiley.\n\n"
    "Cleveland, W. S. (1979). *Robust Locally Weighted Regression and Smoothing Scatterplots*. Journal of the "
    "American Statistical Association, 74(368), 829-836.\n\n"
    "Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall.\n\n"
    "FIK FAIR 2026. (2026). *Data Science Competition FIK FAIR 2026*. Kaggle. "
    "https://www.kaggle.com/competitions/data-science-competition-fikfair2026\n\n"
    "Hollmann, N., Muller, S., Purucker, L., Krishnakumar, A., Korfer, M., Hoo, S. B., Schirrmeister, R. T., & "
    "Hutter, F. (2025). *Accurate predictions on small data with a tabular foundation model*. Nature, 637, "
    "319-326.\n\n"
    "Lee, D.-H. (2013). *Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural "
    "Networks*. ICML Workshop on Challenges in Representation Learning.\n\n"
    "Marchenko, V. A., & Pastur, L. A. (1967). *Distribution of eigenvalues for some sets of random matrices*. "
    "Mathematics of the USSR-Sbornik, 1(4), 457-483.\n\n"
    "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., "
    "Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., "
    "Perrot, M., & Duchesnay, E. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine "
    "Learning Research, 12, 2825-2830.\n\n"
    "Qu, J., Holzmuller, D., Varoquaux, G., & Le Morvan, M. (2025). *TabICL: A Tabular Foundation Model for "
    "In-Context Learning on Large Data*. ICML 2025. arXiv:2502.05564.\n\n"
    "Scholkopf, B., & Smola, A. J. (2002). *Learning with Kernels: Support Vector Machines, Regularization, "
    "Optimization, and Beyond*. MIT Press."
))

cells.append(md(f"{HR}\n\n# Terima Kasih <a name=\"\"></a>\n\n{HR}"))

cells.append(md(
    "**Terima kasih khusus kepada:**\n\n"
    "Panitia **Data Science Competition FIK FAIR 2026**, Fakultas Ilmu Komputer UPN \"Veteran\" Jakarta, "
    "yang telah menyediakan dataset dan platform kompetisi yang menantang."
))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.12"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
