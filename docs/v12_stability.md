# v12: reducing subset risk, not guaranteeing public/private equality

User-reported v11 public RMSE: 1.9549491170. No leaderboard score was used as an optimization target, and no submission was uploaded. The actual private targets and membership are unknown.

## Diagnosis and evidence

In archived v11 OOF, the first 38 training records contribute 39.52% of squared error. Small public subsets can miss, or disproportionately include, difficult records. Equal public/private scores are therefore not an attainable guarantee. Minimizing their difference alone would also be a bad objective: an equally poor model on both subsets could satisfy it.

The search compared 15 fixed residual candidates: Gaussian or local-linear trend removal, with/without residual-feature × rank interactions, Ridge penalties 10/100, and fixed half-strength hedges. Discovery used 100 training/query splits (286/96), with each query split into disjoint public29/private67 subsets 50 times. No foundation-model OOF was reused across incompatible partitions.

The selected extension uses local-linear trends and rank interactions. Local-linear smoothing can reduce boundary bias; this is a motivation, not a guarantee for this competition ([Statsmodels documentation](https://www.statsmodels.org/dev/generated/statsmodels.nonparametric.kernel_regression.KernelReg.html)). Our implementation is a Gaussian-weighted local-linear smoother; no Statsmodels dependency was added.

Confirmation used 200 new 286/96 partitions, each with 25 paired public/private splits. These are new partitions of the same records, **not fresh independent data**. The 5,000 pairs are not 5,000 independent experiments.

| Standalone component | Mean query RMSE | Simulated private p95 | Mean absolute public/private gap |
|---|---:|---:|---:|
| Original Gaussian residual Ridge10 | 3.4692 | 4.6794 | 0.7474 |
| Local-linear + interaction Ridge10 | 3.3586 | 4.5641 | 0.7346 |

This table compares components, not the complete v11/v12 ensembles. Private p95 and the gap are descriptive stability checks, not fitting objectives or forecasts of the actual leaderboard. The modest gap reduction matters: most subset variability remains.

## Full candidate

v12 = 0.5 × exact v11 submission + 0.5 × new selected component. Weights are fixed, not fitted to public scores or outer-validation labels. Inner 5-fold CV chooses between the original Gaussian component and the new local-linear interaction component. All 15 outer folds and full-data CV chose the latter.

| Pipeline | Mean fold RMSE | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---:|---:|---:|---:|
| v11 | 3.3031 | 3.1877 | 3.5095 | 3.3243 |
| v12 | 3.1789 | 3.0665 | 3.4122 | 3.1813 |

Improvement in 14/15 folds and all three repeats. Record-cluster bootstrap, keeping three predictions of a record together, gives a conditional delta interval [-0.1871, -0.0538]. This interval does not include the entire research-selection uncertainty.

Approximate resampling of archived full-pipeline OOF into 96-record cohorts split 29/67 gives mean absolute gap 0.6585 → 0.6397 and private p95 4.1112 → 3.9905. These cohorts mix predictions from different outer models and are **not** coherent held-out 96-row inference; use the component experiment above for that design. Neither simulation predicts that actual private RMSE will be around 4.

First-38-record RMSE falls 6.6636 → 6.1359, and their share of SSE falls 39.52% → 36.05%. Hard high-target records remain a dominant unresolved risk.

## Leakage, interpretation, and limits

- Outer labels are excluded from scaling, PCA, rank calibration, smoothing, Ridge, and inner selection for the new component. Baseline OOF uses the same outer partitions and the prior v11 audit artifacts.
- Low RMSE alone never proves the absence of leakage. File order itself carries target-related information from dataset construction. The team reports organizer permission; permission does not make this a feature-only model or establish generalization to unordered future data.
- The paper must disclose use of order, its assumptions, baseline ancestry, all candidate families, and development reuse. Nested CV addresses within-fold selection, not the fact that methods were invented after examining this dataset.
- No new pseudo-labeling is added by v12, but the inherited v11 → v10 → v9 baseline contains pseudo-labeled foundation-model predictions. Do not call the whole pipeline pseudo-label-free.
- This batch has not established an irreducible error floor or exhausted all possible methods. More adaptive searching on the same 382 records can increase research overfitting. A lower public score would not establish private improvement.
- The latest user says six competition days remain; older local deadline statements were not used to stop this research.

## Reproduction and deliverables

- `notebooks/v12.ipynb`: standalone CPU notebook; attach official data and `submission_v11.csv`. Optional `outputs/stability_search/v11_oof.csv` enables paired validation.
- `outputs/fikfair_v12/submission_v12.csv`: 96 unique IDs, finite predictions, template order preserved. RMS change from v11: 0.55442.
- All code cells actually executed sequentially through a Python harness, not merely parsed. Audit predictions match to absolute tolerance 1e-10. The full-data selection and export ran twice with byte-identical CSV in the tested environment; this is not a cross-platform guarantee.
- Manifest records input hashes, versions, seed560199, and output hash. Preserve the full ancestor pipeline and artifacts for end-to-end reproduction.
- Runnable scripts: `eda/v9/55_stability_search.py`, `56_stability_confirm.py`, `notebooks/build_v12.py`, `eda/v9/57_export_v12.py`. Use the existing `.venv/bin/python`, with OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS set to 1. Assertions check disjointness, constant/linear trend recovery, deterministic component predictions, ID alignment, and notebook/audit agreement.

SHA256 submission: `f4f24e0cf5aa480b77d724f82b2befe69267492dfd06dcb352cff8eda11cb462`.
