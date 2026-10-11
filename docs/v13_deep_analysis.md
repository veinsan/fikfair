# v13: deep EDA, preprocessing ablations, and stability checks

## Scope and conclusion

Public scores 1.8247135700 (v12) and 1.8073794208 (competitor) are user reports. Neither score is used to infer test labels, adjust individual test records, or select parameters. No submissions were uploaded. This work produces an evidence-backed v13 candidate, not a guarantee of rank 1 or a proof of an irreducible dataset floor.

The audit and experiments were actually executed. Figures were inspected after generation, including preprocessing before/after, the correlation matrix, fold comparisons, grouped stress tests, learning curves, kernel sensitivity, and final prediction changes.

## 1. Data and failure modes

382 training records, 96 test records, 24 numeric features. No missing training values and no exact feature duplicates within train or across train/test were found. Nearest standardized-Euclidean train distance is 0.20373; nearest test-to-train distance is 0.17853. These are diagnostics, not thresholds proving identity.

Examples of very close pairs:

| Pair | Feature distance | Absolute target difference |
|---|---:|---:|
| D0250 / D0183 | 0.20373 | 2.74216 |
| D0049 / D0440 | 0.20991 | 6.58195 |
| D0080 / D0149 | 0.21251 | 0.99127 |

Deleting all near neighbors would discard potentially valid observations. Similar feature vectors with different targets do not establish label error or irreducible noise. Instead, proximity-connected components are kept together in a stress test. They are label-blind heuristic groups, not verified entities.

File position versus target Spearman correlation is -0.98090. The first 38 records contribute 36.05% of v12 OOF SSE; the ten largest-error records contribute 24.42%. Error analysis identifies D0419, D0414, and D0203 among the worst records. Errors have both signs, so a blanket upward correction is not justified.

The residual plot shows unequal error variance: high-target early rows are much harder. The correlation matrix describes relationships but does not identify unknown test errors or prove causality.

## 2. Preprocessing: proposed, applied, inspected, evaluated

The inherited rank estimate uses training-only scaling/PCA and neighboring feature vectors. Local-linear trends are subtracted from training features and targets. Query features subtract the training feature trends interpolated at their estimated ranks. Feature residuals and residual × centered-rank interactions then predict deviations around the target curve.

The before/after figure shows that broad feature trends are removed, but residual variance is still unequal. Thus detrending is useful but does not solve the whole task. The target's first few observations remain difficult even after trend removal. A successful preprocessing plot does not imply that any downstream model will work equally well.

Twelve one-change-at-a-time candidates were evaluated on 120 held-out query sets of 96 records:

| Component | Mean query RMSE | Private p95 simulation | Interpretation |
|---|---:|---:|---|
| v12 residual component | 3.51692 | 4.68676 | Reference, not full v12 ensemble |
| Feature clipping | 3.51764 | 4.69713 | No improvement; not retained as preferred method |
| Wider detrending | 3.54417 | 4.71332 | Worse; smoothing can remove useful structure |
| Residual variance scaling | 3.47025 | 4.69186 | Mean better, upper-tail risk not better |
| Log target, naive inverse | 3.47468 | 4.66268 | Modest improvement; no claim of unbiased inverse transform |
| Quadratic rank interaction | 3.45816 | 4.58069 | Some additional structure |
| Kernel residual, alpha1 | 3.31149 | 4.58802 | Strongest mean improvement in this first batch |

All candidate results, including failures, are retained. The nonlinear residual model uses RBF Kernel Ridge. Its squared-error objective and nonlinear function class make it a relevant test for structured residuals, not evidence that it must win. See the [official Kernel Ridge explanation](https://scikit-learn.org/stable/modules/kernel_ridge.html). This source provides method rationale, not a benchmark for this competition.

## 3. Kernel sensitivity and local limits

An initial nine-setting check suggested underfitting by the original kernel. The range was expanded to 30 settings: alpha 0.01/0.03/0.1/0.3/1/3 and gamma multiplier 0.5/1/2/4/8, on 150 additional partitions of the same dataset. These partitions were development data, not an independent test set.

The heatmap shows a broad favorable region around alpha0.1-0.3. Lowest development mean was 2.98493 at alpha0.3/gamma2. Alpha0.01/gamma2 worsened to 3.28068; alpha3/gamma0.5 gave 3.30633. Increasing flexibility indefinitely does not keep improving this family.

The final catalogue contains the original 12 candidates plus eight distinct kernel variants from alpha0.1/0.3/1 and gamma1/2/4. Selection is repeated inside each outer training set. This reduces within-fold parameter-selection leakage, but narrowing the catalogue after research on these same records leaves research-selection optimism.

## 4. Nested full-pipeline comparison

Outer splits match archived v12 OOF: five folds, three repeats, seed42. Inner selection uses only the outer training rows, seed6000+fold. Preprocessing, target trends, residual fitting, and candidate choice exclude outer labels. Fixed final weights are 50% v12 and 50% selected component.

| Pipeline | Mean fold RMSE | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---:|---:|---:|---:|
| v12 | 3.17892 | 3.06654 | 3.41219 | 3.18134 |
| v13 | 3.01556 | 2.87049 | 3.30790 | 2.99831 |

v13 improves all 15 folds and all three repeats. Conditional record-cluster bootstrap delta interval: [-0.23694, -0.07259]. The three repeat predictions of each record are resampled together. This is not a comprehensive uncertainty interval accounting for all prior research decisions.

The first, untuned-kernel catalogue gave v13 mean fold RMSE 3.07614. After the documented sensitivity study and nested reselection it became 3.01556. Do not report the latter as if no intermediate experimentation occurred.

Full-training selection seed6099 chose **kernel alpha0.1, gamma multiplier2**, not the development-table winner alpha0.3. There are 48 residual/interacted inputs, hence actual RBF gamma is 2/48. Selection and prediction were repeated and produced byte-identical CSV in the tested environment.

## 5. Similar-record stress and learning curves

Connected components use standardized feature distances, without targets. Threshold0.35 produces 374 groups, largest size3, with 15 records in non-singleton groups. Threshold0.7 produces 322 groups, largest size22, with 87 records in non-singleton groups. All group members stay on one side of each split, asserted in code. This tests sensitivity to similarity, not known subject-level generalization. See [grouped cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

Across 80 group splits at each threshold, fixed component comparisons are:

| Group distance threshold | Old residual | Kernel alpha1/gamma1 | Kernel alpha0.3/gamma2 |
|---|---:|---:|---:|
| 0.35 | 3.37517 | 3.14051 | 3.02107 |
| 0.70 | 3.53559 | 3.32590 | 3.22299 |

These are development stress tests of fixed components, not nested evaluations of the full final ensemble. Improvement surviving them is reassuring, but cannot prove no entity leakage because real entity labels are unavailable.

Learning curves use the same 96 query records per split and nested training subsets, with 60 splits:

| Training rows | Old residual | Kernel alpha0.3/gamma2 |
|---|---:|---:|
| 143 | 3.97873 | 3.72782 |
| 214 | 3.62518 | 3.33889 |
| 286 | 3.43957 | 3.11664 |

Error still falls with more training data. Therefore an absolute plateau is not established. Additional genuine data could help if permitted, but no external dataset was used here.

## 6. Public/private interpretation

All paired simulations use disjoint public29/private67 samples from the same 96-query block. Matching public and private exactly is not a sensible optimization objective: an equally bad predictor could satisfy that objective. We instead compare mean error, upper-tail private error, and gap as separate diagnostics.

`63_coherent_nested_queries.py` adds 50 held-out query sets with inner selection entirely inside the corresponding 286 training rows, followed by 50 disjoint public/private partitions per query. Results are saved in `coherent_nested_summary.csv` and visualized in `coherent_nested.png`. These compare the old residual component against the nested new component, not the complete inherited ensembles. Repeated subsets are dependent and the underlying 382 records have been used in earlier research.

| Nested-query component | Mean query RMSE | Private p95 simulation | Mean absolute gap |
|---|---:|---:|---:|
| Old residual component | 3.48543 | 4.77904 | 0.82655 |
| Nested new component | 3.19957 | 4.60213 | 0.75221 |

The paired figure still shows substantial overlap and a long upper-error tail. Improvement is supported on average; difficult subsets are not eliminated. The two components share the same query sets and public/private partitions.

Do not subtract CV gain from public1.82471 to forecast a score. Public/private membership, labels, and actual private improvement remain unknown. No leaderboard probing or test-label reconstruction was performed.

## 7. Deliverables and reproduction

Every analysis script prints numbers, writes CSV/JSON as applicable, and saves figures under `outputs/deep_v13/`:

| Script | Purpose | Main images |
|---|---|---|
| `eda/v9/58_deep_audit.py` | Duplicates, closest pairs, correlation, worst records | `audit.png` |
| `eda/v9/59_preprocessing_ablation.py` | Twelve preprocessing/model ablations | `preprocessing.png`, `ablation.png` |
| `eda/v9/61_kernel_sensitivity.py` | Thirty kernel settings and paired subsets | `sensitivity.png` |
| `eda/v9/60_nested_and_group_check.py` | Nested catalogue selection, groups, learning curves | `confirmation.png` |
| `eda/v9/63_coherent_nested_queries.py` | Fully held-out query blocks with nested selection | `coherent_nested.png` |
| `eda/v9/62_export_v13.py` | Execute notebook, assert audit match and save artifacts | `outputs/fikfair_v13/v13_predictions.png` |

Run the analysis in the table's order; before script62, run `notebooks/build_v13.py`. Use the existing `.venv/bin/python` and OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1. Set MPLCONFIGDIR to a writable temporary path. No new packages were installed.

Notebook: `notebooks/v13.ipynb`. Attach official files and exact `submission_v12.csv`; optional `outputs/deep_v13/v12_oof.csv` enables paired validation. CPU only; no token or foundation-model retraining required by this refinement. All notebook code cells were executed sequentially through a Python harness, and predictions match the separate audit at absolute tolerance 1e-10.

Final CSV: `outputs/fikfair_v13/submission_v13.csv`. It has 96 unique IDs, finite predictions, and template order. RMS change from v12: 0.76198. Its plot shows bidirectional local corrections, not a blanket shift or manually edited IDs.

SHA256: `0bd8ba26554f5163d14bfedbc70b9a134cfa458505e268c6ffb105e6cd07af14`.

The manifest stores input hashes, dependency versions, final seed, model selection scores, and output hash. Same-environment reproducibility is verified, not cross-hardware identity. End-to-end reproduction still needs the v9/v10/v11/v12 ancestry. The inherited baseline contains pseudo-labeling; v13 is not a pseudo-label-free pipeline.

For a paper, explicitly disclose the target-related information in file order, organizer permission as reported by the team, repeated dataset exploration, and the absence of an untouched local test set. Low RMSE and exact reproducibility alone do not establish freedom from leakage or overfitting.
