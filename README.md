# Aquatic Ecotoxicity Prediction: Molecular Representations and Similarity-Based Relationships

Code accompanying the manuscript **“Comparative Evaluation of Molecular Representations and Similarity-Based Inter-Chemical Relationships for Aquatic Ecotoxicity Prediction.”**

## Overview

This study compares molecular representations for aquatic toxicity prediction: RDKit 2D descriptors, Count Morgan fingerprints, intramolecular molecular graphs (G1), and inter-chemical similarity graphs (G2). The main assessment of added relational information compares **descriptor-only GraphSAGE** with **inter-chemical GraphSAGE (G2)** using matched molecular descriptors and model settings, with the graph edges as the intended comparison.

Experimental evaluations involve TEST *Daphnia magna*, TEST fathead minnow, and an independent ADORE *Daphnia magna* dataset. **WebTEST is a separate surrogate-label experiment:** its reference values are predictions from a QSAR consensus system, **not independently measured aquatic toxicity endpoints**.

## Models and study design

The models include (1) RDKit 2D descriptor random forest, (2) Count Morgan fingerprint random forest, (3) intramolecular GraphSAGE (G1), (4) six-descriptor random forest, (5) descriptor-only GraphSAGE, and (6) inter-chemical GraphSAGE (G2).

Structural split seeds and model initialization seeds are `42`, `123`, `777`, `2025`, and `3407`. The primary G2 graph uses exact Tanimoto similarities between Count Morgan fingerprints (radius 4, 2048 dimensions, no chirality), with up to 10 training-set neighbors of positive similarity and unweighted message passing. Neighborhood sizes, thresholds, and edge-weighting choices are evaluated separately as sensitivity analyses.

GraphSAGE models use two layers, 128 hidden units, dropout `0.30`, Adam learning rate `0.001`, weight decay `0.0001`, and validation-based early stopping with patience `50`. The earlier experimental-data analyses used a maximum of **500 epochs**. Additional WebTEST training experiments used a maximum of **5,000 epochs** with new model and optimizer initialization. This extended training budget was investigated after earlier results had been inspected; it is an **exploratory follow-up**, not an independent external confirmation.

The random-forest baselines use 500 trees, unrestricted maximum depth, minimum leaf size 1, `max_features=1.0`, and bootstrap sampling. Data-dependent preprocessing is fitted on training partitions only. Held-out test data are reserved for final evaluation, not checkpoint selection.

## Repository layout

| Directory | Contents |
| --- | --- |
| `code/` | Manuscript-facing model training, evaluation, and figure-generation scripts |
| `G1_5000_GPU/` | Fresh WebTEST G1 study runner and unchanged original G1 source files |
| `webtest_fresh5000/` | Fresh WebTEST descriptor-only and G2 study runner and supplementary neighborhood analysis |
| `scripts/` | Additional graph-building and experimental analyses |
| `R15_local_python_Anaconda/` | Chemical-level and structural-subgroup analyses using saved predictions |
| `docs/` | Supporting notes on model pipelines |
| `environment.yml`, `requirements.txt` | Dependency definitions for the earlier pipeline |

These code files are distributed separately from input data and results. Historical checkpoint weights are not included in the public code or accompanying data archives.

## Companion archives and setup

The companion Zenodo release is organized into **two distinct archives**:

- **`Zenodo_input_data_splits.zip`**: the curated WebTEST input table (119,276 chemicals), frozen WebTEST scaffold splits, five sets of precomputed exact Top-10 G2 graph arrays, 50 configuration-only G2 model specifications, G1 row mapping and graph metadata, and selected split/summary inputs.
- **`Zenodo_result_records.zip`**: the saved WebTEST G1/G2 predictions and metrics, sensitivity and absolute-weighted records for R15, per-query similarity summaries, and additional manuscript/supplementary result records. The release-ready archive should name its root-level note **`ZENODO_RESULTS_README.md`**, avoiding a collision with this repository's `README.md`.

Create a **separate analysis workspace**, outside the GitHub repository used to prepare publication. Copy the released code directories there, then extract the *contents* of both Zenodo ZIPs into the same workspace root, preserving their relative paths. Do not commit the unpacked data, outputs, or checkpoints back into the code repository. A typical workspace has:

```text
analysis_workspace/
├── 03_processed/
├── 04_splits/
├── 05_features/
├── 07_results/
├── revision_parallel/
├── fathead_sensitivity/
├── absolute_weight_run/
├── revision_next/
├── code/
├── G1_5000_GPU/
├── webtest_fresh5000/
├── scripts/
└── R15_local_python_Anaconda/
```

Run the examples below **from `analysis_workspace/`**. The two data archives share two identical per-query neighborhood CSV paths; their contents were checked to be identical in the supplied release candidates.

### Scope of the supplied data

The supplied companion archives cover the **fresh WebTEST workflows, S9 recalculation, and R15 reanalysis**, as well as selected records for the experimental datasets. They do **not** include all original processed TEST/ADORE input tables, reference caches, or historical model weights needed to rerun *every* earlier experimental-data script in `code/` from scratch. Treat those scripts and saved metrics as provenance and analysis material, not a claim that every original pipeline is fully executable from these two archives alone.

The original WebTEST graph builders (`code/webtest_build_similarity_graphs.py` and `scripts/P10C_build_WebTEST_exact_Top10_graphs_and_QC.py`) also require their original reference caches under different paths (`external_cache/webtest_reference/` and `TOXICS_LEGACY_ROOT` / `legacy_cache/`, respectively). These caches are **not** distributed. The exact Top-10 graph arrays needed by the **fresh G2 runner** are provided in `Zenodo_input_data_splits.zip`, so rerunning the original graph builder is not necessary for the fresh G2 training workflow.

## Reproducing WebTEST analyses

Install compatible versions of NumPy, pandas, SciPy, RDKit, scikit-learn, PyTorch, and PyTorch Geometric. Recorded G1 runs used RDKit `2025.09.6`, NumPy `1.26.4`, PyTorch `2.10.0+cu128`, and PyTorch Geometric `2.7.0`. **Fresh G1/G2 training requires a supported CUDA GPU.** Exact numerical identity across different platforms or versions is not guaranteed.

Use a **new, empty output directory** for each fresh experiment. Do not point training outputs at the original companion-data directories.

### Descriptor-only and G2 (4 conditions × 5 splits × 5 seeds)

The fresh G2 runner reads the 50 JSON specifications under `07_results/WebTEST_revised/P10D_G2_self_validation/model_specs/`, as well as the split CSVs and 15 precomputed graphs. The specifications include training-fitted preprocessing information but **no historical model weights**.

```bash
python webtest_fresh5000/run_fresh5000.py --project-root . --out ../webtest_G2_reproduction --stage check
python webtest_fresh5000/run_fresh5000.py --project-root . --out ../webtest_G2_reproduction --stage train
python webtest_fresh5000/run_fresh5000.py --project-root . --out ../webtest_G2_reproduction --stage evaluate
```

Run `evaluate` only after all 100 training runs have completed and passed the runner's checks.

### Intramolecular G1 (5 splits × 5 seeds)

The provided input archive contains the G1 graph schema, metadata, and frozen row map but **not** the serialized unlabeled graph tensor `05_features/G1_WebTEST_revised/g1_graph_assets.pt`. Generate that asset from the supplied WebTEST table:

```bash
python G1_5000_GPU/original_scripts/P05G_01_build_G1_graph_assets.py --input_table 03_processed/WebTEST/WebTEST_Daphnia_consensus_unique_connectivity_frozen.csv --out_dir 05_features/G1_WebTEST_revised --smiles_col SMILES_RDKit
```

The builder writes the regenerated `.pt` graph and corresponding metadata/row-map files. Verify the graph schema, SMILES ordering and asset SHA-256 with `--stage check` before training; the regenerated `.pt` need not have the same binary hash as the original historical graph archive. Run the full graph build in a trusted environment; PyTorch serialized objects must not be loaded from untrusted sources.

```bash
python G1_5000_GPU/run_webtest_G1_fresh5000.py --project-root . --out ../webtest_G1_reproduction --stage check
python G1_5000_GPU/run_webtest_G1_fresh5000.py --project-root . --out ../webtest_G1_reproduction --stage train
python G1_5000_GPU/run_webtest_G1_fresh5000.py --project-root . --out ../webtest_G1_reproduction --stage evaluate
```

Run `evaluate` only after all 25 G1 training runs have completed and passed the runner's checks. The five source files in `G1_5000_GPU/original_scripts/` are distributed with their recorded source SHA-256 checksums and should not be modified for historical source verification.

### Supplementary Table S9 — saved prediction reanalysis

This analysis uses **already-saved held-out predictions** and does not train any model. When using `Zenodo_result_records.zip`, run:

```bash
python webtest_fresh5000/analyze_neighborhood_gain.py --predictions webtest_fresh5000/fresh5000_results/results_for_review.zip --neighbors 07_results/diagnostics/P15_cross_context_Top10_similarity_QC/01_per_query_Top10_similarity.csv --out ../s9_recalculated
```

If using a **new G2 reproduction** instead of the archived records, change the prediction argument to `--predictions ../webtest_G2_reproduction/results_for_review.zip`. The neighborhood CSV is a separate input. S9 summarizes descriptive associations; it does not establish a causal mechanism. On the supplied archived inputs, the 60 split-level coefficients and 12 summary records were independently recalculated and matched the distributed result tables.

### Structural-subgroup analyses — R15

`R15_local_python_Anaconda/R15_local_python/run_analysis.py` uses **existing predictions and neighborhood records**. The supplied archives provide the relevant inputs under `revision_parallel/sensitivity_results/`, `fathead_sensitivity/fathead_results/`, `absolute_weight_run/`, and `revision_next/neighbor_results/`.

To check inputs without running the analyses:

```bash
python R15_local_python_Anaconda/R15_local_python/run_analysis.py --project-root . --check-inputs
```

To run all three analysis stages (no model training):

```bash
python R15_local_python_Anaconda/R15_local_python/run_analysis.py --project-root .
```

The runner writes its outputs below `R15_local_python_Anaconda/R15_local_python/output/`. The supplied archive candidates were used to complete all three R15 stages in a separate CPU environment.

## Reproducibility, provenance, and limits

- Source-code and archived-result hashes distinguish the historical experimental sources from the packaged public entry points. The public G2 training runner and engine preserve the study's intended training computations but are not byte-identical to all archived historical source files.
- Historical checkpoint weights are intentionally omitted. The configuration-only fresh G2 workflow uses new models and optimizers; the G1 workflow regenerates the unlabeled molecular graph tensor.
- The two companion archives and the code were checked for file integrity, and S9 plus R15 reanalysis were executed with their archived inputs. **Full end-to-end CUDA retraining of the public G1 and G2 packages has not been independently verified.**
- The supplied archives do not establish permission to redistribute third-party source datasets. Source licensing and relevant author/institutional approvals must be verified before publishing the data release.

Refer to the manuscript and Supplementary Information for detailed endpoint definitions, data curation, model comparison, limitations, and interpretation of the reported findings. Cite the exact version and DOI of the published code/data release in the manuscript once available.
