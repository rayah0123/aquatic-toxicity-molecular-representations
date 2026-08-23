# Aquatic toxicity molecular-representation code

Code accompanying the manuscript **Comparative Evaluation of Molecular Representations and Similarity-Based Inter-Chemical Relationships for Aquatic Ecotoxicity Prediction**.

## Scope

This repository contains the manuscript-facing analysis pipeline for comparing:

1. RDKit 2D descriptor random forest
2. Count Morgan fingerprint random forest
3. Intramolecular GraphSAGE (G1)
4. Six-descriptor random forest
5. Descriptor-only GraphSAGE
6. Inter-chemical GraphSAGE (G2)

The release intentionally excludes exploratory model searches, debugging scripts, and analyses that were not used for the final manuscript claims. Descriptive structural-neighborhood characterization used in the manuscript is retained.

## Fixed evaluation design

- Structural split seeds: `42, 123, 777, 2025, 3407`
- Model seeds: `42, 123, 777, 2025, 3407`
- Random forest: 500 trees, unrestricted depth, minimum leaf size 1, `max_features=1.0`, bootstrap sampling
- Count Morgan fingerprints: radius 4, 2048 dimensions, chirality disabled
- G2 neighborhood: exact Tanimoto similarity, Top-10 positive-similarity training neighbors, unweighted message passing
- GraphSAGE: two 128-unit hidden layers, dropout 0.3, Adam learning rate `1e-3`, weight decay `1e-4`, maximum 500 epochs, early stopping patience 50
- Six node descriptors: MolLogP, MolWt, TPSA, NumHAcceptors, NumHDonors, NumRotatableBonds

All data-dependent preprocessing is fit on the training partition only. Validation is used for early stopping/model selection where applicable. Frozen test partitions are evaluated only by the corresponding test-evaluation scripts.

## Repository layout

`code/` contains executable Python scripts and shared modules. The filenames describe their function rather than the internal chronological identifiers used during development.

Key entry points:

- `daphnia_rf_evaluate_test.py` — Daphnia random-forest test evaluation
- `g1_run_multisplit.py` / `g1_evaluate_test.py` — intramolecular GraphSAGE validation and test evaluation
- `g2_run_multisplit.py` / `g2_evaluate_test.py` — inter-chemical GraphSAGE and descriptor-only GraphSAGE validation/test evaluation
- `adore_run_external.py` — independent ADORE external evaluation
- `fathead_run_validation.py` / `fathead_run_test.py` — fathead minnow evaluation
- `webtest_build_similarity_graphs.py` — exact WebTEST Top-10 graph assets
- `webtest_train_g2.py` / `webtest_evaluate_g2.py` — WebTEST G2 and descriptor-only GraphSAGE
- `webtest_evaluate_rf.py` — WebTEST random-forest evaluation
- `webtest_run_g1.py` — WebTEST G1 evaluation
- `evaluate_six_descriptor_rf.py` — matched six-descriptor RF comparison
- `aggregate_manuscript_metrics.py` — manuscript metric aggregation
- `characterize_top10_similarity.py` — Top-10 neighborhood characterization
- `make_figure2.py` and `make_figure_s1.py` — final figure generation

## Data and directory expectations

The code expects the manuscript analysis data and frozen split files in the same relative project hierarchy used for the final analyses, including `03_processed/`, `04_splits/`, and `07_results/`. Raw or processed data are not bundled automatically because redistribution depends on the terms of the original data sources.

### WebTEST exact-neighbor cache

`webtest_build_similarity_graphs.py` requires the exact Top-50 reference computation cache that was used to construct the final WebTEST Top-10 graphs. For public use, place that cache under:

`external_cache/webtest_reference/`

with the expected source table, edge parquet, and QC JSON hierarchy documented in the script. The script verifies the cache specification before use. For a given query, it falls back to exact current-training-set similarity calculations when the verified Top-50 cache cannot guarantee recovery of the required Top-10 neighbors. No approximate nearest-neighbor method is used. The reference cache itself is not included in this code-only archive.

## Environment

Core dependencies are listed in `environment.yml` and `requirements.txt`. GPU execution is optional but strongly recommended for GraphSAGE training, especially for the large WebTEST evaluation.

## Reproducibility notes

The released code preserves the final scientific model definitions and evaluation logic. Development-only filenames and local computer paths were removed. A manifest with SHA256 hashes of the public files is included.
