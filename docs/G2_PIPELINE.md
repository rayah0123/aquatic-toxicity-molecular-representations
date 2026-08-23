# G2 pipeline

The inter-chemical model uses six physicochemical node descriptors and a structural-similarity graph defined from Count Morgan radius-4 fingerprints and exact Tanimoto similarity. The fixed graph rule selects the top 10 positive-similarity training neighbors.

Recommended order for the experimental Daphnia evaluation:
1. `g2_build_assets.py`
2. `g2_run_multisplit.py` using `g2_train_validation.py`
3. `g2_evaluate_test.py`

The descriptor-only GraphSAGE condition uses the same six node descriptors and model architecture with no inter-chemical message-passing edges.
