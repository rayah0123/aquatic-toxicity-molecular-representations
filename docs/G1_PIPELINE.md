# G1 pipeline

The intramolecular model represents each chemical as an atom-bond graph and applies GraphSAGE to intramolecular connectivity only.

Recommended order:
1. `g1_build_graph_assets.py`
2. `g1_prepare_targets.py`
3. `g1_run_multisplit.py` using `g1_train_validation.py`
4. `g1_evaluate_test.py` for frozen test evaluation
5. `g1_external_inference.py` for independent external chemicals

The corrected validation trainer is the only trainer included in this public release.
