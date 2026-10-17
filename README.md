This repository contains the code and example data used in our study.

## 1. Simulate responses

`simulate_responses/` contains code for preprocessing the benchmark data, estimating the generating IRT parameters, and simulating response matrices.

- `metabench_preproc_csv/` — Preprocessed empirical response data for the six OpenLLM benchmarks.
- `params_pyirt_metabench/` — Estimated item and model parameters used as the true parameters in the simulation.
- `generate_responses.py` — Generates the replicated response matrices and true capability parameters.

## 2. Estimate parameters

`parameter_estimation/` contains the estimation code and example inputs and outputs.

- `mml_em_sim.py`, `vi_sim.py`, `psn_irt_train.py`, `mcmc_sim.R` — Scripts for the four estimators, which apply to all IRT model types and sample sizes.
- `mml_em_theta.py` - Numerically stable, log-space EAP estimation for recovering MML-EM theta estimates lost to numerical overflow or underflow.
- `tqa_1pl/`, `tqa_2pl/`, `tqa_3pl/` — Sample data for TruthfulQA across all IRT model types and sample sizes, including true parameters, input response matrices, and simulation results from replication #50.

## 3. Summarize results

`result_evaluation/` contains code for computing evaluation metrics, summarizing run outcomes and parameter recovery, and producing the reported figures.

- `metrics.R`, `cor_tiny_FI.R` — Compute the evaluation metrics reported in the paper.
- `Result_summary.Rmd` - Code for raw summary figures and tables.
- `ranking_failure_additional_analysis.R`, `shortform_additional_analysis.R` - Code for additional analysis on ranking recovery and short form lengths.
- `summary_table_all_runs/` — Run-level and condition-level result tables used by the summary and plotting code.

## Citation
If you find this study, code, or data useful, please cite:
```bibtex
@inproceedings{
jiang2026irt,
title={Can We Trust Item Response Theory for {AI} Evaluation?},
author={Jiang, Han and Kwon, Sunbeom and Luo, Jinwen and Xiao, Ziang and Zhang, Susu},
booktitle={The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
year={2026},
url={https://openreview.net/forum?id=ZKNcOHJwjy}
}
