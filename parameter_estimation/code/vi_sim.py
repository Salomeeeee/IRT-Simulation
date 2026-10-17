"""
IRT Simulation Study: Fit IRT models to simulated response data using py-irt.

For each condition (benchmark x IRT model x sample size x replication):
  1. Load the simulated response matrix
  2. Subset to top-N rows for the given sample size
  3. Fit the IRT model using py-irt (variational inference)
  4. Extract and save:
     - Point estimates for theta (ability) and item parameters
     - Standard errors (from variational posterior)
     - 95% credible intervals

Usage:
  python run_simulation.py --benchmark gsm8k --model 1pl --sample_size 100 --rep 1
  python run_simulation.py --benchmark gsm8k --model 2pl  # runs all sample sizes & reps
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyro
import torch
from tqdm import tqdm

from py_irt.config import IrtConfig
from py_irt.dataset import Dataset
from py_irt.training import IrtModelTrainer


# ============================================================================
# Configuration
# ============================================================================
BENCHMARKS = ["gsm8k", "winogrande", "arc", "hellaswag", "mmlu", "truthfulqa"]
MODELS = ["1pl", "2pl", "3pl"]  # py-irt uses "4pl" internally for 3pl-like; adjust if needed
SAMPLE_SIZES = [30, 100, 180, 400, 1000]
N_REPS = 50

# Paths — adjust these to match your directory structure
DATA_DIR = Path("/path/to/input/data")            # where resp_sim_repX.csv files live
OUTPUT_DIR = DATA_DIR / Path("results")           # where to save estimation results


# ============================================================================
# Step 1: Convert response matrix (CSV) to py-irt's expected jsonlines format
# ============================================================================
def response_matrix_to_jsonlines(resp_df, output_path):
    """Convert an N_models x N_items binary response matrix to jsonlines.

    py-irt expects a jsonlines file where each line is:
      {"subject_id": "sim_1", "responses": {"item_1": 1, "item_3": 0, ...}}

    Args:
        resp_df: DataFrame with sim_id as index, item numbers as columns, values 0/1
        output_path: path to write the .jsonlines file
    """
    item_cols = [c for c in resp_df.columns if c != "sim_id"]

    records = []
    for _, row in resp_df.iterrows():
        subject_id = str(row["sim_id"]) if "sim_id" in resp_df.columns else f"model_{_}"
        record = {
            "subject_id": subject_id,
            "responses": {f"item_{c}": int(row[c]) for c in item_cols},
        }
        records.append(record)

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# ============================================================================
# Step 2: Fit IRT model and extract parameters
# ============================================================================
def fit_irt_model(jsonlines_path, model_type, num_epochs=2000, lr=0.1, device="cpu"):
    """Fit an IRT model using py-irt and return the trainer object.

    Args:
        jsonlines_path: path to the jsonlines data file
        model_type: one of "1pl", "2pl", "4pl" (py-irt's name for models)
        num_epochs: number of SVI training epochs
        lr: learning rate for Adam optimizer
        device: "cpu" or "cuda"

    Returns:
        trainer: the IrtModelTrainer object (contains the fitted model)
    """
    # Clear Pyro's parameter store to avoid contamination across runs
    pyro.clear_param_store()

    config = IrtConfig(
        model_type=model_type,
        log_every=500,
        num_epochs=num_epochs,
        priors="hierarchical",
        seed=42,
        deterministic=True,
        lr=lr,
        device=device,
    )

    trainer = IrtModelTrainer(
        data_path=Path(jsonlines_path),
        config=config,
        verbose=True,
    )
    trainer.train()

    return trainer


# ============================================================================
# Step 3: Extract parameter estimates, SEs, and credible intervals
# ============================================================================
def extract_results(trainer, model_type):
    """Extract point estimates, standard errors, and 95% CIs from the
    variational posterior stored in Pyro's param store.

    In variational inference, each latent variable has a variational
    distribution q(z) = Normal(loc, scale). The loc is the point estimate,
    the scale is the standard error, and the 95% CI is loc +/- 1.96*scale.

    Returns:
        dict with keys: theta_est, theta_se, theta_ci_lower, theta_ci_upper,
                        diff_est, diff_se, diff_ci_lower, diff_ci_upper,
                        (and disc_* for 2pl+, lambdas for 4pl)
    """
    results = {}

    # --- Theta (ability) ---
    theta_loc = pyro.param("loc_ability").detach().cpu().numpy()
    theta_scale = pyro.param("scale_ability").detach().cpu().numpy()
    results["theta_est"] = theta_loc.tolist()
    results["theta_se"] = theta_scale.tolist()
    results["theta_ci_lower"] = (theta_loc - 1.96 * theta_scale).tolist()
    results["theta_ci_upper"] = (theta_loc + 1.96 * theta_scale).tolist()

    # --- Difficulty (b) ---
    diff_loc = pyro.param("loc_diff").detach().cpu().numpy()
    diff_scale = pyro.param("scale_diff").detach().cpu().numpy()
    results["diff_est"] = diff_loc.tolist()
    results["diff_se"] = diff_scale.tolist()
    results["diff_ci_lower"] = (diff_loc - 1.96 * diff_scale).tolist()
    results["diff_ci_upper"] = (diff_loc + 1.96 * diff_scale).tolist()

    # --- Discrimination (gamma) for 2pl and above ---
    if model_type in ("2pl", "3pl", "4pl"):
        # disc is LogNormal in the model, so loc_disc is the log-scale mean
        # and scale_disc is the log-scale SD.
        # Try 2pl naming first, fall back to 4pl naming
        try:
            disc_log_loc = pyro.param("loc_slope").detach().cpu().numpy()
            disc_log_scale = pyro.param("scale_slope").detach().cpu().numpy()
        except KeyError:
            disc_log_loc = pyro.param("loc_disc").detach().cpu().numpy()
            disc_log_scale = pyro.param("scale_disc").detach().cpu().numpy()

        # Point estimate on original scale: median of LogNormal = exp(mu)
        disc_est = np.exp(disc_log_loc)

        # For LogNormal, the 95% CI on original scale:
        #   [exp(mu - 1.96*sigma), exp(mu + 1.96*sigma)]
        disc_ci_lower = np.exp(disc_log_loc - 1.96 * disc_log_scale)
        disc_ci_upper = np.exp(disc_log_loc + 1.96 * disc_log_scale)

        # SE on original scale (delta method approximation):
        #   SE ≈ exp(mu) * sigma  (first-order approximation)
        disc_se = disc_est * disc_log_scale

        results["disc_est"] = disc_est.tolist()
        results["disc_se"] = disc_se.tolist()
        results["disc_ci_lower"] = disc_ci_lower.tolist()
        results["disc_ci_upper"] = disc_ci_upper.tolist()

    # --- Lambdas (upper asymptote / slipping) for 4pl ---
    # These are pyro.param (point estimates), no posterior uncertainty
    if model_type in ("3pl", "4pl"):
        try:
            lambdas = pyro.param("lambdas").detach().cpu().numpy()
            results["guessing_est"] = lambdas.tolist()
            # No SE/CI for pyro.param — they are point estimates
        except Exception:
            pass

    # Also save the exported params from py-irt's own method
    results["export"] = trainer.irt_model.export()

    return results


# ============================================================================
# Step 4: Run one condition
# ============================================================================
def run_one_condition(benchmark, model_type, sample_size, rep, device="cpu"):
    """Run a single simulation condition.

    Args:
        benchmark: e.g., "gsm8k"
        model_type: e.g., "1pl", "2pl", "3pl"
        sample_size: e.g., 30, 100, 180, 400, 1000
        rep: replication number (1-indexed)
        device: "cpu" or "cuda"
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: {benchmark} | Model: {model_type} | N: {sample_size} | Rep: {rep}")
    print(f"{'='*60}")

    # --- Load response matrix ---
    resp_path = DATA_DIR / benchmark / model_type / f"resp_sim_rep{rep}.csv"
    if not resp_path.exists():
        print(f"  [SKIP] Response file not found: {resp_path}")
        return

    resp_df = pd.read_csv(resp_path)  # has header row with item numbers, first col is sim_id
    print(f"  Full response matrix: {resp_df.shape[0]} models x {resp_df.shape[1] - 1} items")

    # --- Subset to sample size (top-N rows) ---
    resp_subset = resp_df.iloc[:sample_size, :]
    print(f"  Subsetted to: {resp_subset.shape[0]} models x {resp_subset.shape[1] - 1} items")

    # --- Convert to jsonlines ---
    out_dir = OUTPUT_DIR / benchmark / model_type / f"N={sample_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonlines_path = out_dir / f"data_rep{rep}.jsonl"
    response_matrix_to_jsonlines(resp_subset, jsonlines_path)

    # --- Fit model ---
    try:
        trainer = fit_irt_model(
            jsonlines_path=str(jsonlines_path),
            model_type=model_type,
            num_epochs=2000,
            device=device,
        )
    except Exception as e:
        print(f"  [ERROR] Model fitting failed: {e}")
        # Save error log
        with open(out_dir / "error.txt", "w") as f:
            f.write(str(e))
        return

    # --- Extract and save results ---
    results = extract_results(trainer, model_type)

    # Save original item IDs and subject IDs for mapping back
    item_cols = [c for c in resp_subset.columns if c != "sim_id"]
    subject_ids = resp_subset["sim_id"].tolist()

    # Add metadata
    results["metadata"] = {
        "benchmark": benchmark,
        "model_type": model_type,
        "sample_size": sample_size,
        "replication": rep,
        "n_items": len(item_cols),
    }

    # Save as JSON
    results_path = out_dir / f"est_rep{rep}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save theta and item params as CSVs for easier analysis
    '''
    theta_df = pd.DataFrame({
        "subject_id": subject_ids,
        "theta_est": results["theta_est"],
        "theta_se": results["theta_se"],
        "theta_ci_lower": results["theta_ci_lower"],
        "theta_ci_upper": results["theta_ci_upper"],
    })
    theta_df.to_csv(out_dir / "theta_estimates.csv", index=False)

    item_df = pd.DataFrame({"item_id": item_cols,
                             "diff_est": results["diff_est"],
                             "diff_se": results["diff_se"],
                             "diff_ci_lower": results["diff_ci_lower"],
                             "diff_ci_upper": results["diff_ci_upper"]})
    if "disc_est" in results:
        item_df["disc_est"] = results["disc_est"]
        item_df["disc_se"] = results["disc_se"]
        item_df["disc_ci_lower"] = results["disc_ci_lower"]
        item_df["disc_ci_upper"] = results["disc_ci_upper"]
    if "lambdas_est" in results:
        item_df["lambdas_est"] = results["lambdas_est"]
    if "guessing_est" in results:
        item_df["guessing_est"] = results["guessing_est"]
    item_df.to_csv(out_dir / "item_estimates.csv", index=False)
    '''
    os.remove(jsonlines_path)
    print(f"  [DONE] Results saved to {out_dir}")


# ============================================================================
# Step 5: Main — run all or specific conditions
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="IRT Simulation Study")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Benchmark name (e.g., gsm8k). If not set, runs all.")
    parser.add_argument("--model", type=str, default=None,
                        help="IRT model (1pl, 2pl, 3pl). If not set, runs all.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Sample size (30, 100, 180, 400, 1000). If not set, runs all.")
    parser.add_argument("--rep", type=int, default=None,
                        help="Replication number (1-50). If not set, runs all.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: 'cpu' or 'cuda'")
    args = parser.parse_args()

    benchmarks = [args.benchmark] if args.benchmark else BENCHMARKS
    models = [args.model] if args.model else MODELS
    sample_sizes = [args.sample_size] if args.sample_size else SAMPLE_SIZES
    reps = [args.rep] if args.rep else list(range(1, N_REPS + 1))

    total = len(benchmarks) * len(models) * len(sample_sizes) * len(reps)
    print(f"Total conditions to run: {total}")

    count = 0
    for benchmark in benchmarks:
        for model_type in models:
            for sample_size in sample_sizes:
                for rep in tqdm(reps, desc=f'{benchmark}-{model_type}-N={sample_size}'):
                    count += 1
                    print(f"\n[{count}/{total}]")
                    run_one_condition(benchmark, model_type, sample_size, rep, args.device)

    print(f"\nAll done! {count} conditions completed.")


if __name__ == "__main__":
    # main()
    trainer = fit_irt_model(
        jsonlines_path='/path/to/input/data',
        model_type='3pl',
        num_epochs=2000,
        device='cpu',
    )