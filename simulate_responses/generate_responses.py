from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PARAMS_DIR = ROOT / "params_pyirt_metabench"
ANALYSIS_DIR = ROOT / "analysis"
BENCHMARKS = ["arc", "gsm8k", "hellaswag", "mmlu", "truthfulqa", "winogrande"]
MODELS = ["1pl", "2pl", "3pl"]
N_THETA = 5000
N_REPS = 50
BASE_SEED = 20260423


def fourpl_probability(itempars, theta):
    b = itempars[:, 0]
    a = itempars[:, 1]
    c = itempars[:, 2]
    d = itempars[:, 3]
    logits = a[None, :] * (theta[:, None] - b[None, :])
    logits = np.clip(logits, -40.0, 40.0)
    return c[None, :] + (d[None, :] - c[None, :]) / (1.0 + np.exp(-logits))


def sample_response(probability, rng):
    return (rng.random(probability.shape) < probability).astype(np.int8)


def rep_seed(benchmark_ix, model_ix, rep):
    return BASE_SEED + benchmark_ix * 1000 + model_ix * 100 + rep


def benchmark_model_paths(benchmark, model_name):
    params_dir = PARAMS_DIR / benchmark / model_name
    analysis_dir = ANALYSIS_DIR / benchmark / model_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return params_dir, analysis_dir


def main():
    verification_rows = []

    for benchmark_ix, benchmark in enumerate(BENCHMARKS):
        for model_ix, model_name in enumerate(MODELS):
            params_dir, out_dir = benchmark_model_paths(benchmark, model_name)

            theta_df = pd.read_csv(params_dir / "theta.csv")
            itempars_df = pd.read_csv(params_dir / "itempars.csv")

            theta_hat = theta_df["theta"].to_numpy()
            itempars = itempars_df[["b", "a", "c", "d"]].to_numpy()
            item_ids = itempars_df["itemid"].astype(str).to_numpy()
            sim_ids = [f"sim_{i}" for i in range(1, N_THETA + 1)]

            theta_true = np.empty((N_THETA, N_REPS), dtype=float)
            rep1_response = None

            for rep in range(1, N_REPS + 1):
                rng = np.random.default_rng(rep_seed(benchmark_ix, model_ix, rep))
                theta_rep = rng.choice(theta_hat, size=N_THETA, replace=True)
                theta_true[:, rep - 1] = theta_rep

                probability = fourpl_probability(itempars, theta_rep)
                response = sample_response(probability, rng)

                response_df = pd.DataFrame(response, index=sim_ids, columns=item_ids)
                response_df.to_csv(out_dir / f"resp_sim_rep{rep}.csv", index=True, index_label="sim_id")
                if rep == 1:
                    rep1_response = response

            theta_df_out = pd.DataFrame(
                theta_true,
                index=sim_ids,
                columns=[f"rep{rep}" for rep in range(1, N_REPS + 1)],
            )
            theta_df_out.to_csv(out_dir / "theta_true_matrix.csv", index=True, index_label="sim_id")

            rep1_total_score = rep1_response.mean(axis=1)
            rep1_theta = theta_true[:, 0]
            rep1_corr = float(np.corrcoef(rep1_total_score, rep1_theta)[0, 1])

            verification_rows.append({
                "benchmark": benchmark,
                "model": model_name,
                "n_models_sim": N_THETA,
                "n_items": len(item_ids),
                "rep_checked": 1,
                "score_theta_corr": rep1_corr,
            })

            print(
                f"{benchmark} {model_name}: wrote {N_REPS} response files with shape "
                f"({N_THETA}, {len(item_ids)}) and rep1 corr={rep1_corr:.4f}"
            )

    verification_df = pd.DataFrame(verification_rows)
    verification_df.to_csv(ANALYSIS_DIR / "simulation_verification.csv", index=False)
    print(f"saved verification to {ANALYSIS_DIR / 'simulation_verification.csv'}")


if __name__ == "__main__":
    main()
