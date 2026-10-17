import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import pyro
import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYIRT_ROOT = os.path.join(ROOT, "original_repos", "py-irt")
sys.path.insert(0, PYIRT_ROOT)

from py_irt.config import IrtConfig  # noqa: E402
from py_irt.dataset import Dataset  # noqa: E402
from py_irt.training import IrtModelTrainer  # noqa: E402


BENCHMARKS = ["arc", "gsm8k", "hellaswag", "mmlu", "truthfulqa", "winogrande"]
MODELS = ["1pl", "2pl", "3pl"]


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 2:
        return np.nan
    if np.std(x[keep]) == 0 or np.std(y[keep]) == 0:
        return np.nan
    return float(np.corrcoef(x[keep], y[keep])[0, 1])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def read_response(path):
    df = pd.read_csv(path)
    model_ids = df.iloc[:, 0].astype(str).to_numpy()
    item_ids = df.columns[1:].astype(str).to_numpy()
    response = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    df = df.rename(columns={df.columns[0]: "model_id"})
    return df, model_ids, item_ids, response


def fit_pyirt(df, model_name, args):
    pyro.set_rng_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = Dataset.from_pandas(df, subject_column="model_id")
    config = IrtConfig(
        model_type=model_name,
        priors="vague",
        epochs=args.epochs,
        lr=args.lr,
        lr_decay=args.lr_decay,
        log_every=max(1, args.log_every),
    )
    trainer = IrtModelTrainer(
        data_path=None,
        config=config,
        dataset=dataset,
        verbose=args.verbose,
    )
    start = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer.train(epochs=args.epochs, device=args.device)
    seconds = time.time() - start
    warning_text = " | ".join(str(w.message) for w in caught)
    return trainer.last_params, seconds, warning_text


def raw_arrays(params, model_ids, item_ids, model_name):
    ix_to_subject = {int(k): v for k, v in params["subject_ids"].items()}
    ix_to_item = {int(k): v for k, v in params["item_ids"].items()}

    theta_by_id = {
        ix_to_subject[i]: params["ability"][i]
        for i in range(len(params["ability"]))
    }
    b_by_id = {
        ix_to_item[i]: params["diff"][i]
        for i in range(len(params["diff"]))
    }

    theta = np.array([theta_by_id[x] for x in model_ids], dtype=float)
    b = np.array([b_by_id[x] for x in item_ids], dtype=float)

    if model_name == "1pl":
        a = np.ones(len(item_ids))
        c = np.zeros(len(item_ids))
    else:
        a_by_id = {
            ix_to_item[i]: params["disc"][i]
            for i in range(len(params["disc"]))
        }
        a = np.array([a_by_id[x] for x in item_ids], dtype=float)
        c = np.zeros(len(item_ids))

    if model_name == "3pl":
        c_by_id = {
            ix_to_item[i]: params["lambdas"][i]
            for i in range(len(params["lambdas"]))
        }
        c = np.array([c_by_id[x] for x in item_ids], dtype=float)

    d = np.ones(len(item_ids))
    return theta, b, a, c, d


def align_and_standardize(theta, b, a, c, d, response):
    total_score = response.mean(axis=1)
    if corr(theta, total_score) < 0:
        theta = -theta
        b = -b
        a = -a

    theta_mean = float(np.mean(theta))
    theta_sd = float(np.std(theta))
    theta_std = (theta - theta_mean) / theta_sd
    b_std = (b - theta_mean) / theta_sd
    a_std = a * theta_sd
    return theta_std, b_std, a_std, c, d, theta_mean, theta_sd


def predict(theta, b, a, c, d):
    eta = a[None, :] * (theta[:, None] - b[None, :])
    return c[None, :] + (d[None, :] - c[None, :]) * sigmoid(eta)


def export_fit(benchmark, model_name, params, seconds, warning_text, model_ids, item_ids, response, out_root):
    theta_raw, b_raw, a_raw, c_raw, d_raw = raw_arrays(params, model_ids, item_ids, model_name)
    theta, b, a, c, d, raw_aligned_mean, raw_aligned_sd = align_and_standardize(
        theta_raw, b_raw, a_raw, c_raw, d_raw, response
    )
    p_hat = predict(theta, b, a, c, d)

    out_dir = os.path.join(out_root, benchmark, model_name)
    os.makedirs(out_dir, exist_ok=True)

    pd.DataFrame({
        "model_id": model_ids,
        "theta": theta,
    }).to_csv(os.path.join(out_dir, "theta.csv"), index=False)

    pd.DataFrame({
        "itemid": item_ids,
        "b": b,
        "a": a,
        "c": c,
        "d": d,
    }).to_csv(os.path.join(out_dir, "itempars.csv"), index=False)

    with open(os.path.join(out_dir, "pyirt_params_raw.json"), "w") as f:
        json.dump(params, f)

    total_score = response.mean(axis=1)
    item_mean = response.mean(axis=0)
    validation = pd.DataFrame([{
        "benchmark": benchmark,
        "model": model_name,
        "n_models": response.shape[0],
        "n_items": response.shape[1],
        "seconds": seconds,
        "status": "ok",
        "warnings": warning_text,
        "raw_aligned_theta_mean": raw_aligned_mean,
        "raw_aligned_theta_sd": raw_aligned_sd,
        "theta_mean": float(np.mean(theta)),
        "theta_sd": float(np.std(theta)),
        "a_mean": float(np.mean(a)),
        "a_sd": float(np.std(a)),
        "c_mean": float(np.mean(c)),
        "cor_theta_total_score": corr(theta, total_score),
        "cor_b_item_mean": corr(b, item_mean),
        "cor_a_item_mean": corr(a, item_mean),
        "cor_c_item_mean": corr(c, item_mean),
        "cor_phat_item_mean": corr(p_hat.mean(axis=0), item_mean),
        "cor_phat_total_score": corr(p_hat.mean(axis=1), total_score),
    }])
    validation.to_csv(os.path.join(out_dir, "validation.csv"), index=False)
    return validation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "metabench_preproc_csv"))
    parser.add_argument("--out-root", default=os.path.join(ROOT, "params_pyirt_metabench"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-decay", type=float, default=0.9999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    all_validation = []
    for benchmark in args.benchmarks:
        response_file = os.path.join(args.data_dir, benchmark, "response.csv")
        df, model_ids, item_ids, response = read_response(response_file)
        print(f"{benchmark}: response shape={response.shape}", flush=True)
        for model_name in args.models:
            print(f"{benchmark} {model_name}: fitting py-irt, vague priors, {args.epochs} epochs", flush=True)
            try:
                params, seconds, warning_text = fit_pyirt(df, model_name, args)
                validation = export_fit(
                    benchmark, model_name, params, seconds, warning_text,
                    model_ids, item_ids, response, args.out_root
                )
            except Exception as exc:
                out_dir = os.path.join(args.out_root, benchmark, model_name)
                os.makedirs(out_dir, exist_ok=True)
                validation = pd.DataFrame([{
                    "benchmark": benchmark,
                    "model": model_name,
                    "n_models": response.shape[0],
                    "n_items": response.shape[1],
                    "seconds": np.nan,
                    "status": "error",
                    "warnings": str(exc),
                }])
                validation.to_csv(os.path.join(out_dir, "validation.csv"), index=False)
                print(f"{benchmark} {model_name}: ERROR {exc}", flush=True)
            print(validation.to_string(index=False), flush=True)
            all_validation.append(validation)

    summary = pd.concat(all_validation, ignore_index=True)
    os.makedirs(args.out_root, exist_ok=True)
    summary.to_csv(os.path.join(args.out_root, "validation_summary.csv"), index=False)
    print(f"Saved summary to {os.path.join(args.out_root, 'validation_summary.csv')}")


if __name__ == "__main__":
    main()
