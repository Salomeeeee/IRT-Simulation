from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
RDS_DIR = ROOT / "original_repos" / "metabench-main" / "benchmark-data"
OUT_DIR = ROOT / "metabench_preproc_csv"

BENCHMARKS = {
    "arc": "arc-preproc.rds",
    "gsm8k": "gsm8k-preproc.rds",
    "hellaswag": "hellaswag-preproc.rds",
    "mmlu": "mmlu-preproc.rds",
    "truthfulqa": "truthfulqa-preproc.rds",
    "winogrande": "winogrande-preproc.rds",
}


def export_one(benchmark, rds_file):
    out_dir = OUT_DIR / benchmark
    out_dir.mkdir(parents=True, exist_ok=True)

    r_code = f"""
    x <- readRDS({str(RDS_DIR / rds_file)!r})
    data <- as.data.frame(x$data)
    data <- cbind(model_id = rownames(data), data)
    items <- as.data.frame(x$items)
    write.csv(data, {str(out_dir / "response.csv")!r}, row.names = FALSE)
    write.csv(items, {str(out_dir / "items.csv")!r}, row.names = FALSE)
    """
    subprocess.run(["Rscript", "-e", r_code], check=True)
    print(f"{benchmark}: wrote {out_dir / 'response.csv'}")


def main():
    for benchmark, rds_file in BENCHMARKS.items():
        export_one(benchmark, rds_file)


if __name__ == "__main__":
    main()
