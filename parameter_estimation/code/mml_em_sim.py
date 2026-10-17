from pathlib import Path
import argparse
import time
import traceback

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
OUT_ROOT = ROOT / "EM_outputs"
ITEMTYPE_MAP = {"1pl": "Rasch", "2pl": "2PL", "3pl": "3PL"}


def fit_condition(rep_file, n_rows, itemtype, ncycles):
    start = time.time()
    status = "error"
    converged = None
    error_text = ""
    warning_text = ""

    try:
        from rpy2 import robjects
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.conversion import localconverter
        from rpy2.robjects import default_converter

        resp_df = pd.read_csv(rep_file, nrows=n_rows)
        model_ids = resp_df.iloc[:, 0].astype(str).tolist()
        resp_only = resp_df.iloc[:, 1:]
        item_ids = [str(x) for x in resp_only.columns]
        keep_mask = resp_only.nunique(dropna=True) >= 2
        fit_item_ids = [str(x) for x in resp_only.columns[keep_mask]]
        omitted_item_ids = [str(x) for x in resp_only.columns[~keep_mask]]
        resp_fit = resp_only.loc[:, keep_mask]

        if resp_fit.shape[1] == 0:
            raise RuntimeError("No estimable items remain after dropping one-response-category items")

        r = robjects.r
        with localconverter(default_converter + pandas2ri.converter):
            resp_r = robjects.conversion.py2rpy(resp_fit)

        r.assign("resp_df_py", resp_r)
        r.assign("model_ids_py", robjects.StrVector(model_ids))
        r.assign("fit_item_ids_py", robjects.StrVector(fit_item_ids))
        r.assign("itemtype_py", itemtype)
        r.assign("ncycles_py", ncycles)

        r(
            """
            warnings_vec <- character()
            fit_error <- NULL
            fit_obj <- NULL

            withCallingHandlers({
              fit_obj <- tryCatch(
                mirt::mirt(
                  data = resp_df_py,
                  model = 1,
                  itemtype = itemtype_py,
                  SE = TRUE,
                  SE.type = 'MHRM',
                  method = 'MHRM',
                  verbose = FALSE,
                  technical = list(NCYCLES = ncycles_py)
                ),
                error = function(e) {
                  fit_error <<- conditionMessage(e)
                  NULL
                }
              )
            }, warning = function(w) {
              warnings_vec <<- c(warnings_vec, conditionMessage(w))
              invokeRestart('muffleWarning')
            })
            """
        )

        fit_error = r("fit_error")
        warnings_vec = list(r("warnings_vec"))
        warning_text = " | ".join(str(x) for x in warnings_vec)
        if not bool(r("is.null(fit_error)")[0]):
            raise RuntimeError(str(fit_error[0]))

        converged = bool(r("mirt::extract.mirt(fit_obj, 'converged')")[0])

        r(
            """
            fs <- mirt::fscores(fit_obj, method = 'EAP', full.scores = TRUE, full.scores.SE = TRUE)
            fs_df <- data.frame(model_id = model_ids_py, theta = fs[,1], theta_se = fs[,2], stringsAsFactors = FALSE)

            cf <- mirt::coef(fit_obj, IRTpars = TRUE, simplify = TRUE)
            cf_se <- mirt::coef(fit_obj, IRTpars = TRUE, printSE = TRUE, simplify = FALSE)
            items <- cf$items
            get_item_se <- function(x) {
              if ('SE' %in% rownames(x)) {
                x['SE', ]
              } else {
                setNames(rep(NA_real_, ncol(x)), colnames(x))
              }
            }
            item_se_mat <- do.call(rbind, lapply(cf_se[seq_len(nrow(items))], get_item_se))
            rownames(item_se_mat) <- NULL

            if (itemtype_py == 'Rasch') {
              item_df <- data.frame(
                itemid = fit_item_ids_py,
                b = items[, 'b'],
                a = rep(1, nrow(items)),
                c = rep(0, nrow(items)),
                d = rep(1, nrow(items)),
                b_se = item_se_mat[, 'b'],
                a_se = NA_real_,
                c_se = NA_real_,
                d_se = NA_real_,
                stringsAsFactors = FALSE
              )
            } else if (itemtype_py == '2PL') {
              item_df <- data.frame(
                itemid = fit_item_ids_py,
                b = items[, 'b'],
                a = items[, 'a'],
                c = rep(0, nrow(items)),
                d = rep(1, nrow(items)),
                b_se = item_se_mat[, 'b'],
                a_se = item_se_mat[, 'a'],
                c_se = NA_real_,
                d_se = NA_real_,
                stringsAsFactors = FALSE
              )
            } else {
              item_df <- data.frame(
                itemid = fit_item_ids_py,
                b = items[, 'b'],
                a = items[, 'a'],
                c = items[, 'g'],
                d = items[, 'u'],
                b_se = item_se_mat[, 'b'],
                a_se = item_se_mat[, 'a'],
                c_se = item_se_mat[, 'g'],
                d_se = item_se_mat[, 'u'],
                stringsAsFactors = FALSE
              )
            }
            """
        )

        with localconverter(default_converter + pandas2ri.converter):
            theta_df = robjects.conversion.rpy2py(r("fs_df"))
            item_df = robjects.conversion.rpy2py(r("item_df"))

        full_item_df = pd.DataFrame({"itemid": item_ids})
        item_df["itemid"] = item_df["itemid"].astype(str)
        item_df = full_item_df.merge(item_df, on="itemid", how="left")

        status = "ok" if converged and warning_text == "" else "warning"
        return {
            "seconds": time.time() - start,
            "status": status,
            "converged": converged,
            "warning": warning_text,
            "error": error_text,
            "theta_df": theta_df,
            "item_df": item_df,
            "n_rows": resp_only.shape[0],
            "n_items": resp_only.shape[1],
            "n_items_fit": resp_fit.shape[1],
            "n_items_omitted": len(omitted_item_ids),
            "omitted_item_ids": omitted_item_ids,
        }

    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        return {
            "seconds": time.time() - start,
            "status": status,
            "converged": converged,
            "warning": warning_text,
            "error": error_text,
            "theta_df": None,
            "item_df": None,
            "n_rows": n_rows,
            "n_items": None,
            "n_items_fit": None,
            "n_items_omitted": None,
            "omitted_item_ids": [],
            "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--model", choices=["1pl", "2pl", "3pl"], required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--ncycles", type=int, default=2000)
    parser.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    args = parser.parse_args()

    benchmark = args.benchmark.lower()
    model_name = args.model.lower()
    itemtype = ITEMTYPE_MAP[model_name]
    rep_file = ANALYSIS_DIR / benchmark / model_name / f"resp_sim_rep{args.rep}.csv"
    out_dir = Path(args.out_root) / benchmark / model_name / f"N{args.sample_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rep_tag = f"_rep{args.rep:02d}"
    theta_path = out_dir / f"theta_est{rep_tag}.csv"
    item_path = out_dir / f"itempars_est{rep_tag}.csv"
    log_path = out_dir / f"run_log{rep_tag}.csv"
    info_path = out_dir / f"input_info{rep_tag}.txt"
    traceback_path = out_dir / f"traceback{rep_tag}.txt"

    with info_path.open("w") as f:
        f.write(f"input_file={rep_file}\n")
        f.write(f"benchmark={benchmark}\n")
        f.write(f"model={model_name}\n")
        f.write(f"sample_size={args.sample_size}\n")
        f.write(f"rep={args.rep}\n")
        f.write("method=MHRM\n")
        f.write(f"itemtype={itemtype}\n")
        f.write("SE=TRUE\n")
        f.write("SE.type=MHRM\n")
        f.write(f"NCYCLES={args.ncycles}\n")

    print(
        f"running benchmark={benchmark} model={model_name} sample_size={args.sample_size} rep={args.rep}",
        flush=True,
    )
    result = fit_condition(rep_file, args.sample_size, itemtype, args.ncycles)

    with info_path.open("a") as f:
        f.write(f"n_items_total={result['n_items']}\n")
        f.write(f"n_items_fit={result['n_items_fit']}\n")
        f.write(f"n_items_omitted={result['n_items_omitted']}\n")
        f.write("omitted_item_ids=" + ",".join(result["omitted_item_ids"]) + "\n")

    if result["theta_df"] is not None:
        result["theta_df"].to_csv(theta_path, index=False)
    if result["item_df"] is not None:
        result["item_df"].to_csv(item_path, index=False)
    if "traceback" in result:
        with traceback_path.open("w") as f:
            f.write(result["traceback"])
    elif traceback_path.exists():
        traceback_path.unlink()

    log_df = pd.DataFrame(
        [
            {
                "benchmark": benchmark,
                "model": model_name,
                "sample_size": args.sample_size,
                "rep": args.rep,
                "input_file": str(rep_file),
                "method": "MHRM",
                "itemtype": itemtype,
                "SE": True,
                "SE_type": "MHRM",
                "NCYCLES": args.ncycles,
                "seconds": result["seconds"],
                "status": result["status"],
                "converged": result["converged"],
                "warning": result["warning"],
                "error": result["error"],
                "n_items": result["n_items"],
                "n_items_fit": result["n_items_fit"],
                "n_items_omitted": result["n_items_omitted"],
                "omitted_item_ids": ",".join(result["omitted_item_ids"]),
                "theta_file": str(theta_path) if result["theta_df"] is not None else "",
                "item_file": str(item_path) if result["item_df"] is not None else "",
            }
        ]
    )
    log_df.to_csv(log_path, index=False)
    print(log_df.to_string(index=False))


if __name__ == "__main__":
    main()
