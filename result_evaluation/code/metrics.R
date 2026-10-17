library(tidyverse)

RERUN_CONDITIONS <- readLines("rerun.txt") |> trimws()
RERUN_CONDITIONS <- RERUN_CONDITIONS[RERUN_CONDITIONS != ""]

setwd("/home/jhu/hjiang66/scratch_zxiao25/jh_data/irt_simulation")

BASE_EVAL  <- "em_outputs"
BASE_TRUE  <- "ground_truth"   # contains subfolders for all 6 benchmarks
OUT_DIR    <- "summaries"
dir.create(file.path(OUT_DIR, "item_model_lv_summaries"), showWarnings = FALSE)

# EM filenames are zero-padded (rep01); others are not (rep1)
rep_str <- function(method, rep) if (method == "EM") sprintf("%02d", rep) else as.character(rep)

# function for calculating P(correct )
p1_3pl <- function(a, b, c, theta) {
  eta <- sweep(outer(theta, b, "-"), 2, a, "*")
  C <- matrix(c, nrow = length(theta), ncol = length(c), byrow = TRUE)
  C + (1 - C) / (1 + exp(-eta))
}

# fisher information
FI_3pl <- function(p1_mat, a, c) {
  A <- matrix(a, nrow = nrow(p1_mat), ncol = length(a), byrow = TRUE)
  C <- matrix(c, nrow = nrow(p1_mat), ncol = length(c), byrow = TRUE)
  FI_mat <- A^2 * ((p1_mat - C)^2 / (1 - C)^2) * ((1 - p1_mat) / p1_mat)
  FI_mat
}

process_method <- function(method) {
  out_file <- file.path(OUT_DIR, paste0(method, "_param_recovery_4500_new.csv"))
  cat("benchmark,irt,sample_size,rep,cor_b,cor_a,diff_c,cor_theta_pearson,cor_theta_kendall,p_diff_ave,p_diff_models,p_diff_items\n",
      file = out_file)

  benchmarks <- list.dirs(file.path(BASE_EVAL), recursive = FALSE, full.names = FALSE)

  for (bench in benchmarks) {
    irts <- list.dirs(file.path(BASE_EVAL, bench), recursive = FALSE, full.names = FALSE)

    for (irt in irts) {
      true_items <- read_csv(file.path(BASE_TRUE, bench, irt, "itempars.csv"),
                            show_col_types = FALSE)
      true_theta <- read_csv(file.path(BASE_TRUE, bench, irt, "theta_true_matrix.csv"),
                            show_col_types = FALSE)
      n_dirs <- list.dirs(file.path(BASE_EVAL, bench, irt),
                          recursive = FALSE, full.names = FALSE)

      for (n_dir in n_dirs) {
        N <- as.integer(str_extract(n_dir, "\\d+"))
        if (
            bench == "arc" ||
            (bench == "gsm8k" && irt == "1pl") ||
            (bench == "gsm8k" && irt == "2pl" && N <= 400)
          ) next
          
        condition <- paste0(bench, "/", irt, "/N", N)
        use_new_theta <- condition %in% RERUN_CONDITIONS

        message(
          "Running: ", condition,
          if (use_new_theta) " (new theta)" else ""
        )

        run_dir <- file.path(BASE_EVAL, bench, irt, n_dir)
        item_files <- list.files(run_dir, pattern = "^itempars_est_rep.*\\.csv$")

        # Accumulators for per-rep columns; written out after all reps for this condition
        mat_pdiff_models <- list()
        mat_pdiff_items  <- list()
        mat_FI_item      <- list()

        for (f in item_files) {
          rep <- as.integer(str_extract(f, "\\d+"))

          theta_prefix <- if (use_new_theta) "theta_new_rep" else "theta_est_rep"

          est_items <- read_csv(file.path(run_dir, f), show_col_types = FALSE)
          est_theta <- read_csv(file.path(run_dir, paste0(theta_prefix, rep_str(method, rep), ".csv")),
                                show_col_types = FALSE)

          # Files are pre-normalized to the same row order as true parameters
          J     <- nrow(est_items)
          b_hat <- est_items$b
          a_hat <- if (irt == "1pl") rep(1, J) else est_items$a
          c_hat <- if (irt %in% c("1pl", "2pl")) rep(0, J) else est_items$c

          theta_hat  <- est_theta$theta
          theta_true <- true_theta |> slice(1:N) |> pull(paste0("rep", rep))

          # Sign alignment: simultaneous flip of theta/b/a leaves p1_hat unchanged,
          if (cor(theta_hat, theta_true, use = "complete.obs") < 0) {
            theta_hat <- -theta_hat
            b_hat     <- -b_hat
            a_hat     <- -a_hat
          }
          
          p1_true <- p1_3pl(true_items$a, true_items$b, true_items$c, theta_true)
          p1_hat  <- p1_3pl(a_hat, b_hat, c_hat, theta_hat)
          FI_hat  <- FI_3pl(p1_hat, a_hat, c_hat)

          p_diff_ave <- mean(abs(p1_true - p1_hat), na.rm = TRUE)
          p_diff_models <- mean(abs(rowMeans(p1_true - p1_hat, na.rm = TRUE)), na.rm =TRUE)
          p_diff_items <- mean(abs(colMeans(p1_true - p1_hat, na.rm = TRUE)), na.rm =TRUE)

          col <- paste0("rep", rep)
          mat_pdiff_models[[col]] <- rowMeans(p1_true - p1_hat, na.rm = TRUE)
          mat_pdiff_items[[col]]  <- colMeans(p1_true - p1_hat, na.rm = TRUE)
          mat_FI_item[[col]]      <- colSums(FI_hat)

          row <- tibble(
            benchmark         = bench,
            irt               = irt,
            sample_size       = N,
            rep               = rep,
            cor_b             = cor(b_hat, true_items$b, use = "complete.obs"),
            cor_a             = if (irt == "1pl") NA else cor(a_hat, true_items$a, use = "complete.obs"),
            diff_c            = if (irt %in% c("1pl", "2pl")) NA else mean(abs(c_hat - true_items$c)),
            cor_theta_pearson = cor(theta_hat, theta_true, use = "complete.obs"),
            cor_theta_kendall = cor(theta_hat, theta_true, method = "kendall", use = "complete.obs"),
            p_diff_ave        = p_diff_ave,
            p_diff_models     = p_diff_models,
            p_diff_items      = p_diff_items
          )

          write_csv(row, out_file, append = TRUE, col_names = FALSE)
          rm(est_items, est_theta, b_hat, a_hat, c_hat, theta_hat, theta_true,
             p1_true, p1_hat, FI_hat, p_diff_ave, row, p_diff_models, p_diff_items)
          gc()
        }

        # Write per-condition matrices (one file per bench × irt × N)
        if(length(mat_FI_item)>0){
          prefix <- file.path(OUT_DIR, "item_model_lv_summaries", paste0(method, "_", bench, "_", irt, "_N", N))
          write_csv(as.data.frame(mat_pdiff_models), paste0(prefix, "_pdiff_models.csv"))
          write_csv(as.data.frame(mat_pdiff_items),  paste0(prefix, "_pdiff_items.csv"))
          write_csv(as.data.frame(mat_FI_item),      paste0(prefix, "_FI_item.csv"))
        }
        rm(mat_pdiff_models, mat_pdiff_items, mat_FI_item)

        message(glue::glue("{method} | {bench} | {irt} | {n_dir}"))
      }
    }
  }
}

walk(c('EM'), process_method)

# walk(c("EM", "MCMC", "PSN", "VI"), process_method)
