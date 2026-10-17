library(dplyr)
library(stringr)
library(tibble)

fi_dir <- "/home/jhu/hjiang66/scratch_zxiao25/jh_data/irt_simulation/summaries/EM_item_model_lv_summaries"
analysis_dir <- "/home/jhu/hjiang66/scratch_zxiao25/jh_data/irt_simulation"
out_file <- "/home/jhu/hjiang66/scratch_zxiao25/jh_data/irt_simulation/summaries/EM_top100_fi_score_correlations.csv"

fi_files <- list.files(
  fi_dir,
  pattern = "_FI_item\\.csv$",
  full.names = TRUE
)

if (file.exists(out_file)) {
  existing_results <- read.csv(out_file, check.names = FALSE)
} else {
  existing_results <- data.frame(
    method = character(),
    benchmark = character(),
    irt = character(),
    sample_size = integer(),
    rep = integer(),
    pearson_cor = numeric(),
    kendall_cor = numeric()
  )
}

fi_index <- list()

for (fi_file in fi_files) {
  file_name <- basename(fi_file)

  method <- str_split(file_name, "_", simplify = TRUE)[1]
  benchmark <- str_split(file_name, "_", simplify = TRUE)[2]
  irt <- str_split(file_name, "_", simplify = TRUE)[3]
  sample_size <- as.integer(str_extract(file_name, "(?<=N)\\d+"))

  fi_df <- read.csv(fi_file, check.names = FALSE)

  n_items <- floor(nrow(fi_df) * 0.10)

  for (rep_col in colnames(fi_df)) {
    rep <- as.integer(str_extract(rep_col, "\\d+$"))
    top_10pct_items <- order(fi_df[[rep_col]], decreasing = TRUE)[1:n_items]

    fi_index[[length(fi_index) + 1]] <- tibble(
      method = method,
      benchmark = benchmark,
      irt = irt,
      sample_size = sample_size,
      rep = rep,
      top_10pct_items = list(top_10pct_items)
    )
  }

  rm(fi_df, n_items)
  gc()
}

fi_index <- bind_rows(fi_index)

todo_df <- fi_index %>%
  anti_join(
    existing_results %>%
      select(method, benchmark, irt, sample_size, rep),
    by = c("method", "benchmark", "irt", "sample_size", "rep")
  )

todo_groups <- split(
  todo_df,
  list(todo_df$benchmark, todo_df$irt, todo_df$rep),
  drop = TRUE
)

for (group_name in names(todo_groups)) {
  group_df <- todo_groups[[group_name]]

  benchmark <- group_df$benchmark[1]
  irt <- group_df$irt[1]
  rep <- group_df$rep[1]

  resp_file <- file.path(
    analysis_dir,
    benchmark,
    irt,
    paste0("resp_sim_rep", rep, ".csv")
  )

  resp_df <- read.csv(resp_file, check.names = FALSE)
  resp_df <- tail(resp_df, 4000)

  item_df <- resp_df[, !(names(resp_df) %in% c("sim_id", "model_id", "id")), drop = FALSE]
  score_all_items <- rowMeans(item_df)

  for (i in seq_len(nrow(group_df))) {
    method <- group_df$method[i]
    sample_size <- group_df$sample_size[i]
    top_10pct_items <- group_df$top_10pct_items[[i]]

    score_top_10pct <- rowMeans(item_df[, top_10pct_items, drop = FALSE])

    pearson_cor <- cor(score_top_10pct, score_all_items, method = "pearson")
    kendall_cor <- cor(score_top_10pct, score_all_items, method = "kendall")

    out_row <- data.frame(
      method = method,
      benchmark = benchmark,
      irt = irt,
      sample_size = sample_size,
      rep = rep,
      pearson_cor = pearson_cor,
      kendall_cor = kendall_cor
    )

    if (!file.exists(out_file)) {
      write.csv(out_row, out_file, row.names = FALSE)
    } else {
      write.table(
        out_row,
        out_file,
        sep = ",",
        row.names = FALSE,
        col.names = FALSE,
        append = TRUE
      )
    }

    existing_results <- bind_rows(existing_results, out_row)

    rm(top_10pct_items, score_top_10pct, pearson_cor, kendall_cor, out_row)
    gc()
  }

  rm(resp_df, item_df, score_all_items)
  gc()
}
