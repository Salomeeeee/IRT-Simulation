library(brms)
library(tidyverse)
library(cmdstanr)

# Root directory containing parameter_estimation, result_evaluation, and simulate_responses.
setwd("/root/path")

# Response data are created by simulate_responses/generate_responses.py.
analysis_dir <- "analysis"
output_dir <- "parameter_estimation/mcmc_output"

# Read exact conditions passed from SLURM Job Array
args <- commandArgs(trailingOnly = TRUE)
benchmark <- as.character(args[1])
irt_model <- as.character(args[2])
N <- as.numeric(args[3])
r <- as.character(args[4]) 


out_dir <- file.path(output_dir, benchmark, irt_model, paste0("N", N))
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

log_file <- file.path(out_dir, paste0("log_rep", r, ".txt"))


tryCatch({
  withCallingHandlers({
    
    # Load Data
    resp_path <- file.path(
      analysis_dir,
      benchmark,
      irt_model,
      paste0("resp_sim_rep", r, ".csv")
    )
    if (!file.exists(resp_path)) stop(paste("File missing:", resp_path))
    
    resp_full <- read.csv(resp_path, row.names = 1)
    resp <- resp_full[1:N, ]
    df_long <- as.data.frame(resp) %>%
      rownames_to_column("id") %>%
      pivot_longer(cols = -id, names_to = "item", values_to = "response")
    
    ## Fit IRT models ----
    fit <- NULL
    theta_est <- NULL
    item_est <- NULL
    
    start <- Sys.time()
    
    if (irt_model == "1pl") {
      fit <- brm(
        formula = bf(response ~ 1 + (1 | item) + (1 | id)),
        data = df_long, family = brmsfamily("bernoulli", "logit"),
        prior = c(prior("normal(0, 1)", class = "sd", group = "id"),
                  prior("normal(0, 3)", class = "sd", group = "item")),
        chains = 1, iter = 2000, warmup = 1000,
        backend = "cmdstanr", threads = threading(4), cores = 1, init = 0
      )
      theta_est <- ranef(fit)$id[, c("Estimate", "Q2.5", "Q97.5"), "Intercept"]
      item_est <- data.frame(b = -1 * coef(fit)$item[, "Estimate", "Intercept"])
    }
    
    if (irt_model == "2pl") {
      fit <- brm(
        formula = bf(response ~ exp(logalpha) * eta,
                     eta ~ 1 + (1 |i| item) + (1 | id),
                     logalpha ~ 1 + (1 |i| item),
                     nl = TRUE),
        data = df_long, family = brmsfamily("bernoulli", "logit"),
        prior = c(prior("normal(0, 5)", class = "b", nlpar = "eta"),
                  prior("normal(0, 1)", class = "b", nlpar = "logalpha"),
                  prior("constant(1)", class = "sd", group = "id", nlpar = "eta"),
                  prior("normal(0, 3)", class = "sd", group = "item", nlpar = "eta"),
                  prior("normal(0, 1)", class = "sd", group = "item", nlpar = "logalpha")),
        chains = 1, iter = 2000, warmup = 1000,
        backend = "cmdstanr", threads = threading(4), cores = 1, init = 0
      )
      theta_est <- ranef(fit)$id[, c("Estimate", "Q2.5", "Q97.5"), "eta_Intercept"]
      item_est <- data.frame(
        b = -1 * coef(fit)$item[, "Estimate", "eta_Intercept"],
        a = exp(coef(fit)$item[, "Estimate", "logalpha_Intercept"])
      )
    }
    
    if (irt_model == "3pl") {
      fit <- brm(
        formula = bf(response ~ gamma + (1 - gamma) * inv_logit(exp(logalpha) * eta),
                     eta ~ 1 + (1 |i| item) + (1 | id),
                     logalpha ~ 1 + (1 |i| item),
                     logitgamma ~ 1 + (1 |i| item),
                     nlf(gamma ~ inv_logit(logitgamma)),
                     nl = TRUE),
        data = df_long, family = brmsfamily("bernoulli", "identity"),
        prior = c(prior("normal(0, 5)", class = "b", nlpar = "eta"),
                  prior("normal(0, 1)", class = "b", nlpar = "logalpha"),
                  prior("logistic(0, 1)", class = "b", nlpar = "logitgamma"),
                  prior("constant(1)", class = "sd", group = "id", nlpar = "eta"),
                  prior("normal(0, 3)", class = "sd", group = "item", nlpar = "eta"),
                  prior("normal(0, 1)", class = "sd", group = "item", nlpar = "logalpha"),
                  prior("normal(0, 1)", class = "sd", group = "item", nlpar = "logitgamma")),
        chains = 1, iter = 2000, warmup = 1000,
        backend = "cmdstanr", threads = threading(4), cores = 1, init = 0
      )
      theta_est <- ranef(fit)$id[, c("Estimate", "Q2.5", "Q97.5"), "eta_Intercept"]
      item_est <- data.frame(
        b = -1 * coef(fit)$item[, "Estimate", "eta_Intercept"],
        a = exp(coef(fit)$item[, "Estimate", "logalpha_Intercept"]),
        c = plogis(coef(fit)$item[, "Estimate", "logitgamma_Intercept"])
      )
    }
    
    end <- Sys.time()
    run_time <- as.numeric(end - start, units = "secs")
    
    
    write.csv(theta_est, file.path(out_dir, paste0("theta_est_rep", r, ".csv")))
    write.csv(item_est, file.path(out_dir, paste0("itempars_est_rep", r, ".csv")))
    write.csv(run_time, file.path(out_dir, paste0("run_time_rep", r, ".csv")))
    
  }, warning = function(w) {
    write(paste(Sys.time(), "- Warning:", w$message), file = log_file, append = TRUE)
    invokeRestart("muffleWarning") 
  })
  
}, error = function(e) {
  write(paste(Sys.time(), "- Error:", e$message), file = log_file, append = TRUE)
})
