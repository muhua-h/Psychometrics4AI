suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
})

# ============================================================
# Path handling (Rscript & RStudio; otherwise assumes WD)
# ============================================================
.path_from_this_script <- function() {
  # Rscript
  cmd_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- cmd_args[grepl("^--file=", cmd_args)]
  if (length(file_arg) == 1) return(normalizePath(sub("^--file=", "", file_arg)))
  
  # RStudio
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getActiveDocumentContext()$path, error = function(e) "")
    if (nzchar(p)) return(normalizePath(p))
  }
  
  # Fallback: known location
  return(normalizePath("multi_model_studies/correlation_significance_test/study_2_original_score/correlation_significance_test.R"))
}

.this_script <- .path_from_this_script()
.script_dir  <- dirname(.this_script)

# Project root = multi_model_studies
.project_root <- normalizePath(file.path(.script_dir, "..", ".."))

# CSV input and output paths
csv_path <- file.path(.project_root, "study_2a", "unified_analysis_results", "unified_convergent_results.csv")
out_csv  <- file.path(.script_dir, "correlation_significance_results_updated.csv")

# ============================================================
# Normalizers
# ============================================================
normalize_model <- function(x) {
  x <- str_trim(x)
  dplyr::case_when(
    x %in% c("openai_gpt_3.5_turbo_0125", "gpt-3.5", "gpt_3.5") ~ "gpt-3.5",
    x %in% c("gpt_4", "gpt-4")                                   ~ "gpt-4",
    x %in% c("gpt_4o", "gpt-4o", "openai_gpt_4o")                ~ "gpt-4o",
    x %in% c("llama", "meta_llama")                              ~ "llama",
    x %in% c("deepseek")                                         ~ "deepseek",
    TRUE ~ x
  )
}

# Map any reasonable condition spellings to exactly these four:
# "Simple Binary", "Elaborated Binary", "Expanded Format", "Likert Format"
normalize_condition <- function(x) {
  x0 <- str_squish(str_trim(x))
  # lower for matching, but return canonical title-cased strings
  xl <- tolower(x0)
  dplyr::case_when(
    xl %in% c("simple binary", "binary", "simple-binary") ~ "Simple Binary",
    xl %in% c("elaborated binary", "elaborated-binary")   ~ "Elaborated Binary",
    xl %in% c("expanded format", "expanded-format")       ~ "Expanded Format",
    
    # Likert variants commonly seen
    xl %in% c("likert format", "likert", "bfi-2 likert", "bfi-2-likert",
              "bfi2 likert", "bfi2-likert", "bfi-2 likert format",
              "bfi-2-likert-format", "bfi2 likert format", "bfi2-likert-format") ~ "Likert Format",
    
    TRUE ~ x0  # leave as-is if it already matches
  )
}

# Exact ordering you want
condition_levels <- c("Simple Binary", "Elaborated Binary", "Expanded Format", "Likert Format")
model_levels     <- c("gpt-3.5", "gpt-4", "gpt-4o", "llama", "deepseek")

# ============================================================
# Load CSV (AI correlations only)
# ============================================================
raw <- read_csv(csv_path, show_col_types = FALSE)

required_cols <- c(
  "condition", "model",
  "bfi_sim_O", "bfi_sim_C", "bfi_sim_E", "bfi_sim_A", "bfi_sim_N"
)
missing_cols <- setdiff(required_cols, colnames(raw))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing columns in CSV: %s", paste(missing_cols, collapse = ", ")))
}

# ============================================================
# Build correlations (normalize THEN factorize)
# ============================================================
tmp <- raw %>%
  mutate(
    Condition_chr = normalize_condition(condition),
    Model_chr     = normalize_model(model)
  )

# warn if any conditions still outside allowed set
unknown_conditions <- setdiff(unique(tmp$Condition_chr), condition_levels)
if (length(unknown_conditions) > 0) {
  warning(sprintf(
    "Found condition values not in expected set; leaving as-is: %s",
    paste(unknown_conditions, collapse = ", ")
  ))
}

correlations <- tmp %>%
  transmute(
    Condition = factor(Condition_chr, levels = condition_levels),
    Model     = factor(Model_chr,     levels = model_levels),
    O = bfi_sim_O,
    C = bfi_sim_C,
    E = bfi_sim_E,
    A = bfi_sim_A,
    N = bfi_sim_N
  ) %>%
  arrange(Condition, Model) %>%
  as.data.frame()

# Average across domains per row
correlations$Avg <- rowMeans(correlations[, c("O", "C", "E", "A", "N")], na.rm = TRUE)

# ============================================================
# Human reference correlations (FIXED values)
# ============================================================
human_correlations <- c(O = 0.7504, C = 0.8399, E = 0.8846, A = 0.7962, N = 0.7385)
human_avg <- mean(human_correlations)

# ============================================================
# Sample sizes (fixed; adjust if needed)
# ============================================================
n1 <- 438  # AI
n2 <- 438  # Human

# ============================================================
# Fisher's z tests (AI vs Human) + Bonferroni corrections
# ============================================================
domains <- c("O", "C", "E", "A", "N", "Avg")
results_list <- list()

perform_test <- function(ai_corr, human_corr, domain, condition, model) {
  if (is.na(ai_corr) || is.na(human_corr) ||
      abs(ai_corr) >= 1 || abs(human_corr) >= 1) {
    return(list(
      condition = condition,
      model = model,
      domain = domain,
      ai_corr = ai_corr,
      human_corr = human_corr,
      statistic = NA_real_,
      p_value = NA_real_,
      significant = FALSE
    ))
  }
  z_ai <- 0.5 * log((1 + ai_corr) / (1 - ai_corr))
  z_human <- 0.5 * log((1 + human_corr) / (1 - human_corr))
  se_diff <- sqrt((1/(n1 - 3)) + (1/(n2 - 3)))
  z_stat <- (z_ai - z_human) / se_diff
  p_value <- 2 * (1 - pnorm(abs(z_stat)))
  
  list(
    condition = condition,
    model = model,
    domain = domain,
    ai_corr = ai_corr,
    human_corr = human_corr,
    statistic = z_stat,
    p_value = p_value,
    significant = is.finite(p_value) && (p_value < 0.05)
  )
}

for (i in 1:nrow(correlations)) {
  condition <- as.character(correlations$Condition[i])
  model     <- as.character(correlations$Model[i])
  
  for (domain in domains) {
    if (domain == "Avg") {
      ai_corr    <- correlations$Avg[i]
      human_corr <- human_avg
    } else {
      ai_corr    <- correlations[[domain]][i]
      human_corr <- human_correlations[[domain]]
    }
    results_list <- append(results_list, list(
      perform_test(ai_corr, human_corr, domain, condition, model)
    ))
  }
}

results_df <- do.call(rbind, lapply(results_list, function(x) {
  data.frame(
    Condition = ifelse(is.null(x$condition) || is.na(x$condition), "Unknown", x$condition),
    Model = ifelse(is.null(x$model) || is.na(x$model), "Unknown", x$model),
    Domain = ifelse(is.null(x$domain) || is.na(x$domain), "Unknown", x$domain),
    AI_Correlation = ifelse(is.null(x$ai_corr) || is.na(x$ai_corr), NA, round(x$ai_corr, 4)),
    Human_Correlation = ifelse(is.null(x$human_corr) || is.na(x$human_corr), NA, round(x$human_corr, 4)),
    Difference = ifelse(is.null(x$ai_corr) || is.na(x$ai_corr) || is.null(x$human_corr) || is.na(x$human_corr),
                        NA, round(x$ai_corr - x$human_corr, 4)),
    Test_Statistic = ifelse(is.null(x$statistic) || is.na(x$statistic), NA, round(x$statistic, 4)),
    P_Value = ifelse(is.null(x$p_value) || is.na(x$p_value), NA, round(x$p_value, 6)),
    Significant = ifelse(is.null(x$significant) || is.na(x$significant), FALSE, x$significant),
    Bonferroni_P_Global = NA_real_,
    Bonferroni_Significant_Global = FALSE,
    Bonferroni_P_Domain = NA_real_,
    Bonferroni_Significant_Domain = FALSE,
    stringsAsFactors = FALSE
  )
}))
rownames(results_df) <- 1:nrow(results_df)

# ---------------- Global Bonferroni ----------------
valid_p_indices <- !is.na(results_df$P_Value)
valid_p_values  <- results_df$P_Value[valid_p_indices]

if (length(valid_p_values) > 0) {
  n_tests_global <- length(valid_p_values)
  bonferroni_alpha_global <- 0.05 / n_tests_global
  bonferroni_p_values_global <- p.adjust(valid_p_values, method = "bonferroni")
  
  results_df$Bonferroni_P_Global[valid_p_indices] <- round(bonferroni_p_values_global, 6)
  results_df$Bonferroni_Significant_Global[valid_p_indices] <- bonferroni_p_values_global < 0.05
} else {
  n_tests_global <- 0
  bonferroni_alpha_global <- NA_real_
}

# ---------------- Domain-wise Bonferroni ----------------
for (domain in domains) {
  domain_indices <- results_df$Domain == domain & !is.na(results_df$P_Value)
  domain_p_values <- results_df$P_Value[domain_indices]
  
  if (length(domain_p_values) > 0) {
    n_tests_domain <- length(domain_p_values)
    bonferroni_alpha_domain <- 0.05 / n_tests_domain
    bonferroni_p_values_domain <- p.adjust(domain_p_values, method = "bonferroni")
    
    results_df$Bonferroni_P_Domain[domain_indices] <- round(bonferroni_p_values_domain, 6)
    results_df$Bonferroni_Significant_Domain[domain_indices] <- bonferroni_p_values_domain < 0.05
    
    cat(sprintf("Domain %s: %d tests, corrected alpha = %.6f\n",
                domain, n_tests_domain, bonferroni_alpha_domain))
  }
}

# ============================================================
# Summary / Prints
# ============================================================
cat("\n=== CORRELATION SIGNIFICANCE TEST RESULTS (WITH TWO CORRECTION METHODS) ===\n")
cat("Total number of tests:", nrow(results_df), "\n")
cat("Valid tests (non-NA p-values):", sum(!is.na(results_df$P_Value)), "\n")
cat("Number of significant tests (uncorrected):", sum(results_df$Significant, na.rm = TRUE), "\n")
cat("Number of significant tests (Global Bonferroni):", sum(results_df$Bonferroni_Significant_Global, na.rm = TRUE), "\n")
cat("Number of significant tests (Domain-wise Bonferroni):", sum(results_df$Bonferroni_Significant_Domain, na.rm = TRUE), "\n")
cat("Human average correlation:", round(mean(c(0.7504, 0.8399, 0.8846, 0.7962, 0.7385)), 4), "\n\n")

print(results_df)

# ============================================================
# Export results
# ============================================================
write.csv(results_df, out_csv, row.names = FALSE)
cat(sprintf("\nSaved results to: %s\n", out_csv))
