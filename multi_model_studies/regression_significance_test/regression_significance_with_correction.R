## Regression Significance Analysis
#
# This R script reads in aggregated regression results from the provided
# CSV file, maps coding labels to human‑readable names, computes
# z‑statistics to compare AI model coefficients against the human baseline
# using available standard errors, applies Bonferroni corrections, and
# writes out separate CSV files for the Risk and Moral scenarios.  The
# output files are saved under `multi_model_studies/regression_significance_test/`.

# Required packages
if (!requireNamespace("dplyr", quietly = TRUE)) {
  install.packages("dplyr")
}
if (!requireNamespace("tidyr", quietly = TRUE)) {
  install.packages("tidyr")
}

library(dplyr)
library(tidyr)

## Helper function to perform z‑test using standard errors
compute_z_test <- function(ai_coeff, ai_se, human_coeff, human_se) {
  # If any standard error is NA, return NA
  if (is.na(ai_se) || is.na(human_se)) {
    return(c(statistic = NA, p_value = NA))
  }
  denom <- sqrt(ai_se^2 + human_se^2)
  if (denom == 0) {
    return(c(statistic = NA, p_value = NA))
  }
  z_stat <- (ai_coeff - human_coeff) / denom
  p_val <- 2 * (1 - pnorm(abs(z_stat)))
  c(statistic = z_stat, p_value = p_val)
}

## Function to process a single scenario (risk or moral)
process_scenario <- function(scenario_rows, human_coeffs, human_ses) {
  domains <- c("O", "C", "E", "A", "N")
  # Compute AI averages and SE of averages
  scenario_rows <- scenario_rows %>%
    mutate(
      Avg = rowMeans(across(all_of(domains))),
      Avg_se = sqrt(rowSums(across(paste0(domains, "_se"), ~ (.x)^2))) / length(domains)
    )
  human_avg <- mean(unlist(human_coeffs))
  human_se_avg <- sqrt(sum(unlist(human_ses)^2)) / length(domains)
  # Initialize result storage
  results <- list()
  # Loop through each row and domain
  for (i in seq_len(nrow(scenario_rows))) {
    row <- scenario_rows[i, ]
    fmt <- as.character(row$Format)
    mdl <- as.character(row$Model)
    # Domain level comparisons
    for (d in domains) {
      ai_coeff <- row[[d]]
      ai_se <- row[[paste0(d, "_se")]]
      human_coeff <- human_coeffs[[d]]
      human_se <- human_ses[[d]]
      test <- compute_z_test(ai_coeff, ai_se, human_coeff, human_se)
      results[[length(results) + 1]] <- data.frame(
        Format = fmt,
        Model = mdl,
        Domain = d,
        AI_Coefficient = round(ai_coeff, 4),
        Human_Coefficient = round(human_coeff, 4),
        AI_SE = round(ai_se, 6),
        Human_SE = round(human_se, 6),
        Difference = round(ai_coeff - human_coeff, 4),
        Test_Statistic = ifelse(is.na(test["statistic"]), NA, round(test["statistic"], 4)),
        P_Value = ifelse(is.na(test["p_value"]), NA, round(test["p_value"], 6)),
        Significant = ifelse(is.na(test["p_value"]), FALSE, test["p_value"] < 0.05),
        stringsAsFactors = FALSE
      )
    }
    # Average comparison
    ai_coeff <- row$Avg
    ai_se <- row$Avg_se
    human_coeff <- human_avg
    human_se <- human_se_avg
    test <- compute_z_test(ai_coeff, ai_se, human_coeff, human_se)
    results[[length(results) + 1]] <- data.frame(
      Format = fmt,
      Model = mdl,
      Domain = "Avg",
      AI_Coefficient = round(ai_coeff, 4),
      Human_Coefficient = round(human_coeff, 4),
      AI_SE = round(ai_se, 6),
      Human_SE = round(human_se, 6),
      Difference = round(ai_coeff - human_coeff, 4),
      Test_Statistic = ifelse(is.na(test["statistic"]), NA, round(test["statistic"], 4)),
      P_Value = ifelse(is.na(test["p_value"]), NA, round(test["p_value"], 6)),
      Significant = ifelse(is.na(test["p_value"]), FALSE, test["p_value"] < 0.05),
      stringsAsFactors = FALSE
    )
  }
  results_df <- do.call(rbind, results)
  # Global Bonferroni correction
  valid_idx <- !is.na(results_df$P_Value)
  pvals <- results_df$P_Value[valid_idx]
  if (length(pvals) > 0) {
    corrected_global <- pmin(pvals * length(pvals), 1)
    results_df$Bonferroni_P_Global <- NA
    results_df$Bonferroni_P_Global[valid_idx] <- round(corrected_global, 6)
    results_df$Bonferroni_Significant_Global <- FALSE
    results_df$Bonferroni_Significant_Global[valid_idx] <- corrected_global < 0.05
  } else {
    results_df$Bonferroni_P_Global <- NA
    results_df$Bonferroni_Significant_Global <- FALSE
  }
  # Domain-wise Bonferroni correction
  domains_with_avg <- c(domains, "Avg")
  results_df$Bonferroni_P_Domain <- NA
  results_df$Bonferroni_Significant_Domain <- FALSE
  for (dom in domains_with_avg) {
    idx <- which(results_df$Domain == dom & valid_idx)
    pvals_d <- results_df$P_Value[idx]
    if (length(pvals_d) > 0) {
      corrected_d <- pmin(pvals_d * length(pvals_d), 1)
      results_df$Bonferroni_P_Domain[idx] <- round(corrected_d, 6)
      results_df$Bonferroni_Significant_Domain[idx] <- corrected_d < 0.05
    }
  }
  results_df
}

## Main analysis
{
  # Define path to the aggregated results CSV relative to this script
  input_file <- file.path("multi_model_studies", "study_4", "study_4_generalized_analysis_results", "aggregated_measures_regression_results.csv")
  data <- read.csv(input_file, stringsAsFactors = FALSE)
  # Mappings from raw codes to descriptive names
  format_map <- c(
    human_baseline = "human_baseline",
    bfi_binary_simple = "Simple Binary",
    bfi_binary_elaborated = "Elaborated Binary",
    bfi_likert = "BFI-2 Likert",
    bfi_expanded = "BFI-2 Expanded"
  )
  model_map <- c(
    human = "human",
    `openai-gpt-3.5-turbo-0125` = "gpt-3.5-turbo",
    `gpt-4` = "gpt-4",
    `gpt-4o` = "gpt-4o",
    llama = "Llama",
    deepseek = "Deepseek"
  )
  predictor_map <- c(
    openness = "O",
    conscientiousness = "C",
    extraversion = "E",
    agreeableness = "A",
    neuroticism = "N"
  )
  # Apply mappings
  data$Format <- format_map[data$format]
  data$Model <- model_map[data$model]
  data$Domain <- predictor_map[data$predictor]
  data$Scenario <- data$scenario_type
  data$Coefficient <- data$standardized_coefficient
  data$SE <- data$std_error
  # Extract human baseline coefficients and SEs by scenario
  baseline <- data %>% filter(Model == "human")
  baseline_coeffs <- list()
  baseline_ses <- list()
  for (sc in unique(data$Scenario)) {
    base_sub <- baseline %>% filter(Scenario == sc)
    coeffs_vec <- setNames(base_sub$Coefficient, base_sub$Domain)
    ses_vec <- setNames(base_sub$SE, base_sub$Domain)
    baseline_coeffs[[sc]] <- coeffs_vec
    baseline_ses[[sc]] <- ses_vec
  }
  # Prepare AI data (exclude human)
  ai_data <- data %>% filter(Model != "human")
  # Create output directory
  out_dir <- file.path("multi_model_studies", "regression_significance_test")
  if (!dir.exists(out_dir)) {
    dir.create(out_dir, recursive = TRUE)
  }
  # Process each scenario
  scenarios <- unique(data$Scenario)
  for (sc in scenarios) {
    ai_sub <- ai_data %>% filter(Scenario == sc)
    # Pivot to wide for coefficients and SEs
    pivot_coeff <- ai_sub %>% select(Format, Model, Domain, Coefficient) %>%
      pivot_wider(names_from = Domain, values_from = Coefficient)
    pivot_se <- ai_sub %>% select(Format, Model, Domain, SE) %>%
      pivot_wider(names_from = Domain, values_from = SE, names_prefix = "", names_sep = "")
    # Add suffix _se to SE columns
    se_cols <- colnames(pivot_se)[!(colnames(pivot_se) %in% c("Format", "Model"))]
    for (cname in se_cols) {
      pivot_coeff[[paste0(cname, "_se")]] <- pivot_se[[cname]]
    }
    section_df <- pivot_coeff
    # Process scenario
    result_df <- process_scenario(section_df, baseline_coeffs[[sc]], baseline_ses[[sc]])
    # Save
    out_file <- file.path(out_dir, paste0(sc, "_regression_significance_results.csv"))
    write.csv(result_df, out_file, row.names = FALSE)
    # Construct capitalized scenario name for message
    cap_sc <- paste0(toupper(substr(sc, 1, 1)), substr(sc, 2, nchar(sc)))
    cat(sprintf("%s results saved to: %s\n", cap_sc, out_file))
  }
}