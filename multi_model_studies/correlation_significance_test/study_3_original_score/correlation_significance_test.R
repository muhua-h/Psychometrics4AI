# Install and load required package (Fisher's z-test doesn't require cocor)
# install.packages("cocor")  # Optional - we're using Fisher's z-test
# library(cocor)

# Define the correlation data for Study 3 (only Simple Binary and Elaborated Binary)
correlations <- data.frame(
  Condition = c(rep("Simple Binary", 5), rep("Elaborated Binary", 5)),
  Model = rep(c("gpt-3.5", "gpt-4", "gpt-4o", "llama", "deepseek"), 2),
  O = c(0.7265, 0.8897, 0.8967, 0.8770, 0.7993,
        0.7590, 0.8858, 0.9107, 0.9120, 0.8456),
  C = c(0.5735, 0.7756, 0.7749, 0.7816, 0.7655,
        0.6685, 0.7940, 0.7803, 0.7654, 0.7698),
  E = c(0.8662, 0.9695, 0.9677, 0.9770, 0.9676,
        0.9512, 0.9730, 0.9698, 0.9753, 0.9678),
  A = c(0.2986, 0.5167, 0.4995, 0.4090, 0.4481,
        0.3107, 0.5452, 0.5007, 0.4666, 0.5110),
  N = c(0.5368, 0.8916, 0.9037, 0.8724, 0.8877,
        0.5938, 0.8916, 0.9085, 0.8736, 0.8860)
)

# Calculate average scores for each row
correlations$Avg <- rowMeans(correlations[, c("O", "C", "E", "A", "N")], na.rm = TRUE)

# Human correlations (reference values)
human_correlations <- c(O = 0.7504, C = 0.8399, E = 0.8846, A = 0.7962, N = 0.7385)
human_avg <- mean(human_correlations)

# Sample sizes for Study 3
n1 <- 200  # AI models
n2 <- 438  # Human

# Personality domains (now including average)
domains <- c("O", "C", "E", "A", "N", "Avg")

# Initialize results storage
results_list <- list()
all_p_values <- c()

# Function to perform correlation comparison test using Fisher's z-transformation
perform_test <- function(ai_corr, human_corr, domain, condition, model) {
  # Check for valid correlations (must be between -1 and 1, not equal to 1 or -1)
  if (is.na(ai_corr) || is.na(human_corr) || 
      abs(ai_corr) >= 1 || abs(human_corr) >= 1) {
    return(list(
      condition = condition,
      model = model,
      domain = domain,
      ai_corr = ai_corr,
      human_corr = human_corr,
      statistic = NA,
      p_value = NA,
      significant = FALSE
    ))
  }
  
  # Fisher's z-transformation for both correlations
  z_ai <- 0.5 * log((1 + ai_corr) / (1 - ai_corr))
  z_human <- 0.5 * log((1 + human_corr) / (1 - human_corr))
  
  # Standard error for the difference (assuming independent samples)
  se_diff <- sqrt((1/(n1-3)) + (1/(n2-3)))
  
  # Test statistic
  z_stat <- (z_ai - z_human) / se_diff
  
  # Two-sided p-value
  p_value <- 2 * (1 - pnorm(abs(z_stat)))
  
  return(list(
    condition = condition,
    model = model,
    domain = domain,
    ai_corr = ai_corr,
    human_corr = human_corr,
    statistic = z_stat,
    p_value = p_value,
    significant = p_value < 0.05
  ))
}

# Perform tests for each AI model vs Human across all domains (including average)
for(i in 1:nrow(correlations)) {
  condition <- correlations$Condition[i]
  model <- correlations$Model[i]
  
  for(domain in domains) {
    if(domain == "Avg") {
      ai_corr <- correlations$Avg[i]
      human_corr <- human_avg
    } else {
      ai_corr <- correlations[[domain]][i]
      human_corr <- human_correlations[domain]
    }
    
    result <- perform_test(ai_corr, human_corr, domain, condition, model)
    results_list <- append(results_list, list(result))
    all_p_values <- c(all_p_values, result$p_value)
  }
}

# Convert results to data frame for easier viewing
results_df <- do.call(rbind, lapply(results_list, function(x) {
  # Handle NA values safely
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
    Bonferroni_P_Global = NA,  # Global correction (all tests)
    Bonferroni_Significant_Global = FALSE,
    Bonferroni_P_Domain = NA,  # Domain-wise correction (tests per domain)
    Bonferroni_Significant_Domain = FALSE,
    stringsAsFactors = FALSE
  )
}))

# Add row names to avoid the missing row names error
rownames(results_df) <- 1:nrow(results_df)

# Apply Global Bonferroni correction (all tests together)
valid_p_indices <- !is.na(results_df$P_Value)
valid_p_values <- results_df$P_Value[valid_p_indices]

if(length(valid_p_values) > 0) {
  n_tests_global <- length(valid_p_values)
  bonferroni_alpha_global <- 0.05 / n_tests_global
  bonferroni_p_values_global <- p.adjust(valid_p_values, method = "bonferroni")
  
  # Fill in Global Bonferroni results for valid tests
  results_df$Bonferroni_P_Global[valid_p_indices] <- round(bonferroni_p_values_global, 6)
  results_df$Bonferroni_Significant_Global[valid_p_indices] <- bonferroni_p_values_global < 0.05
} else {
  n_tests_global <- 0
  bonferroni_alpha_global <- NA
}

# Apply Domain-wise Bonferroni correction (tests per domain)
for(domain in domains) {
  domain_indices <- results_df$Domain == domain & !is.na(results_df$P_Value)
  domain_p_values <- results_df$P_Value[domain_indices]
  
  if(length(domain_p_values) > 0) {
    n_tests_domain <- length(domain_p_values)
    bonferroni_alpha_domain <- 0.05 / n_tests_domain
    bonferroni_p_values_domain <- p.adjust(domain_p_values, method = "bonferroni")
    
    # Fill in Domain-wise Bonferroni results
    results_df$Bonferroni_P_Domain[domain_indices] <- round(bonferroni_p_values_domain, 6)
    results_df$Bonferroni_Significant_Domain[domain_indices] <- bonferroni_p_values_domain < 0.05
    
    cat(sprintf("Domain %s: %d tests, corrected alpha = %.6f\n", 
                domain, n_tests_domain, bonferroni_alpha_domain))
  }
}

# Print summary statistics
cat("\n=== STUDY 3 CORRELATION SIGNIFICANCE TEST RESULTS (WITH TWO CORRECTION METHODS) ===\n")
cat("Total number of tests:", nrow(results_df), "\n")
cat("Valid tests (non-NA p-values):", sum(!is.na(results_df$P_Value)), "\n")
if(!is.na(bonferroni_alpha_global)) {
  cat("Global Bonferroni-corrected alpha level:", round(bonferroni_alpha_global, 6), "\n")
}
cat("Domain-wise Bonferroni-corrected alpha level (per domain):", 0.05/10, "\n")  # 10 tests per domain (2 conditions * 5 models)
cat("Number of significant tests (uncorrected):", sum(results_df$Significant, na.rm = TRUE), "\n")
cat("Number of significant tests (Global Bonferroni):", sum(results_df$Bonferroni_Significant_Global, na.rm = TRUE), "\n")
cat("Number of significant tests (Domain-wise Bonferroni):", sum(results_df$Bonferroni_Significant_Domain, na.rm = TRUE), "\n")
cat("Human average correlation:", round(human_avg, 4), "\n")
cat("Study 3 sample size:", n1, "\n\n")

# Display results
print(results_df)

# Compare the two correction methods
cat("\n=== COMPARISON OF CORRECTION METHODS ===\n")
comparison <- data.frame(
  Domain = results_df$Domain,
  Condition = results_df$Condition,
  Model = results_df$Model,
  P_Value = results_df$P_Value,
  Global_Corrected = results_df$Bonferroni_Significant_Global,
  Domain_Corrected = results_df$Bonferroni_Significant_Domain,
  Difference = results_df$Bonferroni_Significant_Domain != results_df$Bonferroni_Significant_Global
)

# Show cases where the two methods differ
different_results <- comparison[comparison$Difference & !is.na(comparison$P_Value), ]
if(nrow(different_results) > 0) {
  cat("Cases where Global and Domain-wise corrections differ:\n")
  print(different_results)
} else {
  cat("Global and Domain-wise corrections produce identical results.\n")
}

# Summary by domain for both correction methods
cat("\n=== SIGNIFICANT RESULTS BY DOMAIN (BOTH METHODS) ===\n")
for(domain in domains) {
  domain_data <- results_df[results_df$Domain == domain, ]
  global_sig <- sum(domain_data$Bonferroni_Significant_Global, na.rm = TRUE)
  domain_sig <- sum(domain_data$Bonferroni_Significant_Domain, na.rm = TRUE)
  total_tests <- sum(!is.na(domain_data$P_Value))
  
  cat(sprintf("Domain %s: %d/%d significant (Global), %d/%d significant (Domain-wise)\n", 
              domain, global_sig, total_tests, domain_sig, total_tests))
}

# Show significant results for both methods
cat("\n=== SIGNIFICANT RESULTS (GLOBAL BONFERRONI CORRECTION) ===\n")
global_significant <- results_df[results_df$Bonferroni_Significant_Global & !is.na(results_df$Bonferroni_Significant_Global), ]
if(nrow(global_significant) > 0) {
  print(global_significant[, c("Condition", "Model", "Domain", "AI_Correlation", "Human_Correlation", 
                               "Difference", "P_Value", "Bonferroni_P_Global")])
} else {
  cat("No significant results with global Bonferroni correction.\n")
}

cat("\n=== SIGNIFICANT RESULTS (DOMAIN-WISE BONFERRONI CORRECTION) ===\n")
domain_significant <- results_df[results_df$Bonferroni_Significant_Domain & !is.na(results_df$Bonferroni_Significant_Domain), ]
if(nrow(domain_significant) > 0) {
  print(domain_significant[, c("Condition", "Model", "Domain", "AI_Correlation", "Human_Correlation", 
                               "Difference", "P_Value", "Bonferroni_P_Domain")])
} else {
  cat("No significant results with domain-wise Bonferroni correction.\n")
}

# Save results to CSV
script_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
# Create output file path in the same directory as the script
output_file <- file.path(script_dir, "study3_correlation_significance_results.csv")

# Export results to CSV with error handling
tryCatch({
  write.csv(results_df, output_file, row.names = FALSE)
  cat(sprintf("\nStudy 3 results successfully saved to: %s\n", output_file))
}, error = function(e) {
  # Fallback: save to working directory
  fallback_file <- "study3_correlation_significance_results.csv"
  write.csv(results_df, fallback_file, row.names = FALSE)
  cat(sprintf("\nCould not save to script directory. Results saved to current working directory: %s\n", 
              file.path(getwd(), fallback_file)))
  cat(sprintf("Error was: %s\n", e$message))
})

# Print current working directory and script directory for reference
cat(sprintf("Current working directory: %s\n", getwd()))
if(exists("script_dir")) {
  cat(sprintf("Script directory: %s\n", script_dir))
}

# Study 3 specific summary
cat("\n=== STUDY 3 SPECIFIC NOTES ===\n")
cat("This analysis includes only Simple Binary and Elaborated Binary conditions\n")
cat("Sample size: 200 participants (vs 438 in Study 2)\n")
cat("Total comparisons: 60 tests (2 conditions × 5 models × 6 domains)\n")