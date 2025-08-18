# Install and load required package (Fisher's z-test doesn't require cocor)
# install.packages("cocor")  # Optional - we're using Fisher's z-test
# library(cocor)

# Define the correlation data
correlations <- data.frame(
  Condition = c(rep("Simple Binary", 5), rep("Elaborated Binary", 5), 
                rep("Expanded Format", 5), rep("Likert", 5)),
  Model = rep(c("gpt-3.5", "gpt-4", "gpt-4o", "llama", "deepseek"), 4),
  O = c(0.3580, 0.3905, 0.4034, 0.4087, 0.3188,
        0.3612, 0.3905, 0.3992, 0.3939, 0.3314,
        0.7843, 0.7936, 0.8205, 0.8486, 0.8091,
        0.5667, 0.6773, 0.8576, 0.8866, 0.8165),
  C = c(0.5537, 0.5810, 0.6218, 0.5791, 0.5757,
        0.5757, 0.5848, 0.6098, 0.5744, 0.5653,
        0.7936, 0.8738, 0.9314, 0.9154, 0.9207,
        0.8484, 0.9096, 0.9257, 0.9149, 0.9200),
  E = c(0.6738, 0.6737, 0.6936, 0.6716, 0.6739,
        0.6749, 0.6805, 0.6853, 0.6735, 0.6786,
        0.8608, 0.9046, 0.9169, 0.9206, 0.9071,
        0.8607, 0.9249, 0.9218, 0.9376, 0.9315),
  A = c(0.2651, 0.3895, 0.4640, 0.3707, 0.4329,
        0.2773, 0.4281, 0.4356, 0.3905, 0.4251,
        0.6050, 0.8306, 0.9112, 0.8954, 0.8881,
        0.7156, 0.9065, 0.9117, 0.8780, 0.9073),
  N = c(0.5691, 0.7740, 0.7720, 0.7701, 0.7801,
        0.5674, 0.7746, 0.7696, 0.7703, 0.7799,
        0.8211, 0.9068, 0.9251, 0.9189, 0.9321,
        0.8277, 0.9091, 0.9264, 0.9157, 0.9230)
)

# Calculate average scores for each row
correlations$Avg <- rowMeans(correlations[, c("O", "C", "E", "A", "N")], na.rm = TRUE)

# Human correlations (reference values)
human_correlations <- c(O = 0.7504, C = 0.8399, E = 0.8846, A = 0.7962, N = 0.7385)
human_avg <- mean(human_correlations)

# Sample sizes
n1 <- 438  # AI models
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
    Bonferroni_P_Global = NA,  # Global correction (all 120 tests)
    Bonferroni_Significant_Global = FALSE,
    Bonferroni_P_Domain = NA,  # Domain-wise correction (20 tests per domain)
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

# Apply Domain-wise Bonferroni correction (20 tests per domain)
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
cat("\n=== CORRELATION SIGNIFICANCE TEST RESULTS (WITH TWO CORRECTION METHODS) ===\n")
cat("Total number of tests:", nrow(results_df), "\n")
cat("Valid tests (non-NA p-values):", sum(!is.na(results_df$P_Value)), "\n")
if(!is.na(bonferroni_alpha_global)) {
  cat("Global Bonferroni-corrected alpha level:", round(bonferroni_alpha_global, 6), "\n")
}
cat("Domain-wise Bonferroni-corrected alpha level (per domain):", 0.05/20, "\n")
cat("Number of significant tests (uncorrected):", sum(results_df$Significant, na.rm = TRUE), "\n")
cat("Number of significant tests (Global Bonferroni):", sum(results_df$Bonferroni_Significant_Global, na.rm = TRUE), "\n")
cat("Number of significant tests (Domain-wise Bonferroni):", sum(results_df$Bonferroni_Significant_Domain, na.rm = TRUE), "\n")
cat("Human average correlation:", round(human_avg, 4), "\n\n")

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

# Export results to CSV (optional)
write.csv(results_df, "multi_model_studies/correlation_significance_test/correlation_significance_results_updated.csv", row.names = FALSE)