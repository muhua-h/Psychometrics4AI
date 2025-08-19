# Enhanced CFA Analysis with Explicit Domain-Level Warnings
# Modifications to make warnings and messages domain-specific

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(jsonlite)
  library(lavaan)
  library(psych)
  library(semTools)
})

# Configuration
BASE_DIR <- "/Users/mhhuang/Psychometrics4AI_revision/multi_model_studies"
RESULTS_DIR <- file.path(BASE_DIR, "factor_analysis", "results_r")

# Ensure results directory exists
if (!dir.exists(RESULTS_DIR)) {
  dir.create(RESULTS_DIR, recursive = TRUE)
}


# Define Big Five domains with actual mini-marker items
DOMAINS <- list(
  Extraversion = c("Bold", "Energetic", "Extraverted", "Talkative", "Bashful", "Quiet", "Shy", "Withdrawn"),
  Agreeableness = c("Cooperative", "Kind", "Sympathetic", "Warm", "Cold", "Harsh", "Rude", "Unsympathetic"),
  Conscientiousness = c("Efficient", "Organized", "Practical", "Systematic", "Careless", "Disorganized", "Inefficient", "Sloppy"),
  Neuroticism = c("Envious", "Fretful", "Jealous", "Moody", "Temperamental", "Touchy", "Relaxed", "Unenvious"),
  Openness = c("Complex", "Deep", "Creative", "Imaginative", "Intellectual", "Philosophical", "Uncreative", "Unintellectual")
)

# Items that need reverse scoring
REVERSE_ITEMS <- c(
  "Bashful", "Quiet", "Shy", "Withdrawn", 
  "Cold", "Harsh", "Rude", "Unsympathetic",
  "Careless", "Disorganized", "Inefficient", "Sloppy", 
  "Relaxed", "Unenvious", "Uncreative", "Unintellectual"
)

# Function to detect scale range
get_scale_range <- function(data) {
  max_val <- max(data, na.rm = TRUE)
  if (max_val <= 5) return(5)
  if (max_val <= 7) return(7)
  if (max_val <= 9) return(9)
  return(9)
}

# Function to reverse score items
correct_reverse_score <- function(x, scale_max) {
  (scale_max + 1) - x
}

# Function to load and prepare data with proper reverse coding
load_and_prepare_data <- function(json_path, model_name, format_type, study_name) {
  cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
  cat("🎯 Analyzing:", model_name, "in", format_type, "format (", study_name, ")\n")
  cat(paste(rep("=", 80), collapse = ""), "\n", sep = "")
  
  if (!file.exists(json_path)) {
    cat("❌ File not found:", json_path, "\n")
    return(NULL)
  }
  
  cat("📊 Loading:", basename(json_path), "\n")
  
  tryCatch({
    # Load JSON
    json_data <- fromJSON(json_path)
    
    # Convert to data frame
    if (is.list(json_data) && length(json_data) > 0) {
      df <- as.data.frame(json_data)
    } else {
      df <- as.data.frame(json_data)
    }
    
    # Convert to numeric
    data <- data.frame(lapply(df, as.numeric))
    
    # Remove missing data
    data <- data[complete.cases(data), ]
    
    if (nrow(data) < 10) {
      cat("❌ Insufficient data - only", nrow(data), "rows\n")
      return(NULL)
    }
    
    # Detect scale range
    scale_max <- get_scale_range(data)
    cat("🔍 Detected scale range: 1-", scale_max, "\n", sep = "")
    
    # Apply reverse coding
    existing_reverse <- intersect(REVERSE_ITEMS, names(data))
    if (length(existing_reverse) > 0) {
      cat("↩️ Reverse coding", length(existing_reverse), "items:\n")
      for (item in existing_reverse) {
        original_mean <- mean(data[[item]], na.rm = TRUE)
        data[[item]] <- correct_reverse_score(data[[item]], scale_max)
        new_mean <- mean(data[[item]], na.rm = TRUE)
        cat(sprintf("    %-15s: %.2f → %.2f\n", item, original_mean, new_mean))
      }
    }
    
    # Check for near-zero variance
    variances <- apply(data, 2, var, na.rm = TRUE)
    near_zero <- which(variances < 0.01)
    if (length(near_zero) > 0) {
      cat("⚠️ Removing near-zero variance items:", names(data)[near_zero], "\n")
      data <- data[, -near_zero]
    }
    
    cat("✅ Prepared:", nrow(data), "observations,", ncol(data), "items\n")
    return(data)
    
  }, error = function(e) {
    cat("❌ Error loading", basename(json_path), ":", e$message, "\n")
    return(NULL)
  })
}

# ENHANCED Function to run CFA for a single domain with better warning handling
run_domain_cfa <- function(domain, items, data, model_name, format_type, study_name) {
  domain_header <- paste0("🔸 [", toupper(domain), " DOMAIN] ")
  
  cat("\n", paste(rep("-", 60), collapse = ""), "\n", sep = "")
  cat(domain_header, "Starting CFA Analysis\n")
  cat(paste(rep("-", 60), collapse = ""), "\n", sep = "")
  
  # Capture all warnings for this domain
  domain_warnings <- character(0)
  
  withCallingHandlers({
    tryCatch({
      # Clean item names to avoid special characters issues
      clean_items <- gsub("[^a-zA-Z0-9]", "_", items)
      names(data) <- gsub("[^a-zA-Z0-9]", "_", names(data))
      items <- clean_items[items %in% gsub("[^a-zA-Z0-9]", "_", items)]
      
      if (length(items) < 3) {
        cat(domain_header, "❌ Insufficient items (", length(items), ") - Need at least 3\n")
        return(NULL)
      }
      
      # Check item availability in data
      available_items <- intersect(items, names(data))
      missing_items <- setdiff(items, names(data))
      
      if (length(missing_items) > 0) {
        cat(domain_header, "⚠️ Missing items:", paste(missing_items, collapse = ", "), "\n")
      }
      
      if (length(available_items) < 3) {
        cat(domain_header, "❌ Only", length(available_items), "items available - Need at least 3\n")
        return(NULL)
      }
      
      items <- available_items
      model_syntax <- paste(domain, "=~", paste(items, collapse = " + "))
      
      cat(domain_header, "📋 Model specification:\n")
      # cat(domain_header, "   Items (", length(items), "):", paste(items, collapse = ", "), "\n")
      cat(domain_header, "   Syntax:", model_syntax, "\n")
      
      # Check correlations before fitting
      item_data <- data[, items, drop = FALSE]
      cor_matrix <- cor(item_data, use = "complete.obs")
      
      # Check for problematic correlations
      if (any(is.na(cor_matrix))) {
        cat(domain_header, "⚠️ NA values in correlation matrix\n")
      }
      
      # Check for very high correlations (potential multicollinearity)
      upper_tri <- cor_matrix[upper.tri(cor_matrix)]
      high_cors <- sum(abs(upper_tri) > 0.95, na.rm = TRUE)
      if (high_cors > 0) {
        cat(domain_header, "⚠️ Found", high_cors, "correlations > 0.95 (potential multicollinearity)\n")
      }
      
      # Check for very low correlations
      low_cors <- sum(abs(upper_tri) < 0.1, na.rm = TRUE)
      if (low_cors > length(upper_tri) * 0.5) {
        cat(domain_header, "⚠️ Many low correlations - items may not be measuring same construct\n")
      }
      
      cat(domain_header, "🔄 Fitting CFA model...\n")
      
      # Fit the model
      fit <- cfa(model_syntax, data = data, estimator = "MLR", std.lv = TRUE)
      
      # Check convergence
      if (!lavInspect(fit, "converged")) {
        cat(domain_header, "❌ Model did not converge\n")
        return(NULL)
      }
      
      # Check for estimation problems
      # if (lavInspect(fit, "post.check")) {
      #   cat(domain_header, "⚠️ Post-fitting check issues detected\n")
      # }
      # 
      # Get fit measures
      fit_measures <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
      
      # Calculate reliability measures
      cat(domain_header, "📊 Calculating reliability measures...\n")
      
      # Cronbach's Alpha
      alpha_result <- psych::alpha(data[, items])
      alpha <- alpha_result$total$raw_alpha
      
      # Check for negative alpha (indicates problems)
      if (alpha < 0) {
        cat(domain_header, "⚠️ Negative Cronbach's Alpha (", round(alpha, 3), ") - Check item scoring\n")
      }
      
      # McDonald's Omega
      omega_result <- psych::omega(data[, items], plot = FALSE)
      omega <- omega_result$omega.tot
      
      # Display detailed results
      # cat(domain_header, "✅ CFA Results Summary:\n")
      # cat(domain_header, sprintf("   📈 Sample Size: %d participants\n", nrow(data)))
      # cat(domain_header, sprintf("   📊 Items Analyzed: %d\n", length(items)))
      # cat(domain_header, sprintf("   🔸 Cronbach's Alpha: %.3f\n", alpha))
      # cat(domain_header, sprintf("   🔸 McDonald's Omega: %.3f\n", omega))
      # cat(domain_header, sprintf("   🔸 CFI: %.3f\n", fit_measures["cfi"]))
      # cat(domain_header, sprintf("   🔸 TLI: %.3f\n", fit_measures["tli"]))
      # cat(domain_header, sprintf("   🔸 RMSEA: %.3f\n", fit_measures["rmsea"]))
      # cat(domain_header, sprintf("   🔸 SRMR: %.3f\n", fit_measures["srmr"]))
      # cat(domain_header, sprintf("   🔸 Chi²: %.3f (df=%d, p=%.3f)\n", 
      #                            fit_measures["chisq"], fit_measures["df"], fit_measures["pvalue"]))
      
      # Interpret fit indices
      # cat(domain_header, "🔍 Fit Assessment:\n")
      # if (fit_measures["cfi"] >= 0.95) {
      #   cat(domain_header, "   ✅ CFI: Excellent fit (≥0.95)\n")
      # } else if (fit_measures["cfi"] >= 0.90) {
      #   cat(domain_header, "   ⚠️ CFI: Acceptable fit (0.90-0.94)\n")
      # } else {
      #   cat(domain_header, "   ❌ CFI: Poor fit (<0.90)\n")
      # }
      # 
      # if (fit_measures["rmsea"] <= 0.06) {
      #   cat(domain_header, "   ✅ RMSEA: Good fit (≤0.06)\n")
      # } else if (fit_measures["rmsea"] <= 0.08) {
      #   cat(domain_header, "   ⚠️ RMSEA: Acceptable fit (0.06-0.08)\n")
      # } else {
      #   cat(domain_header, "   ❌ RMSEA: Poor fit (>0.08)\n")
      # }
      
      # Report any collected warnings
      if (length(domain_warnings) > 0) {
        cat(domain_header, "⚠️ Warnings encountered:\n")
        for (w in domain_warnings) {
          cat(domain_header, "   ", w, "\n")
        }
      }
      
      cat(domain_header, "✅ Analysis completed successfully\n")
      
      # Return results
      return(data.frame(
        Study = toupper(study_name),
        Format = format_type,
        Model = model_name,
        Factor_Domain = toupper(domain),
        N_Items = length(items),
        N_Participants = nrow(data),
        Alpha = round(alpha, 3),
        Omega = round(omega, 3),
        CFI = round(fit_measures["cfi"], 3),
        TLI = round(fit_measures["tli"], 3),
        RMSEA = round(fit_measures["rmsea"], 3),
        SRMR = round(fit_measures["srmr"], 3),
        Chi_Square = round(fit_measures["chisq"], 3),
        DF = fit_measures["df"],
        P_Value = round(fit_measures["pvalue"], 3),
        stringsAsFactors = FALSE
      ))
      
    }, error = function(e) {
      cat(domain_header, "❌ CFA FAILED - Error:", e$message, "\n")
      return(NULL)
    })
    
  }, warning = function(w) {
    # Capture warnings with domain context
    warning_msg <- paste0("Warning in ", toupper(domain), ": ", w$message)
    domain_warnings <<- c(domain_warnings, warning_msg)
    cat(domain_header, "⚠️ WARNING:", w$message, "\n")
    invokeRestart("muffleWarning")
  })
}

# ENHANCED Function to run complete analysis with domain-by-domain reporting
run_single_analysis <- function(json_path, model_name, format_type, study_name) {
  data <- load_and_prepare_data(json_path, model_name, format_type, study_name)
  if (is.null(data)) return(NULL)
  
  results <- list()
  successful_domains <- character(0)
  failed_domains <- character(0)
  
  cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
  cat("🔄 STARTING DOMAIN-BY-DOMAIN CFA ANALYSIS\n")
  cat(paste(rep("=", 80), collapse = ""), "\n", sep = "")
  
  # Run CFA for each domain with detailed reporting
  for (i in seq_along(DOMAINS)) {
    domain <- names(DOMAINS)[i]
    domain_items <- DOMAINS[[domain]]
    
    cat(sprintf("\n📍 DOMAIN %d of %d: %s\n", i, length(DOMAINS), toupper(domain)))
    
    # Check item availability
    available_items <- intersect(domain_items, names(data))
    missing_items <- setdiff(domain_items, names(data))
    
    cat(sprintf("   Expected items: %d\n", length(domain_items)))
    cat(sprintf("   Available items: %d\n", length(available_items)))
    
    if (length(missing_items) > 0) {
      cat("   Missing items:", paste(missing_items, collapse = ", "), "\n")
    }
    
    if (length(available_items) >= 3) {
      result <- run_domain_cfa(domain, available_items, data, model_name, format_type, study_name)
      if (!is.null(result)) {
        results[[length(results) + 1]] <- result
        successful_domains <- c(successful_domains, domain)
        cat("✅ ", toupper(domain), " - COMPLETED SUCCESSFULLY\n")
      } else {
        failed_domains <- c(failed_domains, domain)
        cat("❌ ", toupper(domain), " - FAILED\n")
      }
    } else {
      failed_domains <- c(failed_domains, domain)
      cat("❌ ", toupper(domain), " - INSUFFICIENT ITEMS (need ≥3, have ", length(available_items), ")\n")
    }
  }
  
  # Final summary
  cat("\n", paste(rep("=", 80), collapse = ""), "\n", sep = "")
  cat("📋 FINAL ANALYSIS SUMMARY\n")
  cat(paste(rep("=", 80), collapse = ""), "\n", sep = "")
  cat("✅ Successful domains (", length(successful_domains), "):", paste(successful_domains, collapse = ", "), "\n")
  if (length(failed_domains) > 0) {
    cat("❌ Failed domains (", length(failed_domains), "):", paste(failed_domains, collapse = ", "), "\n")
  }
  
  if (length(results) == 0) {
    cat("❌ No valid CFA results for", model_name, "\n")
    return(NULL)
  }
  
  final_results <- do.call(rbind, results)
  
  # Save results
  output_dir <- file.path(RESULTS_DIR, study_name, paste0(format_type, "_format"))
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  output_file <- file.path(output_dir, paste0(model_name, "_R_factor_analysis.csv"))
  write_csv(final_results, output_file)
  
  cat("💾 Results saved to:", output_file, "\n")
  return(final_results)
}

# Keep all your existing function definitions for the specific analyses...
# [All the run_study2_* and run_study3_* functions remain the same]

# ==============================================
# STUDY 2 - ALL MODEL × FORMAT COMBINATIONS
# ==============================================
# ===============================
# STUDY 2 - EXPANDED
# ===============================
run_study2_expanded_gpt_3_5_turbo <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_expanded_results","bfi_to_minimarker_gpt_3_5_turbo_temp1_0.json"),
                      "gpt_3_5_turbo","expanded","study_2")
}
run_study2_expanded_gpt_4 <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_expanded_results","bfi_to_minimarker_gpt_4_temp1_0.json"),
                      "gpt_4","expanded","study_2")
}
run_study2_expanded_gpt_4o <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_expanded_results","bfi_to_minimarker_gpt_4o_temp1_0.json"),
                      "gpt_4o","expanded","study_2")
}
run_study2_expanded_llama <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_expanded_results","bfi_to_minimarker_llama_temp1_0.json"),
                      "llama","expanded","study_2")
}
run_study2_expanded_deepseek <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_expanded_results","bfi_to_minimarker_deepseek_temp1_0.json"),
                      "deepseek","expanded","study_2")
}

# ===============================
# STUDY 2 - BINARY SIMPLE
# ===============================
run_study2_binary_simple_gpt_3_5_turbo <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_binary_simple_results","bfi_to_minimarker_gpt_3_5_turbo_temp1_0.json"),
                      "gpt_3_5_turbo","binary_simple","study_2")
}
run_study2_binary_simple_gpt_4 <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_binary_simple_results","bfi_to_minimarker_gpt_4_temp1_0.json"),
                      "gpt_4","binary_simple","study_2")
}
run_study2_binary_simple_gpt_4o <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_binary_simple_results","bfi_to_minimarker_gpt_4o_temp1_0.json"),
                      "gpt_4o","binary_simple","study_2")
}
run_study2_binary_simple_llama <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_binary_simple_results","bfi_to_minimarker_llama_temp1_0.json"),
                      "llama","binary_simple","study_2")
}
run_study2_binary_simple_deepseek <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_binary_simple_results","bfi_to_minimarker_deepseek_temp1_0.json"),
                      "deepseek","binary_simple","study_2")
}

# ===============================
# STUDY 2 - BINARY ELABORATED
# ===============================
run_study2_binary_elaborated_gpt_3_5_turbo <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_binary_elaborated_results","bfi_to_minimarker_gpt_3_5_turbo_temp1_0.json"),
                      "gpt_3_5_turbo","binary_elaborated","study_2")
}
run_study2_binary_elaborated_gpt_4 <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_binary_elaborated_results","bfi_to_minimarker_gpt_4_temp1_0.json"),
                      "gpt_4","binary_elaborated","study_2")
}
run_study2_binary_elaborated_gpt_4o <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_binary_elaborated_results","bfi_to_minimarker_gpt_4o_temp1_0.json"),
                      "gpt_4o","binary_elaborated","study_2")
}
run_study2_binary_elaborated_llama <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_binary_elaborated_results","bfi_to_minimarker_llama_temp1_0.json"),
                      "llama","binary_elaborated","study_2")
}
run_study2_binary_elaborated_deepseek <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_binary_elaborated_results","bfi_to_minimarker_deepseek_temp1_0.json"),
                      "deepseek","binary_elaborated","study_2")
}

# ===============================
# STUDY 2 - LIKERT
# ===============================
run_study2_likert_gpt_3_5_turbo <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_likert_results","bfi_to_minimarker_gpt_3_5_turbo_temp1_0.json"),
                      "gpt_3_5_turbo","likert","study_2")
}
run_study2_likert_gpt_4 <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_likert_results","bfi_to_minimarker_gpt_4_temp1_0.json"),
                      "gpt_4","likert","study_2")
}
run_study2_likert_gpt_4o <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_likert_results","bfi_to_minimarker_gpt_4o_temp1_0.json"),
                      "gpt_4o","likert","study_2")
}
run_study2_likert_llama <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_likert_results","bfi_to_minimarker_llama_temp1_0.json"),
                      "llama","likert","study_2")
}
run_study2_likert_deepseek <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_2","study_2_likert_results","bfi_to_minimarker_deepseek_temp1_0.json"),
                      "deepseek","likert","study_2")
}

##=====
# ===============================
# STUDY 3 - EXPANDED
# ===============================
run_study3_expanded_gpt_3_5_turbo <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_expanded_results","bfi_to_minimarker_gpt_3_5_turbo_temp1_0.json"),
                      "gpt_3_5_turbo","expanded","study_3")
}
run_study3_expanded_gpt_4 <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_expanded_results","bfi_to_minimarker_gpt_4_temp1_0.json"),
                      "gpt_4","expanded","study_3")
}
run_study3_expanded_gpt_4o <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_expanded_results","bfi_to_minimarker_gpt_4o_temp1_0.json"),
                      "gpt_4o","expanded","study_3")
}
run_study3_expanded_llama <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_expanded_results","bfi_to_minimarker_llama_temp1_0.json"),
                      "llama","expanded","study_3")
}
run_study3_expanded_deepseek <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_expanded_results","bfi_to_minimarker_deepseek_temp1_0.json"),
                      "deepseek","expanded","study_3")
}

# ===============================
# STUDY 3 - BINARY SIMPLE
# ===============================
run_study3_binary_simple_gpt_3_5_turbo <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_binary_simple_results","bfi_to_minimarker_gpt_3_5_turbo_temp1_0.json"),
                      "gpt_3_5_turbo","binary_simple","study_3")
}
run_study3_binary_simple_gpt_4 <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_binary_simple_results","bfi_to_minimarker_gpt_4_temp1_0.json"),
                      "gpt_4","binary_simple","study_3")
}
run_study3_binary_simple_gpt_4o <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_binary_simple_results","bfi_to_minimarker_gpt_4o_temp1_0.json"),
                      "gpt_4o","binary_simple","study_3")
}
run_study3_binary_simple_llama <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_binary_simple_results","bfi_to_minimarker_llama_temp1_0.json"),
                      "llama","binary_simple","study_3")
}
run_study3_binary_simple_deepseek <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_binary_simple_results","bfi_to_minimarker_deepseek_temp1_0.json"),
                      "deepseek","binary_simple","study_3")
}

# ===============================
# STUDY 3 - BINARY ELABORATED
# ===============================
run_study3_binary_elaborated_gpt_3_5_turbo <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_binary_elaborated_results","bfi_to_minimarker_gpt_3_5_turbo_temp1_0.json"),
                      "gpt_3_5_turbo","binary_elaborated","study_3")
}
run_study3_binary_elaborated_gpt_4 <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_binary_elaborated_results","bfi_to_minimarker_gpt_4_temp1_0.json"),
                      "gpt_4","binary_elaborated","study_3")
}
run_study3_binary_elaborated_gpt_4o <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_binary_elaborated_results","bfi_to_minimarker_gpt_4o_temp1_0.json"),
                      "gpt_4o","binary_elaborated","study_3")
}
run_study3_binary_elaborated_llama <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_binary_elaborated_results","bfi_to_minimarker_llama_temp1_0.json"),
                      "llama","binary_elaborated","study_3")
}
run_study3_binary_elaborated_deepseek <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_binary_elaborated_results","bfi_to_minimarker_deepseek_temp1_0.json"),
                      "deepseek","binary_elaborated","study_3")
}

# ===============================
# STUDY 3 - LIKERT
# ===============================
run_study3_likert_gpt_3_5_turbo <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_likert_results","bfi_to_minimarker_gpt_3_5_turbo_temp1_0.json"),
                      "gpt_3_5_turbo","likert","study_3")
}
run_study3_likert_gpt_4 <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_likert_results","bfi_to_minimarker_gpt_4_temp1_0.json"),
                      "gpt_4","likert","study_3")
}
run_study3_likert_gpt_4o <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_likert_results","bfi_to_minimarker_gpt_4o_temp1_0.json"),
                      "gpt_4o","likert","study_3")
}
run_study3_likert_llama <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_likert_results","bfi_to_minimarker_llama_temp1_0.json"),
                      "llama","likert","study_3")
}
run_study3_likert_deepseek <- function() {
  run_single_analysis(file.path(BASE_DIR,"study_3","study_3_likert_results","bfi_to_minimarker_deepseek_temp1_0.json"),
                      "deepseek","likert","study_3")
}

#########################################################
#########################################################
#########################################################
#########################################################
#########################################################
# ==== STUDY 2 - EXPANDED
run_study2_expanded_gpt_3_5_turbo()
run_study2_expanded_gpt_4()
run_study2_expanded_gpt_4o()
run_study2_expanded_llama()
run_study2_expanded_deepseek()

# ==== STUDY 2 - BINARY SIMPLE
run_study2_binary_simple_gpt_3_5_turbo()
run_study2_binary_simple_gpt_4()
run_study2_binary_simple_gpt_4o()
run_study2_binary_simple_llama()
run_study2_binary_simple_deepseek()

# ==== STUDY 2 - BINARY ELABORATED
run_study2_binary_elaborated_gpt_3_5_turbo()
run_study2_binary_elaborated_gpt_4()
run_study2_binary_elaborated_gpt_4o()
run_study2_binary_elaborated_llama()
run_study2_binary_elaborated_deepseek()

# ==== STUDY 2 - LIKERT
run_study2_likert_gpt_3_5_turbo()
run_study2_likert_gpt_4()
run_study2_likert_gpt_4o()
run_study2_likert_llama()
run_study2_likert_deepseek()

# ==== STUDY 3 - EXPANDED
run_study3_expanded_gpt_3_5_turbo()
run_study3_expanded_gpt_4()
run_study3_expanded_gpt_4o()
run_study3_expanded_llama()
run_study3_expanded_deepseek()

# ==== STUDY 3 - BINARY SIMPLE
run_study3_binary_simple_gpt_3_5_turbo()
run_study3_binary_simple_gpt_4()
run_study3_binary_simple_gpt_4o()
run_study3_binary_simple_llama()
run_study3_binary_simple_deepseek()

# ==== STUDY 3 - BINARY ELABORATED
run_study3_binary_elaborated_gpt_3_5_turbo()
run_study3_binary_elaborated_gpt_4()
run_study3_binary_elaborated_gpt_4o()
run_study3_binary_elaborated_llama()
run_study3_binary_elaborated_deepseek()

# ==== STUDY 3 - LIKERT
run_study3_likert_gpt_3_5_turbo()
run_study3_likert_gpt_4()
run_study3_likert_gpt_4o()
run_study3_likert_llama()
run_study3_likert_deepseek()
