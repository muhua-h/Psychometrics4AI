# Explicit CFA Analysis for ALL Model × Domain × Format Combinations
# No loops - every combination written out explicitly
# Usage: Run line by line in R console

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
    cat("📏 Detected scale range: 1-", scale_max, "\n", sep = "")
    
    # Apply reverse coding
    existing_reverse <- intersect(REVERSE_ITEMS, names(data))
    if (length(existing_reverse) > 0) {
      cat("↩️  Reverse coding", length(existing_reverse), "items:\n")
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
      cat("⚠️  Removing near-zero variance items:", names(data)[near_zero], "\n")
      data <- data[, -near_zero]
    }
    
    cat("✅ Prepared:", nrow(data), "observations,", ncol(data), "items\n")
    return(data)
    
  }, error = function(e) {
    cat("❌ Error loading", basename(json_path), ":", e$message, "\n")
    return(NULL)
  })
}

# Function to run CFA for a single domain
run_domain_cfa <- function(domain, items, data, model_name, format_type, study_name) {
  tryCatch({
    # Clean item names to avoid special characters issues
    clean_items <- gsub("[^a-zA-Z0-9]", "_", items)
    names(data) <- gsub("[^a-zA-Z0-9]", "_", names(data))
    items <- clean_items[items %in% gsub("[^a-zA-Z0-9]", "_", items)]
    
    model_syntax <- paste(domain, "=~", paste(items, collapse = " + "))
    
    cat("\n📊 Running CFA for", toupper(domain), "domain\n")
    cat("   Items:", length(items), "-", paste(items, collapse = ", "), "\n")
    cat("   Model:", model_syntax, "\n")
    
    fit <- cfa(model_syntax, data = data, estimator = "MLR", std.lv = TRUE)
    
    if (!lavInspect(fit, "converged")) {
      cat("❌ Model did not converge\n")
      return(NULL)
    }
    
    fit_measures <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
    
    alpha <- psych::alpha(data[, items])$total$raw_alpha
    omega <- psych::omega(data[, items])$omega.tot
    
    cat("✅ Results for", toupper(domain), "\n")
    cat(sprintf("   Alpha: %.3f\n", alpha))
    cat(sprintf("   Omega: %.3f\n", omega))
    cat(sprintf("   CFI:   %.3f\n", fit_measures["cfi"]))
    cat(sprintf("   TLI:   %.3f\n", fit_measures["tli"]))
    cat(sprintf("   RMSEA: %.3f\n", fit_measures["rmsea"]))
    cat(sprintf("   SRMR:  %.3f\n", fit_measures["srmr"]))
    cat(sprintf("   Chi²:  %.3f (df=%d, p=%.3f)\n", 
                fit_measures["chisq"], fit_measures["df"], fit_measures["pvalue"]))
    
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
    cat("❌ CFA failed for", toupper(domain), ":", e$message, "\n")
    return(NULL)
  })
}

# Function to run complete analysis for one model/format/study
run_single_analysis <- function(json_path, model_name, format_type, study_name) {
  data <- load_and_prepare_data(json_path, model_name, format_type, study_name)
  if (is.null(data)) return(NULL)
  
  results <- list()
  
  # Run CFA for each domain
  for (domain in names(DOMAINS)) {
    items <- intersect(DOMAINS[[domain]], names(data))
    
    if (length(items) >= 3) {
      result <- run_domain_cfa(domain, items, data, model_name, format_type, study_name)
      if (!is.null(result)) {
        results[[length(results) + 1]] <- result
      }
    } else {
      cat("⚠️  Skipping", toupper(domain), "- insufficient items (", length(items), ")\n")
    }
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
  
  cat("\n💾 Results saved to:", output_file, "\n")
  return(final_results)
}

# ==============================================
# STUDY 2 - ALL MODEL × FORMAT COMBINATIONS
# ==============================================

# STUDY 2 - EXPANDED FORMAT
run_study2_expanded_gpt4 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_expanded_results", "bfi_to_minimarker_gpt_4_temp1_0.json"),
    model_name = "gpt_4",
    format_type = "expanded",
    study_name = "study_2"
  )
}

run_study2_expanded_gpt4o <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_expanded_results", "bfi_to_minimarker_gpt_4o_temp1_0.json"),
    model_name = "gpt_4o",
    format_type = "expanded",
    study_name = "study_2"
  )
}

run_study2_expanded_llama <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_expanded_results", "bfi_to_minimarker_llama_temp1_0.json"),
    model_name = "llama_3.3_70b",
    format_type = "expanded",
    study_name = "study_2"
  )
}

run_study2_expanded_deepseek <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_expanded_results", "bfi_to_minimarker_deepseek_temp1_0.json"),
    model_name = "deepseek_v3",
    format_type = "expanded",
    study_name = "study_2"
  )
}

run_study2_expanded_gpt35 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_expanded_results", "bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1_0.json"),
    model_name = "gpt_3.5_turbo",
    format_type = "expanded",
    study_name = "study_2"
  )
}

# STUDY 2 - LIKERT FORMAT
run_study2_likert_gpt4 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_likert_results", "bfi_to_minimarker_gpt_4.json"),
    model_name = "gpt_4",
    format_type = "likert",
    study_name = "study_2"
  )
}

run_study2_likert_gpt4o <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_likert_results", "bfi_to_minimarker_gpt_4o.json"),
    model_name = "gpt_4o",
    format_type = "likert",
    study_name = "study_2"
  )
}

run_study2_likert_llama <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_likert_results", "bfi_to_minimarker_llama.json"),
    model_name = "llama_3.3_70b",
    format_type = "likert",
    study_name = "study_2"
  )
}

run_study2_likert_deepseek <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_likert_results", "bfi_to_minimarker_deepseek.json"),
    model_name = "deepseek_v3",
    format_type = "likert",
    study_name = "study_2"
  )
}

run_study2_likert_gpt35 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_likert_results", "bfi_to_minimarker_openai_gpt_3.5_turbo_0125.json"),
    model_name = "gpt_3.5_turbo",
    format_type = "likert",
    study_name = "study_2"
  )
}

# STUDY 2 - BINARY SIMPLE FORMAT
run_study2_binary_simple_gpt4 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_simple_binary_results", "bfi_to_minimarker_binary_gpt_4_temp1_0.json"),
    model_name = "gpt_4",
    format_type = "binary_simple",
    study_name = "study_2"
  )
}

run_study2_binary_simple_gpt4o <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_simple_binary_results", "bfi_to_minimarker_binary_gpt_4o_temp1_0.json"),
    model_name = "gpt_4o",
    format_type = "binary_simple",
    study_name = "study_2"
  )
}

run_study2_binary_simple_llama <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_simple_binary_results", "bfi_to_minimarker_binary_llama_temp1_0.json"),
    model_name = "llama_3.3_70b",
    format_type = "binary_simple",
    study_name = "study_2"
  )
}

run_study2_binary_simple_deepseek <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_simple_binary_results", "bfi_to_minimarker_binary_deepseek_temp1_0.json"),
    model_name = "deepseek_v3",
    format_type = "binary_simple",
    study_name = "study_2"
  )
}

run_study2_binary_simple_gpt35 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_simple_binary_results", "bfi_to_minimarker_binary_openai_gpt_3.5_turbo_0125_temp1_0.json"),
    model_name = "gpt_3.5_turbo",
    format_type = "binary_simple",
    study_name = "study_2"
  )
}

# STUDY 2 - BINARY ELABORATED FORMAT
run_study2_binary_elaborated_gpt4 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_elaborated_binary_results", "bfi_to_minimarker_binary_gpt_4_temp1_0.json"),
    model_name = "gpt_4",
    format_type = "binary_elaborated",
    study_name = "study_2"
  )
}

run_study2_binary_elaborated_gpt4o <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_elaborated_binary_results", "bfi_to_minimarker_binary_gpt_4o_temp1_0.json"),
    model_name = "gpt_4o",
    format_type = "binary_elaborated",
    study_name = "study_2"
  )
}

run_study2_binary_elaborated_llama <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_elaborated_binary_results", "bfi_to_minimarker_binary_llama_temp1_0.json"),
    model_name = "llama_3.3_70b",
    format_type = "binary_elaborated",
    study_name = "study_2"
  )
}

run_study2_binary_elaborated_deepseek <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_elaborated_binary_results", "bfi_to_minimarker_binary_deepseek_temp1_0.json"),
    model_name = "deepseek_v3",
    format_type = "binary_elaborated",
    study_name = "study_2"
  )
}

run_study2_binary_elaborated_gpt35 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_2", "study_2_elaborated_binary_results", "bfi_to_minimarker_binary_openai_gpt_3.5_turbo_0125_temp1_0.json"),
    model_name = "gpt_3.5_turbo",
    format_type = "binary_elaborated",
    study_name = "study_2"
  )
}

# ==============================================
# STUDY 3 - ALL MODEL × FORMAT COMBINATIONS
# ==============================================

# STUDY 3 - EXPANDED FORMAT
run_study3_expanded_gpt4 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_expanded_results", "bfi_to_minimarker_gpt_4_temp1.json"),
    model_name = "gpt_4",
    format_type = "expanded",
    study_name = "study_3"
  )
}

run_study3_expanded_gpt4o <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_expanded_results", "bfi_to_minimarker_gpt_4o_temp1.json"),
    model_name = "gpt_4o",
    format_type = "expanded",
    study_name = "study_3"
  )
}

run_study3_expanded_llama <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_expanded_results", "bfi_to_minimarker_llama_temp1.json"),
    model_name = "llama_3.3_70b",
    format_type = "expanded",
    study_name = "study_3"
  )
}

run_study3_expanded_deepseek <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_expanded_results", "bfi_to_minimarker_deepseek_temp1.json"),
    model_name = "deepseek_v3",
    format_type = "expanded",
    study_name = "study_3"
  )
}

run_study3_expanded_gpt35 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_expanded_results", "bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1.json"),
    model_name = "gpt_3.5_turbo",
    format_type = "expanded",
    study_name = "study_3"
  )
}

# STUDY 3 - LIKERT FORMAT
run_study3_likert_gpt4 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_likert_results", "bfi_to_minimarker_gpt_4_temp1.json"),
    model_name = "gpt_4",
    format_type = "likert",
    study_name = "study_3"
  )
}

run_study3_likert_gpt4o <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_likert_results", "bfi_to_minimarker_gpt_4o_temp1.json"),
    model_name = "gpt_4o",
    format_type = "likert",
    study_name = "study_3"
  )
}

run_study3_likert_llama <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_likert_results", "bfi_to_minimarker_llama_temp1.json"),
    model_name = "llama_3.3_70b",
    format_type = "likert",
    study_name = "study_3"
  )
}

run_study3_likert_deepseek <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_likert_results", "bfi_to_minimarker_deepseek_temp1.json"),
    model_name = "deepseek_v3",
    format_type = "likert",
    study_name = "study_3"
  )
}

run_study3_likert_gpt35 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_likert_results", "bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1.json"),
    model_name = "gpt_3.5_turbo",
    format_type = "likert",
    study_name = "study_3"
  )
}

# STUDY 3 - BINARY SIMPLE FORMAT
run_study3_binary_simple_gpt4 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_binary_simple_results", "bfi_to_minimarker_binary_gpt_4_temp1.json"),
    model_name = "gpt_4",
    format_type = "binary_simple",
    study_name = "study_3"
  )
}

run_study3_binary_simple_gpt4o <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_binary_simple_results", "bfi_to_minimarker_binary_gpt_4o_temp1.json"),
    model_name = "gpt_4o",
    format_type = "binary_simple",
    study_name = "study_3"
  )
}

run_study3_binary_simple_llama <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_binary_simple_results", "bfi_to_minimarker_binary_llama_temp1.json"),
    model_name = "llama_3.3_70b",
    format_type = "binary_simple",
    study_name = "study_3"
  )
}

run_study3_binary_simple_deepseek <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_binary_simple_results", "bfi_to_minimarker_binary_deepseek_temp1.json"),
    model_name = "deepseek_v3",
    format_type = "binary_simple",
    study_name = "study_3"
  )
}

run_study3_binary_simple_gpt35 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_binary_simple_results", "bfi_to_minimarker_binary_openai_gpt_3.5_turbo_0125_temp1.json"),
    model_name = "gpt_3.5_turbo",
    format_type = "binary_simple",
    study_name = "study_3"
  )
}

# STUDY 3 - BINARY ELABORATED FORMAT
run_study3_binary_elaborated_gpt4 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_binary_elaborated_results", "bfi_to_minimarker_binary_gpt_4_temp1.json"),
    model_name = "gpt_4",
    format_type = "binary_elaborated",
    study_name = "study_3"
  )
}

run_study3_binary_elaborated_gpt4o <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_binary_elaborated_results", "bfi_to_minimarker_binary_gpt_4o_temp1.json"),
    model_name = "gpt_4o",
    format_type = "binary_elaborated",
    study_name = "study_3"
  )
}

run_study3_binary_elaborated_llama <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_binary_elaborated_results", "bfi_to_minimarker_binary_llama_temp1.json"),
    model_name = "llama_3.3_70b",
    format_type = "binary_elaborated",
    study_name = "study_3"
  )
}

run_study3_binary_elaborated_deepseek <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_binary_elaborated_results", "bfi_to_minimarker_binary_deepseek_temp1.json"),
    model_name = "deepseek_v3",
    format_type = "binary_elaborated",
    study_name = "study_3"
  )
}

run_study3_binary_elaborated_gpt35 <- function() {
  run_single_analysis(
    json_path = file.path(BASE_DIR, "study_3", "study_3_binary_elaborated_results", "bfi_to_minimarker_binary_openai_gpt_3.5_turbo_0125_temp1.json"),
    model_name = "gpt_3.5_turbo",
    format_type = "binary_elaborated",
    study_name = "study_3"
  )
}

# ==============================================
# USAGE EXAMPLES - COPY/PASTE INTO R CONSOLE
# ==============================================

# Quick test examples:
# run_study2_expanded_gpt4()
# run_study2_expanded_gpt4o()
# run_study3_likert_deepseek()

# Run all combinations for a specific study and format:
run_study2_expanded_gpt4()
run_study2_expanded_gpt4o()
run_study2_expanded_llama()
run_study2_expanded_deepseek()
run_study2_expanded_gpt35()

# Check if files exist
# file.exists("/Users/mhhuang/Psychometrics4AI_revision/multi_model_studies/study_2/study_2_expanded_results/bfi_to_minimarker_gpt_4_temp1_0.json")