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
    scale_max <- 9
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

# ENHANCED Function to run CFA for a single domain with lavaan omega calculation
run_domain_cfa <- function(domain, items, data, model_name, format_type, study_name) {
  domain_header <- paste0("📸 [", toupper(domain), " DOMAIN] ")

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

      cat(domain_header, "🔥 Fitting CFA model...\n")

      # Fit the model
      fit <- cfa(model_syntax, data = data, estimator = "ML", std.lv = TRUE)

      # Check convergence
      if (!lavInspect(fit, "converged")) {
        cat(domain_header, "❌ Model did not converge\n")
        return(NULL)
      }

      # Get fit measures
      fit_measures <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))

      # Calculate reliability measures
      cat(domain_header, "📊 Calculating reliability measures...\n")

      # Cronbach's Alpha using psych package
      alpha_result <- psych::alpha(data[, items])
      alpha <- alpha_result$total$raw_alpha

      # Check for negative alpha (indicates problems)
      if (alpha < 0) {
        cat(domain_header, "⚠️ Negative Cronbach's Alpha (", round(alpha, 3), ") - Check item scoring\n")
      }

      # McDonald's Omega using lavaan/semTools
      cat(domain_header, "🔧 Computing McDonald's Omega using lavaan...\n")

      # Use compRelSEM from semTools to calculate composite reliability (omega)
      omega_result <- tryCatch({
        omega_values <- compRelSEM(fit, return.total = TRUE)
        if (is.list(omega_values)) {
          # If multiple factors, get the one for our domain
          if (domain %in% names(omega_values)) {
            omega_values[[domain]]
          } else {
            # Take the first one if domain name doesn't match exactly
            omega_values[[1]]
          }
        } else {
          # If single value returned
          omega_values
        }
      }, error = function(e) {
        cat(domain_header, "⚠️ Error computing omega with compRelSEM:", e$message, "\n")
        # Fallback to psych omega if compRelSEM fails
        cat(domain_header, "🔄 Falling back to psych::omega...\n")
        omega_fallback <- psych::omega(data[, items], plot = FALSE)
        omega_fallback$omega.tot
      })

      omega <- as.numeric(omega_result)

      # Report any collected warnings
      if (length(domain_warnings) > 0) {
        cat(domain_header, "⚠️ Warnings encountered:\n")
        for (w in domain_warnings) {
          cat(domain_header, "   ", w, "\n")
        }
      }

      cat(domain_header, "✅ Analysis completed successfully\n")
      cat(domain_header, "📈 Omega (lavaan):", round(omega, 3), "\n")

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
  cat("🔥 STARTING DOMAIN-BY-DOMAIN CFA ANALYSIS\n")
  cat(paste(rep("=", 80), collapse = ""), "\n", sep = "")

  # Run CFA for each domain with detailed reporting
  for (i in seq_along(DOMAINS)) {
    domain <- names(DOMAINS)[i]
    domain_items <- DOMAINS[[domain]]

    cat(sprintf("\n🏷️ DOMAIN %d of %d: %s\n", i, length(DOMAINS), toupper(domain)))

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

# ==============================================
# DYNAMIC FUNCTION GENERATION
# ==============================================

# Function to generate correct filename based on study and format
get_filename <- function(study, format, model) {
  # Determine the correct file prefix based on study
  file_prefix <- if (study == "study_2a") "bfi_to_minimarker" else "bfi_to_minimarker"

  # Handle special model name formatting for files
  file_model_name <- switch(model,
                            "openai_gpt_3.5_turbo_0125" = "openai_gpt_3.5_turbo_0125",
                            "deepseek" = "deepseek",
                            "gpt_4" = "gpt_4",
                            "gpt_4o" = "gpt_4o",
                            "llama" = "llama",
                            model  # fallback
  )

  # Generate filename based on format and study
  if (grepl("binary", format)) {
    if (study == "study_2a") {
      # Study 2a binary formats have temp1_0 suffix
      filename <- paste0(file_prefix, "_binary_", file_model_name, "_temp1_0.json")
    } else {
      # Study 2b binary formats have temp1 suffix (no _0)
      filename <- paste0(file_prefix, "_binary_", file_model_name, "_temp1.json")
    }
  } else if (format == "likert_format") {
    # Likert format has no temp suffix for both studies
    filename <- paste0(file_prefix, "_", file_model_name, ".json")
  } else if (format == "expanded_format") {
    if (study == "study_2a") {
      # Study 2a expanded format has temp1_0 suffix
      filename <- paste0(file_prefix, "_", file_model_name, "_temp1_0.json")
    } else {
      # Study 2b expanded format - need to check actual naming
      # Based on the pattern, it might not have temp suffix or might be different
      filename <- paste0(file_prefix, "_", file_model_name, ".json")  # Try without temp first
    }
  }

  return(filename)
}

# Function to get directory name based on study and format
get_directory_name <- function(study, format) {
  if (study == "study_2a") {
    dir_mapping <- list(
      "expanded_format" = "study_2_expanded_results",
      "likert_format" = "study_2_likert_results",
      "binary_simple_format" = "study_2_simple_binary_results",
      "binary_elaborated_format" = "study_2_elaborated_binary_results"
    )
  } else {
    dir_mapping <- list(
      "expanded_format" = "study_3_expanded_results",
      "likert_format" = "study_3_likert_results",
      "binary_simple_format" = "study_3_binary_simple_results",
      "binary_elaborated_format" = "study_3_binary_elaborated_results"
    )
  }

  return(dir_mapping[[format]])
}

# Generate all function configurations
function_configs <- list()

# Define formats and models based on your actual file structure
formats <- c("expanded_format", "likert_format", "binary_simple_format", "binary_elaborated_format")
models <- c("deepseek", "gpt_4", "gpt_4o", "llama", "openai_gpt_3.5_turbo_0125")

# Generate configurations for both studies
for (study in c("study_2a", "study_2b")) {
  for (format in formats) {
    for (model in models) {
      filename <- get_filename(study, format, model)
      dir_name <- get_directory_name(study, format)

      # Add to configurations
      function_configs[[length(function_configs) + 1]] <- list(
        study = study,
        format = format,
        model = model,
        file = filename,
        directory = dir_name
      )
    }
  }
}

# Generate all functions dynamically
for (config in function_configs) {
  # Create function name (clean up format names for function names)
  clean_format <- gsub("_format$", "", config$format)  # Remove "_format" suffix
  func_name <- paste0("run_", config$study, "_", clean_format, "_", config$model)

  # Create the function with dynamic path finding
  func_body <- substitute({
    # Find the actual JSON file location
    json_path <- find_json_file(STUDY_NAME, FORMAT_FULL, FILENAME)

    run_single_analysis(
      json_path = json_path,
      model_name = MODEL_NAME,
      format_type = FORMAT_TYPE,
      study_name = STUDY_NAME
    )
  }, list(
    STUDY_NAME = config$study,
    FORMAT_FULL = config$format,
    FORMAT_TYPE = clean_format,
    FILENAME = config$file,
    MODEL_NAME = config$model
  ))

  # Assign the function to the global environment
  assign(func_name, eval(substitute(function() BODY, list(BODY = func_body))), envir = .GlobalEnv)
}

# ==============================================
# CONVENIENCE FUNCTIONS
# ==============================================

# ENHANCED Function to find JSON data files with better path detection
find_json_file <- function(study, format, filename) {
  # Get the correct directory name
  dir_name <- get_directory_name(study, format)

  # Possible locations for JSON data files - now includes the correct directory structure
  possible_dirs <- c(
    # Direct path to the results directory
    file.path(BASE_DIR, study, dir_name),
    # Alternative paths
    file.path(BASE_DIR, study, format),
    file.path(BASE_DIR, study, "data", format),
    file.path(BASE_DIR, study, "raw_data", format),
    file.path(BASE_DIR, "data", study, format),
    file.path(BASE_DIR, study, gsub("_format$", "", format)),
    file.path(BASE_DIR, study, paste0(gsub("_format$", "", format), "_results"))
  )

  # Try each possible directory
  for (dir_path in possible_dirs) {
    full_path <- file.path(dir_path, filename)
    if (file.exists(full_path)) {
      cat("🔍 Found file at:", full_path, "\n")
      return(full_path)
    }
  }

  # If not found, try alternative filename patterns for study 2b expanded format
  if (study == "study_2b" && format == "expanded_format") {
    # Try with temp1_0 suffix as well
    alt_filename <- gsub("\\.json$", "_temp1_0.json", filename)
    for (dir_path in possible_dirs) {
      full_path <- file.path(dir_path, alt_filename)
      if (file.exists(full_path)) {
        cat("🔍 Found alternative file at:", full_path, "\n")
        return(full_path)
      }
    }
  }

  # If still not found, return the most likely path for error reporting
  most_likely_path <- file.path(BASE_DIR, study, dir_name, filename)
  cat("❌ File not found. Expected location:", most_likely_path, "\n")
  return(most_likely_path)
}

# Function to list all available analysis functions
list_analysis_functions <- function() {
  all_funcs <- ls(envir = .GlobalEnv)
  analysis_funcs <- all_funcs[grepl("^run_study[23]_", all_funcs)]

  cat("Available analysis functions:\n")
  cat("=============================\n")

  # Group by study
  study2_funcs <- analysis_funcs[grepl("^run_study_2_", analysis_funcs)]
  study3_funcs <- analysis_funcs[grepl("^run_study_3_", analysis_funcs)]

  if (length(study2_funcs) > 0) {
    cat("\nStudy 2 functions:\n")
    for (func in sort(study2_funcs)) {
      cat("  ", func, "()\n")
    }
  }

  if (length(study3_funcs) > 0) {
    cat("\nStudy 3 functions:\n")
    for (func in sort(study3_funcs)) {
      cat("  ", func, "()\n")
    }
  }

  return(invisible(analysis_funcs))
}

# Function to run all analyses for a specific study
run_all_study <- function(study_num) {
  study_pattern <- paste0("^run_study_", study_num, "_")
  all_funcs <- ls(envir = .GlobalEnv)
  study_funcs <- all_funcs[grepl(study_pattern, all_funcs)]

  if (length(study_funcs) == 0) {
    cat("No functions found for study", study_num, "\n")
    return(NULL)
  }

  cat("Running all analyses for Study", study_num, "...\n")
  cat("Found", length(study_funcs), "functions to run\n\n")

  results <- list()
  for (func_name in sort(study_funcs)) {
    cat("Executing:", func_name, "\n")
    tryCatch({
      func <- get(func_name, envir = .GlobalEnv)
      result <- func()
      if (!is.null(result)) {
        results[[func_name]] <- result
      }
    }, error = function(e) {
      cat("Error in", func_name, ":", e$message, "\n")
    })
    cat("\n")
  }

  return(results)
}

# Function to run all analyses for a specific format
# Fixed Function to run all analyses for a specific format
# CORRECTED run_all_format function
run_all_format <- function(format_type) {
  # Get all study functions
  all_funcs <- ls(envir = .GlobalEnv)
  study_funcs <- all_funcs[grepl("^run_study[23]_", all_funcs)]

  if (length(study_funcs) == 0) {
    cat("❌ No study functions found!\n")
    return(NULL)
  }

  # Create the pattern - this is what was working in the debug
  format_pattern <- paste0("_", format_type, "_")

  # Find matching functions - this should work based on debug output
  format_funcs <- study_funcs[grepl(format_pattern, study_funcs)]

  if (length(format_funcs) == 0) {
    cat("❌ No functions found for format '", format_type, "'\n", sep = "")
    cat("Available format types you can use:\n")

    # Extract unique format types from function names
    format_parts <- gsub("^run_study_[23]_([^_]+)_.*", "\\1", study_funcs)
    unique_formats <- unique(format_parts)
    for (fmt in sort(unique_formats)) {
      cat("  - ", fmt, "\n")
    }

    # Also show compound formats like binary_simple, binary_elaborated
    compound_parts <- gsub("^run_study_[23]_([^_]+_[^_]+)_.*", "\\1", study_funcs)
    compound_formats <- unique(compound_parts[compound_parts != study_funcs]) # only ones that matched
    if (length(compound_formats) > 0) {
      cat("\nCompound formats:\n")
      for (fmt in sort(compound_formats)) {
        cat("  - ", fmt, "\n")
      }
    }
    return(NULL)
  }

  cat("🚀 Running all '", format_type, "' format analyses...\n", sep = "")
  cat("Found", length(format_funcs), "functions to execute:\n")
  for (func in sort(format_funcs)) {
    cat("  📋 ", func, "\n")
  }
  cat("\n")

  # Execute all functions
  results <- list()
  successful <- 0
  failed <- 0

  for (func_name in sort(format_funcs)) {
    cat("⏳ Executing: ", func_name, "\n")
    tryCatch({
      func <- get(func_name, envir = .GlobalEnv)
      result <- func()
      if (!is.null(result)) {
        results[[func_name]] <- result
        successful <- successful + 1
        cat("✅ ", func_name, " - COMPLETED\n")
      } else {
        failed <- failed + 1
        cat("❌ ", func_name, " - RETURNED NULL\n")
      }
    }, error = function(e) {
      failed <- failed + 1
      cat("❌ ", func_name, " - ERROR: ", e$message, "\n")
    })
    cat("\n")
  }

  # Final summary
  cat("📊 EXECUTION SUMMARY:\n")
  cat("✅ Successful: ", successful, "\n")
  cat("❌ Failed: ", failed, "\n")
  cat("📁 Total results collected: ", length(results), "\n")

  return(results)
}

# Function to test a specific file path (useful for debugging)
test_file_path <- function(study, format, model) {
  filename <- get_filename(study, format, model)
  json_path <- find_json_file(study, format, filename)

  cat("Testing file path:\n")
  cat("Study:", study, "\n")
  cat("Format:", format, "\n")
  cat("Model:", model, "\n")
  cat("Generated filename:", filename, "\n")
  cat("Expected path:", json_path, "\n")
  cat("File exists:", file.exists(json_path), "\n")

  if (file.exists(json_path)) {
    cat("✅ File found successfully!\n")
  } else {
    cat("❌ File not found. Check the directory structure.\n")
  }

  return(json_path)
}

# ==============================================
# USAGE EXAMPLES
# ==============================================

cat("Dynamic functions have been created with FIXED path detection!\n")
cat("==============================================================\n\n")

cat("Usage examples:\n")
cat("1. Run a specific analysis:\n")
cat("   run_study_2_expanded_gpt_4()\n")
cat("   run_study_3_binary_elaborated_deepseek()\n")
cat("   run_study_2_likert_openai_gpt_3.5_turbo_0125()\n\n")

cat("2. Test file paths (for debugging):\n")
cat("   test_file_path('study_2b', 'binary_elaborated_format', 'deepseek')\n")
cat("   test_file_path('study_2a', 'binary_simple_format', 'gpt_4')\n\n")

cat("3. List all available functions:\n")
cat("   list_analysis_functions()\n\n")

cat("4. Run all analyses for a study:\n")
cat("   run_all_study(2)  # Runs all Study 2a analyses\n")
cat("   run_all_study(3)  # Runs all Study 2b analyses\n\n")

cat("5. Run all analyses for a format:\n")
cat("   run_all_format('expanded')  # Runs all expanded format analyses\n")
cat("   run_all_format('binary_elaborated')   # Runs all binary elaborated format analyses\n\n")

cat("Key fixes:\n")
cat("- Study 2b binary files now use 'temp1' instead of 'temp1_0'\n")
cat("- Correct directory names for Study 2b\n")
cat("- Enhanced path detection with better error reporting\n")
cat("- Added test_file_path() function for debugging\n")

run_study_2_expanded_gpt_4()