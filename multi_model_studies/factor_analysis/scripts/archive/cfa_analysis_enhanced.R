#!/usr/bin/env Rscript

# Enhanced R-based Confirmatory Factor Analysis
# Comprehensive CFA with proper fit indices and reliability measures
# Designed specifically for multi-model psychometric validation

suppressPackageStartupMessages({
  library(lavaan)
  library(psych)
  library(semTools)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(readr)
})

# Enhanced Big Five domains with comprehensive item mapping
BIG_FIVE_DOMAINS <- list(
  Extraversion = list(
    positive = c("Bold", "Energetic", "Extraverted", "Talkative"),
    negative = c("Bashful", "Quiet", "Shy", "Withdrawn")
  ),
  Agreeableness = list(
    positive = c("Cooperative", "Kind", "Sympathetic", "Warm"),
    negative = c("Cold", "Harsh", "Rude", "Unsympathetic")
  ),
  Conscientiousness = list(
    positive = c("Efficient", "Organized", "Practical", "Systematic"),
    negative = c("Careless", "Disorganized", "Inefficient", "Sloppy")
  ),
  Neuroticism = list(
    positive = c("Envious", "Fretful", "Jealous", "Moody", "Temperamental", "Touchy"),
    negative = c("Relaxed", "Unenvious")
  ),
  Openness = list(
    positive = c("Complex", "Creative", "Deep", "Imaginative", "Intellectual", "Philosophical"),
    negative = c("Uncreative", "Unintellectual")
  )
)

# Comprehensive reverse coding function with validation
reverse_code_items <- function(data, scale_range = c(1, 9), verbose = FALSE) {
  negative_items <- c(
    "Bashful", "Quiet", "Shy", "Withdrawn",
    "Cold", "Harsh", "Rude", "Unsympathetic", 
    "Careless", "Disorganized", "Inefficient", "Sloppy",
    "Relaxed", "Unenvious",
    "Uncreative", "Unintellectual"
  )
  
  negative_available <- negative_items[negative_items %in% names(data)]
  
  if (length(negative_available) > 0) {
    if (verbose) {
      cat("  Reversing items:", paste(negative_available, collapse = ", "), "\n")
    }
    
    # Validate scale range
    actual_range <- range(data[negative_available], na.rm = TRUE)
    if (any(actual_range < scale_range[1] | actual_range > scale_range[2], na.rm = TRUE)) {
      warning("Scale range mismatch detected. Adjusting...")
    }
    
    data[negative_available] <- scale_range[2] + scale_range[1] - data[negative_available]
  }
  
  return(data)
}

# Advanced reliability calculation with multiple indices
calculate_reliabilities <- function(data, items) {
  if (length(items) < 2) {
    return(list(
      alpha = NA, omega = NA, omega_hierarchical = NA, glb = NA,
      mean_inter_item = NA, item_total = NA
    ))
  }
  
  tryCatch({
    # Cronbach's Alpha (most stable)
    alpha_result <- psych::alpha(data[items], check.keys = TRUE)
    alpha <- alpha_result$total$raw_alpha
    
    # McDonald's Omega (with error handling)
    omega <- tryCatch({
      omega_result <- psych::omega(data[items], check.keys = TRUE)
      omega_result$omega.total
    }, error = function(e) NA)
    
    # Other reliability measures (with error handling)
    omega_hierarchical <- tryCatch({
      omega_result <- psych::omega(data[items], check.keys = TRUE)
      omega_result$omega.hierarchical
    }, error = function(e) NA)
    
    glb <- tryCatch({
      glb_result <- psych::glb(data[items])
      glb_result$glb
    }, error = function(e) NA)
    
    # Item-total correlations
    item_total <- alpha_result$item.stats$raw.r
    
    list(
      alpha = alpha,
      omega = omega,
      omega_hierarchical = omega_hierarchical,
      glb = glb,
      mean_inter_item = mean(alpha_result$total$mean, na.rm = TRUE),
      item_total = mean(item_total, na.rm = TRUE)
    )
  }, error = function(e) {
    logger(paste("Reliability calculation failed:", as.character(e)))
    list(
      alpha = NA, omega = NA, omega_hierarchical = NA, glb = NA,
      mean_inter_item = NA, item_total = NA
    )
  })
}

# Simple logger function
logger <- function(message) {
  cat(message, "\n")
}

# Enhanced CFA analysis with comprehensive fit indices
analyze_domain_cfa <- function(data, domain, scale_range = c(1, 9), 
                              estimator = "MLR", missing = "fiml") {
  
  items <- c(BIG_FIVE_DOMAINS[[domain]]$positive, BIG_FIVE_DOMAINS[[domain]]$negative)
  available_items <- items[items %in% names(data)]
  
  if (length(available_items) < 3) {
    return(list(error = paste("Insufficient items for", domain, "- only", length(available_items), "available")))
  }
  
  # Get negative items for reverse coding
  negative_items <- BIG_FIVE_DOMAINS[[domain]]$negative
  negative_items <- negative_items[negative_items %in% available_items]
  
  # Apply reverse coding
  data_processed <- reverse_code_items(data, scale_range)
  domain_data <- data_processed[available_items]
  
  # Handle missing data
  domain_data <- na.omit(domain_data)
  
  if (nrow(domain_data) < 10) {
    return(list(error = paste("Insufficient participants for", domain)))
  }
  
  # Create CFA model specification
  latent_var <- domain
  model_spec <- paste(latent_var, "=~", paste(available_items, collapse = " + "))
  
  tryCatch({
    # Fit CFA model with robust estimator
    fit <- cfa(model_spec, data = domain_data, std.lv = TRUE, 
               estimator = estimator, missing = missing)
    
    # Comprehensive fit indices
    fit_measures <- fitMeasures(fit, c(
      "chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr",
      "aic", "bic", "logl", "npar",
      "rmsea.ci.lower", "rmsea.ci.upper", "rmsea.pvalue"
    ))
    
    # Standardized loadings with significance testing
    loadings <- standardizedSolution(fit)
    loadings <- loadings[loadings$op == "=~", c("rhs", "est.std", "se", "z", "pvalue")]
    colnames(loadings) <- c("item", "loading", "se", "z", "pvalue")
    
    # Reliability measures
    reliabilities <- calculate_reliabilities(domain_data, available_items)
    
    # Calculate additional statistics
    loadings_vec <- loadings$loading
    eigenvalue <- sum(loadings_vec^2)
    variance_explained <- eigenvalue / length(available_items)
    
    # Average variance extracted (AVE)
    ave <- mean(loadings_vec^2)
    
    # Composite reliability (CR)
    cr <- (sum(loadings_vec)^2) / (sum(loadings_vec)^2 + sum(1 - loadings_vec^2))
    
    # Model convergence check
    converged <- lavInspect(fit, "converged")
    
    list(
      domain = domain,
      n_items = length(available_items),
      n_participants = nrow(domain_data),
      items_used = available_items,
      
      # Fit indices
      chisq = fit_measures["chisq"],
      df = fit_measures["df"],
      pvalue = fit_measures["pvalue"],
      cfi = fit_measures["cfi"],
      tli = fit_measures["tli"],
      rmsea = fit_measures["rmsea"],
      rmsea_ci_lower = fit_measures["rmsea.ci.lower"],
      rmsea_ci_upper = fit_measures["rmsea.ci.upper"],
      rmsea_pvalue = fit_measures["rmsea.pvalue"],
      srmr = fit_measures["srmr"],
      aic = fit_measures["aic"],
      bic = fit_measures["bic"],
      
      # Reliability
      alpha = reliabilities$alpha,
      omega = reliabilities$omega,
      omega_hierarchical = reliabilities$omega_hierarchical,
      glb = reliabilities$glb,
      composite_reliability = cr,
      ave = ave,
      
      # Loadings
      loadings = loadings,
      eigenvalue = eigenvalue,
      variance_explained = variance_explained,
      mean_loading = mean(abs(loadings_vec)),
      min_loading = min(abs(loadings_vec)),
      max_loading = max(abs(loadings_vec)),
      
      # Model diagnostics
      converged = converged,
      estimator = estimator,
      missing_handling = missing
    )
    
  }, error = function(e) {
    list(error = as.character(e))
  })
}

# Enhanced file processing with comprehensive metadata
analyze_simulation_file <- function(json_path, output_dir, scale_range = c(1, 9)) {
  
  # Create output directory structure
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  tryCatch({
    # Load and validate data
    data <- fromJSON(json_path)
    if (is.list(data) && length(data) > 0 && is.list(data[[1]])) {
      # Handle array of objects
      data <- as.data.frame(data)
    } else {
      # Handle single object
      data <- as.data.frame(data)
    }
    
    # Extract metadata from filename and path
    model_name <- tools::file_path_sans_ext(basename(json_path))
    
    # Parse study and format information from path
    path_parts <- strsplit(json_path, "/")[[1]]
    
    # Extract study info
    study_patterns <- c("study_2", "study_3", "study_4")
    study <- "UNKNOWN"
    for (pattern in study_patterns) {
      if (any(grepl(pattern, path_parts))) {
        study <- toupper(pattern)
        break
      }
    }
    
    # Extract format info
    format_patterns <- c("binary_simple", "binary_elaborated", "expanded", "likert", "elaborated_binary", "simple_binary")
    format <- "UNKNOWN"
    for (pattern in format_patterns) {
      if (any(grepl(pattern, path_parts))) {
        format <- ifelse(pattern %in% c("elaborated_binary"), "binary_elaborated",
                   ifelse(pattern %in% c("simple_binary"), "binary_simple", pattern))
        break
      }
    }
    
    # Validate scale range based on data
    all_values <- unlist(data)
    actual_range <- range(all_values, na.rm = TRUE)
    
    # Adjust scale range if needed
    if (actual_range[1] >= 1 && actual_range[2] <= 5) {
      scale_range <- c(1, 5)
      cat("Detected 5-point scale, adjusting range to", scale_range, "\n")
    } else if (actual_range[1] >= 1 && actual_range[2] <= 7) {
      scale_range <- c(1, 7)
      cat("Detected 7-point scale, adjusting range to", scale_range, "\n")
    } else {
      cat("Using standard 9-point scale range", scale_range, "\n")
    }
    
    # Apply reverse coding
    all_negative_items <- c(
      "Bashful", "Quiet", "Shy", "Withdrawn",
      "Cold", "Harsh", "Rude", "Unsympathetic",
      "Careless", "Disorganized", "Inefficient", "Sloppy", 
      "Relaxed", "Unenvious",
      "Uncreative", "Unintellectual"
    )
    
    negative_in_data <- all_negative_items[all_negative_items %in% names(data)]
    if (length(negative_in_data) > 0) {
      data <- reverse_code_items(data, scale_range)
    }
    
    # Analyze each domain
    results <- list()
    domain_summaries <- list()
    
    cat("Analyzing", model_name, "from", study, "-", format, "\n")
    
    for (domain in names(BIG_FIVE_DOMAINS)) {
      cat("  Processing", domain, "domain...")
      
      domain_result <- analyze_domain_cfa(data, domain, scale_range)
      results[[domain]] <- domain_result
      
      if (!"error" %in% names(domain_result)) {
        # Create summary row
        summary_row <- data.frame(
          Study = study,
          Format = format,
          Model = model_name,
          Structure_Type = "Original",
          Factor_Domain = domain,
          N_Items = domain_result$n_items,
          N_Participants = domain_result$n_participants,
          Alpha = domain_result$alpha,
          Omega = domain_result$omega,
          Omega_Hierarchical = domain_result$omega_hierarchical,
          GLB = domain_result$glb,
          Composite_Reliability = domain_result$composite_reliability,
          AVE = domain_result$ave,
          Eigenvalue = domain_result$eigenvalue,
          Variance_Explained = domain_result$variance_explained,
          Mean_Loading = domain_result$mean_loading,
          Min_Loading = domain_result$min_loading,
          Max_Loading = domain_result$max_loading,
          Chi_Square = domain_result$chisq,
          DF = domain_result$df,
          P_Value = domain_result$pvalue,
          CFI = domain_result$cfi,
          TLI = domain_result$tli,
          RMSEA = domain_result$rmsea,
          RMSEA_CI_Lower = domain_result$rmsea_ci_lower,
          RMSEA_CI_Upper = domain_result$rmsea_ci_upper,
          RMSEA_P_Value = domain_result$rmsea_pvalue,
          SRMR = domain_result$srmr,
          AIC = domain_result$aic,
          BIC = domain_result$bic,
          Converged = domain_result$converged,
          Estimator = domain_result$estimator,
          Missing_Handling = domain_result$missing_handling
        )
        
        domain_summaries[[domain]] <- summary_row
        cat(" ✅ Omega:", round(domain_result$omega, 3), 
            "CFI:", round(domain_result$cfi, 3), 
            "RMSEA:", round(domain_result$rmsea, 3), "\n")
      } else {
        cat(" ❌", domain_result$error, "\n")
      }
    }
    
    if (length(domain_summaries) > 0) {
      # Combine all domain results
      final_summary <- bind_rows(domain_summaries)
      
      # Save comprehensive results
      summary_file <- file.path(output_dir, paste0(model_name, "_factor_summary_R.csv"))
      write_csv(final_summary, summary_file)
      
      # Save detailed results with loadings
      detailed_results <- list(
        summary = final_summary,
        loadings_by_domain = map(names(results), function(domain) {
          if (!"error" %in% names(results[[domain]])) {
            results[[domain]]$loadings
          } else {
            NULL
          }
        }),
        metadata = list(
          study = study,
          format = format,
          model = model_name,
          file_source = basename(json_path),
          analysis_timestamp = Sys.time(),
          scale_range = scale_range
        )
      )
      
      detailed_file <- file.path(output_dir, paste0(model_name, "_detailed_results_R.json"))
      write_json(detailed_results, detailed_file, auto_unbox = TRUE, pretty = TRUE)
      
      # Save loadings separately for easy access
      loadings_file <- file.path(output_dir, paste0(model_name, "_factor_loadings_R.csv"))
      loadings_df <- map_df(names(results), function(domain) {
        if (!"error" %in% names(results[[domain]])) {
          loadings <- results[[domain]]$loadings
          loadings$Domain <- domain
          loadings
        }
      })
      
      if (nrow(loadings_df) > 0) {
        write_csv(loadings_df, loadings_file)
      }
      
      cat("Analysis complete for", model_name, "\n")
      cat("Summary saved to:", summary_file, "\n")
      cat("Detailed results saved to:", detailed_file, "\n")
      
      return(final_summary)
    }
    
    return(NULL)
    
  }, error = function(e) {
    cat("Error processing", json_path, ":", as.character(e), "\n")
    return(NULL)
  })
}

# Batch processing function
process_all_simulations <- function(base_dir = "../results", output_base_dir = "../results_r") {
  
  # Find all JSON files across studies
  json_files <- list.files(base_dir, pattern = "\\.json$", recursive = TRUE, full.names = TRUE)
  
  cat("Found", length(json_files), "JSON files to process\n")
  
  results_summary <- list()
  
  for (json_path in json_files) {
    
    # Determine output directory based on path structure
    path_parts <- strsplit(json_path, "/")[[1]]
    
    # Extract study and format
    study_idx <- grep("study_[234]", path_parts)
    format_idx <- grep("_format", path_parts)
    
    if (length(study_idx) > 0 && length(format_idx) > 0) {
      study <- path_parts[study_idx]
      format <- gsub("_results.*", "", path_parts[format_idx])
      
      output_dir <- file.path(output_base_dir, study, paste0(format, "_format"))
      
      cat("Processing:", basename(json_path), "->", output_dir, "\n")
      
      result <- analyze_simulation_file(json_path, output_dir)
      
      if (!is.null(result)) {
        results_summary[[paste(study, format, basename(json_path), sep = "_")]] <- result
      }
    }
  }
  
  # Create overall summary
  if (length(results_summary) > 0) {
    all_results <- bind_rows(results_summary)
    overall_summary_file <- file.path(output_base_dir, "overall_factor_analysis_summary_R.csv")
    write_csv(all_results, overall_summary_file)
    
    cat("\n=== ANALYSIS COMPLETE ===\n")
    cat("Total files processed:", length(results_summary), "\n")
    cat("Overall summary saved to:", overall_summary_file, "\n")
    
    # Print summary statistics
    cat("\nSummary Statistics:\n")
    cat("Average Omega:", round(mean(all_results$Omega, na.rm = TRUE), 3), "\n")
    cat("Average CFI:", round(mean(all_results$CFI, na.rm = TRUE), 3), "\n")
    cat("Average RMSEA:", round(mean(all_results$RMSEA, na.rm = TRUE), 3), "\n")
    cat("Average Alpha:", round(mean(all_results$Alpha, na.rm = TRUE), 3), "\n")
  }
  
  return(results_summary)
}

# Command line interface
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 1) {
    cat("Usage: Rscript cfa_analysis_enhanced.R <mode> [args...]\n")
    cat("Modes:\n")
    cat("  single <json_path> <output_dir> [scale_min] [scale_max] - Process single file\n")
    cat("  batch <base_dir> <output_base_dir> - Process all JSON files\n")
    quit(status = 1)
  }
  
  mode <- args[1]
  
  if (mode == "single") {
    if (length(args) < 3) {
      cat("Usage: Rscript cfa_analysis_enhanced.R single <json_path> <output_dir> [scale_min] [scale_max]\n")
      quit(status = 1)
    }
    
    json_path <- args[2]
    output_dir <- args[3]
    scale_min <- ifelse(length(args) >= 4, as.numeric(args[4]), 1)
    scale_max <- ifelse(length(args) >= 5, as.numeric(args[5]), 9)
    
    result <- analyze_simulation_file(json_path, output_dir, c(scale_min, scale_max))
    
    if (!is.null(result)) {
      cat("\n=== ANALYSIS COMPLETE ===\n")
      print(result[, c("Factor_Domain", "N_Items", "N_Participants", "Alpha", "Omega", "CFI", "RMSEA")])
    }
    
  } else if (mode == "batch") {
    base_dir <- ifelse(length(args) >= 2, args[2], "../results")
    output_base_dir <- ifelse(length(args) >= 3, args[3], "../results_r")
    
    process_all_simulations(base_dir, output_base_dir)
    
  } else {
    cat("Invalid mode. Use 'single' or 'batch'\n")
    quit(status = 1)
  }
}