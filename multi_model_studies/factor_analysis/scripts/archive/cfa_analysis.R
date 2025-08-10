#!/usr/bin/env Rscript

# R-based Confirmatory Factor Analysis for Multi-Model Psychometrics
# Uses lavaan package for proper CFA with fit indices and reliability

suppressPackageStartupMessages({
  library(lavaan)
  library(psych)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(purrr)
})

# Define Big Five domains and their Mini-Marker items
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

# Reverse coding function
reverse_code <- function(data, items_to_reverse, scale_min = 1, scale_max = 9) {
  data[items_to_reverse] <- scale_max + scale_min - data[items_to_reverse]
  return(data)
}

# Get all items for a domain
domain_items <- function(domain) {
  domain_config <- BIG_FIVE_DOMAINS[[domain]]
  c(domain_config$positive, domain_config$negative)
}

# Get negative items for reverse coding
get_negative_items <- function() {
  negative_items <- c()
  for (domain in names(BIG_FIVE_DOMAINS)) {
    negative_items <- c(negative_items, BIG_FIVE_DOMAINS[[domain]]$negative)
  }
  return(unique(negative_items))
}

# Calculate McDonald's Omega
calculate_omega <- function(data, items) {
  if (length(items) < 2) return(NA)
  tryCatch({
    # Use psych::omega for McDonald's omega
    omega_result <- psych::omega(data[items])
    return(omega_result$omega.total)
  }, error = function(e) {
    return(NA)
  })
}

# Calculate Cronbach's Alpha
calculate_alpha <- function(data, items) {
  if (length(items) < 2) return(NA)
  tryCatch({
    alpha_result <- psych::alpha(data[items])
    return(alpha_result$total$raw_alpha)
  }, error = function(e) {
    return(NA)
  })
}

# Perform CFA for a single domain
analyze_domain_cfa <- function(data, domain, scale_range = c(1, 9)) {
  items <- domain_items(domain)
  
  # Check if all items exist
  available_items <- items[items %in% names(data)]
  if (length(available_items) < 3) {
    return(list(error = paste("Insufficient items for", domain, "- only", length(available_items), "available")))
  }
  
  # Get negative items for this domain
  negative_items <- BIG_FIVE_DOMAINS[[domain]]$negative
  negative_items <- negative_items[negative_items %in% available_items]
  
  # Apply reverse coding
  data_processed <- reverse_code(data, negative_items, scale_range[1], scale_range[2])
  
  # Prepare data for CFA
  domain_data <- data_processed[available_items]
  
  # Remove rows with any missing values
  domain_data <- na.omit(domain_data)
  
  if (nrow(domain_data) < 10) {
    return(list(error = paste("Insufficient participants for", domain)))
  }
  
  # Create CFA model specification
  latent_var <- domain
  model_spec <- paste(latent_var, "=~", paste(available_items, collapse = " + "))
  
  tryCatch({
    # Fit CFA model
    fit <- cfa(model_spec, data = domain_data, std.lv = TRUE)
    
    # Get fit indices
    fit_measures <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
    
    # Get standardized loadings
    loadings <- standardizedSolution(fit)
    loadings <- loadings[loadings$op == "=~", c("rhs", "est.std")]
    colnames(loadings) <- c("item", "loading")
    
    # Calculate reliability
    omega <- calculate_omega(domain_data, available_items)
    alpha <- calculate_alpha(domain_data, available_items)
    
    # Eigenvalue and variance explained (from factor loadings)
    loadings_vec <- loadings$loading
    eigenvalue <- sum(loadings_vec^2)
    variance_explained <- eigenvalue / length(available_items)
    
    list(
      domain = domain,
      n_items = length(available_items),
      n_participants = nrow(domain_data),
      alpha = alpha,
      omega = omega,
      eigenvalue = eigenvalue,
      variance_explained = variance_explained,
      chisq = fit_measures["chisq"],
      df = fit_measures["df"],
      pvalue = fit_measures["pvalue"],
      cfi = fit_measures["cfi"],
      tli = fit_measures["tli"],
      rmsea = fit_measures["rmsea"],
      srmr = fit_measures["srmr"],
      loadings = loadings,
      converged = TRUE,
      items_used = available_items
    )
  }, error = function(e) {
    list(error = as.character(e))
  })
}

# Main analysis function
analyze_simulation_file <- function(json_path, output_dir, scale_range = c(1, 9)) {
  
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Load simulation data
  tryCatch({
    data <- fromJSON(json_path)
    data <- as.data.frame(data)
    
    # Get model name from filename
    model_name <- tools::file_path_sans_ext(basename(json_path))
    
    # Get study and format info from path
    path_parts <- strsplit(json_path, "/")[[1]]
    study_info <- path_parts[grep("study_", path_parts)]
    format_info <- path_parts[grep("_results", path_parts)]
    
    study <- ifelse(length(study_info) > 0, study_info[1], "unknown")
    format <- ifelse(length(format_info) > 0, 
                     gsub("_results.*", "", format_info[1]), 
                     "unknown")
    
    # Get all negative items for reverse coding
    all_negative_items <- get_negative_items()
    negative_in_data <- all_negative_items[all_negative_items %in% names(data)]
    
    if (length(negative_in_data) > 0) {
      data <- reverse_code(data, negative_in_data, scale_range[1], scale_range[2])
    }
    
    # Analyze each domain
    results <- list()
    
    for (domain in names(BIG_FIVE_DOMAINS)) {
      cat("Analyzing", domain, "for", model_name, "\n")
      domain_result <- analyze_domain_cfa(data, domain, scale_range)
      results[[domain]] <- domain_result
    }
    
    # Create summary dataframe
    summary_data <- map_df(names(results), function(domain) {
      res <- results[[domain]]
      if ("error" %in% names(res)) {
        data.frame(
          Study = toupper(study),
          Format = format,
          Model = model_name,
          Structure_Type = "Original",
          Factor_Domain = domain,
          N_Items = NA,
          N_Participants = NA,
          Alpha = NA,
          Omega = NA,
          Eigenvalue = NA,
          Variance_Explained = NA,
          Mean_Loading_Abs = NA,
          Max_Loading_Abs = NA,
          Min_Loading_Abs = NA,
          RMSEA = NA,
          CFI = NA,
          TLI = NA,
          SRMR = NA,
          Total_Variance_Explained = NA,
          N_Factors_Total = 5,
          Error = res$error
        )
      } else {
        data.frame(
          Study = toupper(study),
          Format = format,
          Model = model_name,
          Structure_Type = "Original",
          Factor_Domain = domain,
          N_Items = res$n_items,
          N_Participants = res$n_participants,
          Alpha = res$alpha,
          Omega = res$omega,
          Eigenvalue = res$eigenvalue,
          Variance_Explained = res$variance_explained,
          Mean_Loading_Abs = mean(abs(res$loadings$loading)),
          Max_Loading_Abs = max(abs(res$loadings$loading)),
          Min_Loading_Abs = min(abs(res$loadings$loading)),
          RMSEA = res$rmsea,
          CFI = res$cfi,
          TLI = res$tli,
          SRMR = res$srmr,
          Total_Variance_Explained = NA,
          N_Factors_Total = 5,
          Error = NA
        )
      }
    })
    
    # Save results
    summary_file <- file.path(output_dir, paste0(model_name, "_factor_summary.csv"))
    write.csv(summary_data, summary_file, row.names = FALSE)
    
    # Save detailed results
    detailed_file <- file.path(output_dir, paste0(model_name, "_detailed_results.json"))
    write_json(results, detailed_file, auto_unbox = TRUE, pretty = TRUE)
    
    cat("Analysis complete for", model_name, "\n")
    cat("Results saved to:", summary_file, "\n")
    
    return(summary_data)
    
  }, error = function(e) {
    cat("Error processing", json_path, ":", as.character(e), "\n")
    return(NULL)
  })
}

# Command line interface
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 2) {
    cat("Usage: Rscript cfa_analysis.R <input_json> <output_dir> [scale_min] [scale_max]\n")
    cat("Example: Rscript cfa_analysis.R results.json output_dir 1 9\n")
    quit(status = 1)
  }
  
  json_path <- args[1]
  output_dir <- args[2]
  scale_min <- ifelse(length(args) >= 3, as.numeric(args[3]), 1)
  scale_max <- ifelse(length(args) >= 4, as.numeric(args[4]), 9)
  
  result <- analyze_simulation_file(json_path, output_dir, c(scale_min, scale_max))
  
  if (!is.null(result)) {
    cat("\nAnalysis Summary:\n")
    print(result[, c("Factor_Domain", "N_Items", "N_Participants", "Alpha", "Omega", "CFI", "RMSEA")])
  }
}