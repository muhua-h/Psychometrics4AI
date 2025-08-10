#!/usr/bin/env Rscript

# Corrected R-based Confirmatory Factor Analysis
# Uses lavaan package properly for CFA fit indices

suppressPackageStartupMessages({
  library(lavaan)
  library(psych)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
  library(purrr)
})

# Big Five domains with Mini-Marker items
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

# Function to reverse code negative items
reverse_code_items <- function(data, scale_range = c(1, 9)) {
  negative_items <- c("Bashful", "Quiet", "Shy", "Withdrawn",
                       "Cold", "Harsh", "Rude", "Unsympathetic",
                       "Careless", "Disorganized", "Inefficient", "Sloppy",
                       "Relaxed", "Unenvious",
                       "Uncreative", "Unintellectual")
  
  negative_available <- negative_items[negative_items %in% names(data)]
  if (length(negative_available) > 0) {
    data[negative_available] <- scale_range[2] + scale_range[1] - data[negative_available]
  }
  return(data)
}

# Function to get domain items
domain_items <- function(domain) {
  items <- BIG_FIVE_DOMAINS[[domain]]
  return(c(items$positive, items$negative))
}

# Main analysis function
analyze_file <- function(json_path, output_dir) {
  
  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Load data
  cat("Loading:", json_path, "\n")
  json_data <- fromJSON(json_path)
  
  # Convert list of objects to data frame
  if (is.list(json_data) && length(json_data) > 0) {
    data <- as.data.frame(json_data)
  } else {
    data <- as.data.frame(json_data)
  }
  
  # Reverse code negative items
  data <- reverse_code_items(data, c(1, 9))
  
  # Get model name from filename
  model_name <- tools::file_path_sans_ext(basename(json_path))
  model_name <- gsub("bfi_to_minimarker_|_temp.*", "", model_name)
  model_name <- gsub("openai_gpt_3.5_turbo_0125", "gpt_3.5_turbo", model_name)
  
  # Determine study and format from path
  path_parts <- strsplit(json_path, "/")[[1]]
  study <- ifelse("study_2" %in% path_parts, "STUDY_2", "STUDY_3")
  format <- ifelse("binary" %in% path_parts, 
                   ifelse("elaborated" %in% path_parts, "binary_elaborated", "binary_simple"),
                   ifelse("expanded" %in% path_parts, "expanded", "likert"))
  
  # Process each domain
  results <- list()
  
  for (domain in names(BIG_FIVE_DOMAINS)) {
    cat("  Analyzing", domain, "\n")
    
    items <- domain_items(domain)
    available_items <- items[items %in% names(data)]
    
    if (length(available_items) < 3) {
      cat("    Warning: Insufficient items for", domain, "-", length(available_items), "available\n")
      next
    }
    
    # Clean data for this domain
    domain_data <- data[available_items]
    domain_data <- na.omit(domain_data)
    
    if (nrow(domain_data) < 5) {
      cat("    Warning: Insufficient participants for", domain, "-", nrow(domain_data), "rows\n")
      next
    }
    
    # Create CFA model
    model_spec <- paste(domain, "=~", paste(available_items, collapse = " + "))
    
    tryCatch({
      # Fit CFA model
      fit <- cfa(model_spec, data = domain_data, std.lv = TRUE, se = "standard")
      
      # Get fit measures
      fit_stats <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
      
      # Get standardized loadings
      loadings <- standardizedSolution(fit)
      factor_loadings <- loadings[loadings$op == "=~", c("rhs", "est.std")]
      
      # Calculate reliabilities
      alpha <- psych::alpha(domain_data)$total$raw_alpha
      omega <- psych::omega(domain_data)$omega.total
      
      # Calculate eigenvalue and variance explained
      loadings_vec <- factor_loadings$est.std
      eigenvalue <- sum(loadings_vec^2)
      variance_explained <- eigenvalue / length(available_items)
      
      result <- data.frame(
        Study = study,
        Format = format,
        Model = model_name,
        Structure_Type = "Original",
        Factor_Domain = domain,
        N_Items = length(available_items),
        N_Participants = nrow(domain_data),
        Alpha = alpha,
        Omega = omega,
        Eigenvalue = eigenvalue,
        Variance_Explained = variance_explained,
        Mean_Loading_Abs = mean(abs(loadings_vec)),
        Max_Loading_Abs = max(abs(loadings_vec)),
        Min_Loading_Abs = min(abs(loadings_vec)),
        RMSEA = fit_stats["rmsea"],
        CFI = fit_stats["cfi"],
        TLI = fit_stats["tli"],
        SRMR = fit_stats["srmr"],
        Total_Variance_Explained = NA,
        N_Factors_Total = 5,
        File_Source = basename(json_path)
      )
      
      results[[domain]] <- result
      cat("    ✅", domain, "- Omega:", round(omega, 3), "RMSEA:", round(fit_stats["rmsea"], 3), "CFI:", round(fit_stats["cfi"], 3), "\n")
      
    }, error = function(e) {
      cat("    ❌", domain, "- Error:", as.character(e), "\n")
      
      # Fallback with basic stats
      result <- data.frame(
        Study = study,
        Format = format,
        Model = model_name,
        Structure_Type = "Original",
        Factor_Domain = domain,
        N_Items = length(available_items),
        N_Participants = nrow(domain_data),
        Alpha = psych::alpha(domain_data)$total$raw_alpha,
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
        File_Source = basename(json_path)
      )
      results[[domain]] <- result
    })
  }
  
  if (length(results) > 0) {
    # Combine all domain results
    combined_results <- bind_rows(results)
    
    # Save results
    summary_file <- file.path(output_dir, paste0(model_name, "_factor_summary.csv"))
    write.csv(combined_results, summary_file, row.names = FALSE)
    
    cat("Results saved to:", summary_file, "\n")
    cat("Summary:", nrow(combined_results), "domains analyzed\n")
    
    return(combined_results)
  }
  
  return(NULL)
}

# Command line usage
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 2) {
    cat("Usage: Rscript cfa_analysis_corrected.R <input_json> <output_dir>\n")
    quit(status = 1)
  }
  
  json_path <- args[1]
  output_dir <- args[2]
  
  result <- analyze_file(json_path, output_dir)
  
  if (!is.null(result)) {
    cat("\\nAnalysis Complete\\n")
    cat("Domains analyzed:", nrow(result), "\\n")
    cat("Average Omega:", round(mean(result$Omega, na.rm = TRUE), 3), "\\n")
    cat("Average RMSEA:", round(mean(result$RMSEA, na.rm = TRUE), 3), "\\n")
    cat("Average CFI:", round(mean(result$CFI, na.rm = TRUE), 3), "\\n")
  }
}