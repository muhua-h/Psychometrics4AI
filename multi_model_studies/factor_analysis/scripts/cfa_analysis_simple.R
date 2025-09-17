#!/usr/bin/env Rscript

# Simple R-based Confirmatory Factor Analysis
# Focuses on core CFA with proper fit indices

suppressPackageStartupMessages({
  library(lavaan)
  library(psych)
  library(jsonlite)
  library(dplyr)
  library(semTools)
})

# Big Five domains
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

# Reverse code items - Fixed to always use 1-9 scale
reverse_items <- function(data) {
  negative <- c("Bashful", "Quiet", "Shy", "Withdrawn", "Cold", "Harsh", "Rude", "Unsympathetic",
                "Careless", "Disorganized", "Inefficient", "Sloppy", "Relaxed", "Unenvious",
                "Uncreative", "Unintellectual")

  to_reverse <- negative[negative %in% names(data)]
  if (length(to_reverse) > 0) {
    # Always use 1-9 scale reverse coding (10 - value)
    data[to_reverse] <- 10 - data[to_reverse]
  }
  data
}

# Main analysis function
analyze_file <- function(json_path, output_dir) {

  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  tryCatch({
    # Load data
    cat("Loading data from:", json_path, "\n")
    data <- fromJSON(json_path)
    data <- as.data.frame(data)

    # Reverse code using fixed 1-9 scale
    data <- reverse_items(data)

    # Clean data
    data <- na.omit(data)

    # Extract model name
    model_name <- tools::file_path_sans_ext(basename(json_path))

    # Determine study and format from path
    path_parts <- strsplit(json_path, "/")[[1]]
    study <- ifelse(any(grepl("study_2a", path_parts)), "STUDY_2",
               ifelse(any(grepl("study_2b", path_parts)), "STUDY_3", "STUDY_4"))

    format <- ifelse(any(grepl("binary_elaborated", path_parts)), "binary_elaborated",
                ifelse(any(grepl("binary_simple", path_parts)), "binary_simple",
                ifelse(any(grepl("expanded", path_parts)), "expanded", "likert")))

    results <- list()

    for (domain in names(BIG_FIVE_DOMAINS)) {
      cat("Processing", domain, "...")

      # Get items for domain
      items <- c(BIG_FIVE_DOMAINS[[domain]]$positive, BIG_FIVE_DOMAINS[[domain]]$negative)
      available <- items[items %in% names(data)]

      if (length(available) < 3) {
        cat("⌘ insufficient items (", length(available), ")\n")
        next
      }

      # Subset data
      domain_data <- data[available]

      # CFA model
      model_spec <- paste(domain, "=~", paste(available, collapse = " + "))

      tryCatch({
        # Fit CFA
        fit <- cfa(model_spec, data = domain_data, std.lv = TRUE, estimator = "ML", std.lv = TRUE)

        # Fit indices
        fit_stats <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))

        # Loadings
        loadings <- standardizedSolution(fit)
        loadings <- loadings[loadings$op == "=~", c("rhs", "est.std")]

        # Reliability
        alpha <- psych::alpha(domain_data, check.keys = TRUE)$total$raw_alpha

        # McDonald's Omega using multiple methods
        omega <- tryCatch({
          # Primary method: compRelSEM
          compRelSEM(fit, return.total = TRUE)
        }, error = function(e1) {
          tryCatch({
            # Fallback method: psych::omega
            omega_res <- psych::omega(domain_data)
            omega_res$omega.tot[1]
          }, error = function(e2) {
            # Final fallback: Cronbach's alpha approximation
            cat("Warning: Both compRelSEM and psych::omega failed, using NA\n")
            NA
          })
        })

        # Summary
        results[[domain]] <- data.frame(
          Study = study,
          Format = format,
          Model = model_name,
          Factor_Domain = domain,
          N_Items = length(available),
          N_Participants = nrow(domain_data),
          Alpha = round(alpha, 3),
          Omega = round(omega, 3),
          CFI = round(fit_stats["cfi"], 3),
          RMSEA = round(fit_stats["rmsea"], 3),
          SRMR = round(fit_stats["srmr"], 3),
          Chi_Square = round(fit_stats["chisq"], 3),
          DF = fit_stats["df"],
          P_Value = round(fit_stats["pvalue"], 3)
        )

        cat("✅ Alpha:", round(alpha, 3), "Omega:", round(omega, 3), "CFI:", round(fit_stats["cfi"], 3), "\n")

      }, error = function(e) {
        cat("⌘ CFA failed:", substr(as.character(e), 1, 50), "...\n")
      })
    }
    
    if (length(results) > 0) {
      final_results <- bind_rows(results)
      
      # Save results
      summary_file <- file.path(output_dir, paste0(model_name, "_R_factor_analysis.csv"))
      write.csv(final_results, summary_file, row.names = FALSE)
      
      cat("Results saved to:", summary_file, "\n")
      print(final_results)
    }
    
  }, error = function(e) {
    cat("Error processing file:", as.character(e), "\n")
  })
}

# Command line usage
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  
  if (length(args) < 2) {
    cat("Usage: Rscript cfa_analysis_simple.R <json_path> <output_dir>\n")
    quit(status = 1)
  }
  
  json_path <- args[1]
  output_dir <- args[2]
  
  analyze_file(json_path, output_dir)
}