#!/usr/bin/env Rscript

# Create cross-format comparison for R-based CFA results
# Similar to existing Python cross-format comparison

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
})

# Function to collect all R results
collect_all_r_results <- function() {
  base_dir <- "/Users/mhhuang/Psychometrics4AI_revision/multi_model_studies/factor_analysis/results_r"
  
  all_files <- list()
  
  # Process each study
  for (study in c("study_2", "study_3", "study_4")) {
    study_path <- file.path(base_dir, study)
    
    if (dir.exists(study_path)) {
      # Find all CSV files in format directories
      pattern <- file.path(study_path, "*format", "*_R_factor_analysis.csv")
      files <- Sys.glob(pattern)
      
      for (file_path in files) {
        tryCatch({
          df <- read_csv(file_path, show_col_types = FALSE)
          
          # Extract format from directory name
          format_dir <- basename(dirname(file_path))
          format <- gsub("_format$", "", format_dir)
          
          # Ensure consistent study naming
          df$Study <- toupper(study)
          df$Format <- format
          
          all_files[[length(all_files) + 1]] <- df
        }, error = function(e) {
          cat("Error processing", file_path, ":", e$message, "\n")
        })
      }
    }
  }
  
  if (length(all_files) > 0) {
    bind_rows(all_files)
  } else {
    data.frame()
  }
}

# Function to create cross-format summary at domain level
create_cross_format_summary <- function(df) {
  if (nrow(df) == 0) {
    return(data.frame())
  }
  
  # Keep domain-level information (no aggregation)
  summary <- df %>%
    select(Study, Format, Model, Factor_Domain, N_Items, N_Participants, Alpha, Omega, CFI, TLI, RMSEA, SRMR, Chi_Square, DF, P_Value) %>%
    arrange(Study, Format, Model, Factor_Domain)
  
  summary
}

# Main execution
main <- function() {
  cat("Creating R-based cross-format comparison...\n")
  
  # Collect all results
  df <- collect_all_r_results()
  
  if (nrow(df) == 0) {
    cat("No R results found!\n")
    return()
  }
  
  cat(sprintf("Found %d total records across all studies and formats\n", nrow(df)))
  
  # Create cross-format summary
  summary <- create_cross_format_summary(df)
  
  # Create output directory
  output_dir <- "/Users/mhhuang/Psychometrics4AI_revision/multi_model_studies/factor_analysis/results_r"
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Create cross_format_comparison directory
  cross_format_dir <- file.path(output_dir, "cross_format_comparison")
  if (!dir.exists(cross_format_dir)) {
    dir.create(cross_format_dir, recursive = TRUE)
  }
  
  # Save all individual records
  write_csv(df, file.path(output_dir, "all_r_results.csv"))
  
  # Save cross-format summary
  write_csv(summary, file.path(cross_format_dir, "R_factor_analysis_summary.csv"))
  
  # Create study-specific summaries
  for (study in unique(summary$Study)) {
    study_df <- summary %>% filter(Study == study)
    study_dir <- file.path(output_dir, tolower(study), "cross_format_analysis")
    if (!dir.exists(study_dir)) {
      dir.create(study_dir, recursive = TRUE)
    }
    write_csv(study_df, file.path(study_dir, sprintf("%s_cross_format_R_summary.csv", tolower(study))))
  }
  
  cat("Cross-format comparison created!\n")
  cat(sprintf("Total records: %d\n", nrow(df)))
  cat(sprintf("Summary records: %d\n", nrow(summary)))
  cat(sprintf("Studies: %s\n", paste(unique(df$Study), collapse = ", ")))
  cat(sprintf("Formats: %s\n", paste(unique(df$Format), collapse = ", ")))
  
  # Print sample of results
  cat("\n=== Sample Cross-Format Summary ===\n")
  print(head(summary, 10))
}

# Execute if run as script
if (!interactive()) {
  main()
}