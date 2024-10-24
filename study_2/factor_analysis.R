library(psych)
library(lavaan)
library(semTools)
library(readr)

# Read the CSV file
data <- read_csv("~/psychometrics_AI/study_1/likert_format/gpt3-5-turbo/study1_with_simulation_result.csv")

# Function to reverse code items
reverse_code <- function(x) {
  return(10 - x)
}

# Prepare data for Set 1 (original)
set1_items <- paste0("tda", 1:40)
set1_data <- data[, set1_items]

# Reverse code necessary items for Set 1
#reverse_items <- c(1, 3, 4, 9, 15, 17, 21, 25, 26, 27, 28, 29, 35, 37, 38, 40)
reverse_items <- c(1, 3, 4, 9, 15, 17, 25, 26, 27, 28, 29, 35, 36, 37, 38, 40)
for (item in reverse_items) {
  set1_data[, paste0("tda", item)] <- reverse_code(set1_data[, paste0("tda", item)])
}

# Prepare data for Set 2 (simulated)
set2_items <- c("Bashful", "Bold", "Careless", "Cold", "Complex", "Cooperative", "Creative", "Deep",
                "Disorganized", "Efficient", "Energetic", "Envious", "Extraverted", "Fretful", "Harsh",
                "Imaginative", "Inefficient", "Intellectual", "Jealous", "Kind", "Moody", "Organized",
                "Philosophical", "Practical", "Quiet", "Relaxed", "Rude", "Shy", "Sloppy", "Sympathetic",
                "Systematic", "Talkative", "Temperamental", "Touchy", "Uncreative", "Unenvious",
                "Unintellectual", "Unsympathetic", "Warm", "Withdrawn")
set2_data <- data[, set2_items]

reverse_items2 <- c("Bashful", "Cold", "Careless", "Disorganized", "Harsh", "Inefficient", "Quiet",
                    "Relaxed", "Rude", "Shy", "Sloppy", "Uncreative",
                    "Unenvious", "Unintellectual", "Unsympathetic", "Withdrawn")
for (item in reverse_items2) {
  set2_data[, item] <- reverse_code(set2_data[, item])
}

# Check for near-zero variance in both sets
check_and_remove_near_zero_var <- function(data) {
  near_zero_var <- which(apply(data, 2, var) < 0.01)
  if (length(near_zero_var) > 0) {
    print("Variables with near-zero variance:")
    print(names(data)[near_zero_var])
    data <- data[, -near_zero_var]
  }
  return(data)
}

set1_data <- check_and_remove_near_zero_var(set1_data)
set2_data <- check_and_remove_near_zero_var(set2_data)

# Define items for each domain in Set 1 (original data)
set1_domains <- list(
  Extraversion = c("tda2", "tda11", "tda13", "tda32", "tda1", "tda25", "tda28", "tda40"),
  Agreeableness = c("tda6", "tda20", "tda30", "tda39", "tda4", "tda15", "tda27", "tda38"),
  Conscientiousness = c("tda10", "tda22", "tda24", "tda31", "tda3", "tda9", "tda17", "tda29"),
  Neuroticism = c("tda12", "tda14", "tda19", "tda21", "tda33", "tda34", "tda26", "tda36"),
  Openness = c("tda5", "tda7", "tda8", "tda16", "tda18", "tda23", "tda35", "tda37")
)

# Define items for each domain in Set 2 (simulated data)
set2_domains <- list(
  Extraversion = c("Bold", "Energetic", "Extraverted", "Talkative", "Bashful", "Quiet", "Shy", "Withdrawn"),
  Agreeableness = c("Cooperative", "Kind", "Sympathetic", "Warm", "Cold", "Harsh", "Rude", "Unsympathetic"),
  Conscientiousness = c("Efficient", "Organized", "Practical", "Systematic", "Careless", "Disorganized", "Inefficient", "Sloppy"),
  Neuroticism = c("Envious", "Fretful", "Jealous", "Moody", "Temperamental", "Touchy", "Relaxed", "Unenvious"),
  Openness = c("Complex", "Deep", "Creative", "Imaginative", "Intellectual", "Philosophical", "Uncreative", "Unintellectual")
)

# Function to create a CFA model for a single domain
create_domain_model <- function(domain, items) {
  paste(domain, "=~", paste(items, collapse = " + "))
}

# Function to fit CFA model, compute fit measures, and calculate reliability for a single domain
fit_single_domain_model <- function(domain, items, data) {
  tryCatch({
    model <- create_domain_model(domain, items)
    fit <- cfa(model, data = data, estimator = "MLR")
    
    # Check if the model converged
    if (!lavInspect(fit, "converged")) {
      return(list(error = "Model did not converge"))
    }
    
    # Compute fit measures
    fit_measures <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
    
    # Get standardized factor loadings
    loadings <- standardizedSolution(fit)
    loadings <- loadings[loadings$op == "=~", c("rhs", "est.std")]
    
    # Calculate Cronbach's alpha
    reliability <- psych::alpha(data[, items])$total$std.alpha
    
    return(list(
      fit_measures = fit_measures,
      factor_loadings = loadings,
      reliability = reliability
    ))
  }, error = function(e) {
    return(list(error = paste("Error in fitting model:", e$message)))
  })
}

# Fit models, compute fit measures, and calculate reliability for each domain in Set 1
set1_results <- list()
for (domain in names(set1_domains)) {
  cat(paste("\nProcessing domain:", domain, "\n"))
  result <- fit_single_domain_model(domain, set1_domains[[domain]], set1_data)
  set1_results[[domain]] <- result
  
  if ("error" %in% names(result)) {
    cat(paste("Error:", result$error, "\n"))
  } else {
    cat("Fit measures:\n")
    print(result$fit_measures)
    cat("\nStandardized factor loadings:\n")
    print(result$factor_loadings)
    cat("\nReliability (Cronbach's alpha):\n")
    print(result$reliability)
  }
  cat("\n")
}

# Fit models, compute fit measures, and calculate reliability for each domain in Set 2
set2_results <- list()
for (domain in names(set2_domains)) {
  cat(paste("\nProcessing domain:", domain, "\n"))
  result <- fit_single_domain_model(domain, set2_domains[[domain]], set2_data)
  set2_results[[domain]] <- result
  
  if ("error" %in% names(result)) {
    cat(paste("Error:", result$error, "\n"))
  } else {
    cat("Fit measures:\n")
    print(result$fit_measures)
    cat("\nStandardized factor loadings:\n")
    print(result$factor_loadings)
    cat("\nReliability (Cronbach's alpha):\n")
    print(result$reliability)
  }
  cat("\n")
}

# Summary of results
cat("\nSummary of results for Set 1:\n")
for (domain in names(set1_results)) {
  if ("error" %in% names(set1_results[[domain]])) {
    cat(paste(domain, ": Failed to converge or error in fitting\n"))
  } else {
    cat(paste(domain, ": Successfully fitted, Reliability =", round(set1_results[[domain]]$reliability, 3), "\n"))
  }
}

cat("\nSummary of results for Set 2:\n")
for (domain in names(set2_results)) {
  if ("error" %in% names(set2_results[[domain]])) {
    cat(paste(domain, ": Failed to converge or error in fitting\n"))
  } else {
    cat(paste(domain, ": Successfully fitted, Reliability =", round(set2_results[[domain]]$reliability, 3), "\n"))
  }
}

# Define the revised Neuroticism items
neuroticism_items_revised <- c("Jealous", "Fretful", "Moody", "Temperamental", "Touchy", "Relaxed", "Unenvious")

# Ensure neuroticism_data_revised contains only these items
neuroticism_data_revised <- set2_data[, neuroticism_items_revised]

# Apply the fit_single_domain_model function to the revised Neuroticism domain
neuroticism_results <- fit_single_domain_model("Neuroticism", neuroticism_items_revised, neuroticism_data_revised)

# Display the results
if ("error" %in% names(neuroticism_results)) {
  cat("Error:", neuroticism_results$error, "\n")
} else {
  cat("Fit measures:\n")
  print(neuroticism_results$fit_measures)
  
  cat("\nStandardized factor loadings:\n")
  print(neuroticism_results$factor_loadings)
  
  cat("\nReliability (Cronbach's alpha):\n")
  print(neuroticism_results$reliability)
}


### Expanded Scale
###

library(psych)
library(lavaan)
library(semTools)
library(readr)

# Read the CSV file
data <- read_csv("~/psychometrics_AI/study_1/expanded_format/gpt3-5_turbo/study1_with_simulation_result.csv")

# Function to reverse code items
reverse_code <- function(x) {
  return(10 - x)
}

# Prepare data for Set 1 (original)
set1_items <- paste0("tda", 1:40)
set1_data <- data[, set1_items]

# Reverse code necessary items for Set 1
reverse_items <- c(1, 3, 4, 9, 15, 17, 21, 25, 26, 27, 28, 29, 35, 37, 38, 40)
for (item in reverse_items) {
  set1_data[, paste0("tda", item)] <- reverse_code(set1_data[, paste0("tda", item)])
}

# Prepare data for Set 2 (simulated)
set2_items <- c("Bashful", "Bold", "Careless", "Cold", "Complex", "Cooperative", "Creative", "Deep",
                "Disorganized", "Efficient", "Energetic", "Envious", "Extraverted", "Fretful", "Harsh",
                "Imaginative", "Inefficient", "Intellectual", "Jealous", "Kind", "Moody", "Organized",
                "Philosophical", "Practical", "Quiet", "Relaxed", "Rude", "Shy", "Sloppy", "Sympathetic",
                "Systematic", "Talkative", "Temperamental", "Touchy", "Uncreative", "Unenvious",
                "Unintellectual", "Unsympathetic", "Warm", "Withdrawn")
set2_data <- data[, set2_items]

reverse_items2 <- c("Bashful", "Cold", "Careless", "Disorganized", "Harsh", "Inefficient", "Quiet",
                    "Relaxed", "Rude", "Shy", "Sloppy", "Uncreative",
                    "Unenvious", "Unintellectual", "Unsympathetic", "Withdrawn")
for (item in reverse_items2) {
  set2_data[, item] <- reverse_code(set2_data[, item])
}

# Check for near-zero variance in both sets
check_and_remove_near_zero_var <- function(data) {
  near_zero_var <- which(apply(data, 2, var) < 0.01)
  if (length(near_zero_var) > 0) {
    print("Variables with near-zero variance:")
    print(names(data)[near_zero_var])
    data <- data[, -near_zero_var]
  }
  return(data)
}

set1_data <- check_and_remove_near_zero_var(set1_data)
set2_data <- check_and_remove_near_zero_var(set2_data)

# Define items for each domain in Set 1 (original data)
set1_domains <- list(
  Extraversion = c("tda2", "tda11", "tda13", "tda32", "tda1", "tda25", "tda28", "tda40"),
  Agreeableness = c("tda6", "tda20", "tda30", "tda39", "tda4", "tda15", "tda27", "tda38"),
  Conscientiousness = c("tda10", "tda22", "tda24", "tda31", "tda3", "tda9", "tda17", "tda29"),
  Neuroticism = c("tda12", "tda14", "tda19", "tda21", "tda33", "tda34", "tda26", "tda36"),
  Openness = c("tda5", "tda7", "tda8", "tda16", "tda18", "tda23", "tda35", "tda37")
)

# Define items for each domain in Set 2 (simulated data)
set2_domains <- list(
  Extraversion = c("Bold", "Energetic", "Extraverted", "Talkative", "Bashful", "Quiet", "Shy", "Withdrawn"),
  Agreeableness = c("Cooperative", "Kind", "Sympathetic", "Warm", "Cold", "Harsh", "Rude", "Unsympathetic"),
  Conscientiousness = c("Efficient", "Organized", "Practical", "Systematic", "Careless", "Disorganized", "Inefficient", "Sloppy"),
  Neuroticism = c("Envious", "Fretful", "Jealous", "Moody", "Temperamental", "Touchy", "Relaxed", "Unenvious"),
  Openness = c("Complex", "Deep", "Creative", "Imaginative", "Intellectual", "Philosophical", "Uncreative", "Unintellectual")
)

# Function to create a CFA model for a single domain
create_domain_model <- function(domain, items) {
  paste(domain, "=~", paste(items, collapse = " + "))
}

# Function to fit CFA model, compute fit measures, and calculate reliability for a single domain
fit_single_domain_model <- function(domain, items, data) {
  tryCatch({
    model <- create_domain_model(domain, items)
    fit <- cfa(model, data = data, estimator = "MLR")
    
    # Check if the model converged
    if (!lavInspect(fit, "converged")) {
      return(list(error = "Model did not converge"))
    }
    
    # Compute fit measures
    fit_measures <- fitMeasures(fit, c("chisq", "df", "pvalue", "cfi", "tli", "rmsea", "srmr"))
    
    # Get standardized factor loadings
    loadings <- standardizedSolution(fit)
    loadings <- loadings[loadings$op == "=~", c("rhs", "est.std")]
    
    # Calculate Cronbach's alpha
    reliability <- psych::alpha(data[, items])$total$std.alpha
    
    return(list(
      fit_measures = fit_measures,
      factor_loadings = loadings,
      reliability = reliability
    ))
  }, error = function(e) {
    return(list(error = paste("Error in fitting model:", e$message)))
  })
}

# Fit models, compute fit measures, and calculate reliability for each domain in Set 1
set1_results <- list()
for (domain in names(set1_domains)) {
  cat(paste("\nProcessing domain:", domain, "\n"))
  result <- fit_single_domain_model(domain, set1_domains[[domain]], set1_data)
  set1_results[[domain]] <- result
  
  if ("error" %in% names(result)) {
    cat(paste("Error:", result$error, "\n"))
  } else {
    cat("Fit measures:\n")
    print(result$fit_measures)
    cat("\nStandardized factor loadings:\n")
    print(result$factor_loadings)
    cat("\nReliability (Cronbach's alpha):\n")
    print(result$reliability)
  }
  cat("\n")
}

# Fit models, compute fit measures, and calculate reliability for each domain in Set 2
set2_results <- list()
for (domain in names(set2_domains)) {
  cat(paste("\nProcessing domain:", domain, "\n"))
  result <- fit_single_domain_model(domain, set2_domains[[domain]], set2_data)
  set2_results[[domain]] <- result
  
  if ("error" %in% names(result)) {
    cat(paste("Error:", result$error, "\n"))
  } else {
    cat("Fit measures:\n")
    print(result$fit_measures)
    cat("\nStandardized factor loadings:\n")
    print(result$factor_loadings)
    cat("\nReliability (Cronbach's alpha):\n")
    print(result$reliability)
  }
  cat("\n")
}

# Summary of results
cat("\nSummary of results for Set 1:\n")
for (domain in names(set1_results)) {
  if ("error" %in% names(set1_results[[domain]])) {
    cat(paste(domain, ": Failed to converge or error in fitting\n"))
  } else {
    cat(paste(domain, ": Successfully fitted, Reliability =", round(set1_results[[domain]]$reliability, 3), "\n"))
  }
}

cat("\nSummary of results for Set 2:\n")
for (domain in names(set2_results)) {
  if ("error" %in% names(set2_results[[domain]])) {
    cat(paste(domain, ": Failed to converge or error in fitting\n"))
  } else {
    cat(paste(domain, ": Successfully fitted, Reliability =", round(set2_results[[domain]]$reliability, 3), "\n"))
  }
}

## Looking into Neuroticism
# Define the revised Neuroticism items
neuroticism_items_revised <- c("Jealous", "Fretful", "Moody", "Temperamental", "Touchy", "Relaxed", "Unenvious")

# Ensure neuroticism_data_revised contains only these items
neuroticism_data_revised <- set2_data[, neuroticism_items_revised]

# Apply the fit_single_domain_model function to the revised Neuroticism domain
neuroticism_results <- fit_single_domain_model("Neuroticism", neuroticism_items_revised, neuroticism_data_revised)

# Display the results
if ("error" %in% names(neuroticism_results)) {
  cat("Error:", neuroticism_results$error, "\n")
} else {
  cat("Fit measures:\n")
  print(neuroticism_results$fit_measures)
  
  cat("\nStandardized factor loadings:\n")
  print(neuroticism_results$factor_loadings)
  
  cat("\nReliability (Cronbach's alpha):\n")
  print(neuroticism_results$reliability)
}
