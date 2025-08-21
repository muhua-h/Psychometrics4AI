#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lavaan)
  library(jsonlite)
  library(dplyr)
  library(readr)
  library(purrr)
  library(stringr)
  library(tibble)
})

# ---------- CLI args ----------
args <- commandArgs(trailingOnly = TRUE)
read_arg <- function(flag, default = NULL) {
  ix <- which(args == flag)
  if (length(ix) == 0 || ix == length(args)) return(default)
  args[ix + 1]
}
INPUT_DIR   <- read_arg("--input_dir",   ".")
OUT_CSV     <- read_arg("--out_csv",     file.path(".", "factor_loadings_summary.csv"))
LOAD_DIR    <- read_arg("--loadings_dir",file.path(".", "loadings"))
if (!dir.exists(LOAD_DIR)) dir.create(LOAD_DIR, recursive = TRUE, showWarnings = FALSE)

# ---------- Mini-Marker items (edit to match your column names) ----------
DOMAINS <- list(
  Extraversion = c("Bold", "Energetic", "Extraverted", "Talkative",
                   "Bashful", "Quiet", "Shy", "Withdrawn"),
  Agreeableness = c("Cooperative", "Kind", "Sympathetic", "Warm",
                    "Cold", "Harsh", "Rude", "Unsympathetic"),
  Conscientiousness = c("Efficient", "Organized", "Practical", "Systematic",
                        "Careless", "Disorganized", "Inefficient", "Sloppy"),
  Neuroticism = c("Envious", "Fretful", "Jealous", "Moody",
                  "SelfConfident", "SelfAssured", "Unemotional", "Unselfconscious"),
  Openness = c("Artistic", "Imaginative", "Intellectual", "Philosophical",
               "Unartistic", "Uncreative", "Unintellectual", "Unreflective")
)
ALL_ITEMS <- unlist(DOMAINS, use.names = FALSE)

normalize_names <- function(x) {
  x %>%
    str_replace_all("[^A-Za-z0-9]+", "") %>%
    str_trim() %>%
    tolower()
}

# ---------- Data extraction ----------
extract_data <- function(json_path, expected_items) {
  raw <- jsonlite::fromJSON(json_path)
  df <- NULL
  if (is.data.frame(raw)) df <- raw
  if (is.null(df) && !is.null(raw$data) && is.data.frame(raw$data)) df <- raw$data
  if (is.null(df) && !is.null(raw$responses) && is.data.frame(raw$responses)) df <- raw$responses
  if (is.null(df) && !is.null(raw$items) && is.data.frame(raw$items)) df <- raw$items
  if (is.null(df)) stop(sprintf("No tabular data found in %s", basename(json_path)))

  # keep numeric columns
  df <- dplyr::select(df, where(is.numeric))

  # normalize to match names
  df_names_norm <- normalize_names(colnames(df))
  expected_norm <- normalize_names(expected_items)
  map_idx <- match(expected_norm, df_names_norm)
  keep <- !is.na(map_idx)
  matched_df <- df[, map_idx[keep], drop = FALSE]
  colnames(matched_df) <- expected_items[keep]
  matched_df
}

# ---------- Build model syntax dynamically using only present items ----------
build_cfa_model_for <- function(domains, present_cols) {
  lines <- c()
  for (lat in names(domains)) {
    items_present <- intersect(domains[[lat]], present_cols)
    if (length(items_present) >= 2) {  # be permissive; std.lv=TRUE helps identification
      rhs <- paste(items_present, collapse = " + ")
      lines <- c(lines, sprintf("%s =~ %s", lat, rhs))
    }
  }
  if (length(lines) == 0) return(NULL)
  paste(lines, collapse = "\n")
}

# ---------- Parse metadata from filename ----------
sanitize <- function(x) {
  x %>% tolower() %>%
    str_replace_all("[^a-z0-9]+", "_") %>%
    str_replace_all("^_|_$", "")
}

parse_meta <- function(filename) {
  # Condition
  condition <- case_when(
    str_detect(filename, regex("binary[_-]?simple|simple[_-]?binary", ignore_case = TRUE)) ~ "Simple Binary",
    str_detect(filename, regex("binary[_-]?elaborated|elaborated[_-]?binary", ignore_case = TRUE)) ~ "Elaborated Binary",
    str_detect(filename, regex("expanded|expanded[_-]?format|expformat", ignore_case = TRUE)) ~ "Expanded Format",
    str_detect(filename, regex("likert", ignore_case = TRUE)) ~ "Likert",
    str_detect(filename, regex("\\bbinary\\b", ignore_case = TRUE)) ~ "Binary",
    TRUE ~ "Unknown"
  )

  # Model
  model_patterns <- c("gpt[-_]?3[.]?5", "gpt[-_]?4o", "gpt[-_]?4", "llama", "deepseek",
                      "claude", "gemini", "mistral", "qwen", "phi", "o1", "o3")
  model <- "unknown"
  for (pat in model_patterns) {
    if (str_detect(filename, regex(pat, ignore_case = TRUE))) {
      model <- str_extract(filename, regex(pat, ignore_case = TRUE))
      break
    }
  }
  list(condition = condition, model = model)
}

# ---------- Fit & extract standardized loadings ----------
fit_and_extract <- function(dat) {
  present_cols <- colnames(dat)
  model_syntax <- build_cfa_model_for(DOMAINS, present_cols)
  if (is.null(model_syntax)) {
    warning("No factor has >=2 present items; skipping fit.")
    return(list(fit=NULL, loadings=NA))
  }

  fit <- tryCatch({
    cfa(model_syntax, data = dat, estimator = "MLR", std.lv = TRUE, missing = "fiml")
  }, error = function(e) e)

  if (inherits(fit, "error")) {
    warning(sprintf("CFA failed: %s", fit$message))
    return(list(fit=NULL, loadings=NA))
  }

  std <- inspect(fit, what = "std")
  lambda <- std[["lambda"]]
  if (is.null(lambda)) {
    warning("inspect(..., 'std')[['lambda']] returned NULL; returning NA loadings")
    return(list(fit = fit, loadings = NA))
  }

  if (is.null(rownames(lambda))) {
    warning("Lambda has no rownames; cannot align items. Returning NA.")
    return(list(fit = fit, loadings = NA))
  }

  item_names <- rownames(lambda)
  item_loads <- apply(lambda, 1, function(row) row[which.max(abs(row))])
  names(item_loads) <- item_names
  list(fit = fit, loadings = item_loads)
}

# ---------- Main ----------
json_files <- list.files(INPUT_DIR, pattern = "\\.json$", recursive = TRUE, full.names = TRUE)
if (length(json_files) == 0) stop(sprintf("No JSON files found under: %s", INPUT_DIR))

rows <- list()

for (fp in json_files) {
  fname <- basename(fp)
  meta <- parse_meta(fname)
  message(sprintf("➡️  Processing: %s  |  Condition=%s  Model=%s", fname, meta$condition, meta$model))

  dat <- tryCatch({
    extract_data(fp, ALL_ITEMS)
  }, error = function(e) {
    warning(sprintf("Skipping %s due to data extraction error: %s", fname, e$message))
    NULL
  })
  if (is.null(dat)) next

  res <- fit_and_extract(dat)
  if (is.null(res$fit) || all(is.na(res$loadings))) {
    # produce NA row with all items
    load_vec <- setNames(rep(NA_real_, length(ALL_ITEMS)), ALL_ITEMS)
  } else {
    # align to full item list, NAs for missing
    load_vec <- res$loadings
    missing_items <- setdiff(ALL_ITEMS, names(load_vec))
    if (length(missing_items) > 0) {
      load_vec <- c(load_vec, setNames(rep(NA_real_, length(missing_items)), missing_items))
    }
    load_vec <- load_vec[ALL_ITEMS]
  }

  # save per-fit loadings
  per_df <- tibble(Item = names(load_vec), Loading = as.numeric(load_vec))
  per_name <- paste0(sanitize(meta$condition), "__", sanitize(meta$model), "__", sanitize(tools::file_path_sans_ext(fname)), "_loadings.csv")
  readr::write_csv(per_df, file.path(LOAD_DIR, per_name))

  # keep names when building the summary row
  row_df <- tibble::as_tibble_row(as.list(load_vec))
  row_df <- mutate(row_df, Condition = meta$condition, Model = meta$model, .before = 1L)
  rows[[length(rows) + 1]] <- row_df
}

summary_df <- bind_rows(rows)
readr::write_csv(summary_df, OUT_CSV)

message("✅ Done.")
message(sprintf("  - Aggregated CSV: %s", OUT_CSV))
message(sprintf("  - Per-model loadings: %s", LOAD_DIR))
