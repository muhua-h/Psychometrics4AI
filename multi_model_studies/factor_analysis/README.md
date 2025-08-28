# Multi-Model Factor Analysis

R-based confirmatory factor analysis for validating personality structure across formats and models.

## Scripts

| Script | Purpose                   | Usage |
|---|---------------------------|---|
| `cfa_analysis_simple.R` | Main CFA with fit indices | `Rscript cfa_analysis_simple.R input.json output_dir/` |
| `export_factor_loadings_v2.R` | Extract factor loadings   | `Rscript export_factor_loadings_v2.R --input_dir dir/ --out_csv results.csv` |
| `create_r_cross_format_comparison.R` | Cross-format summaries    | `Rscript create_r_cross_format_comparison.R` |
| `run_batch_r_analysis.sh` | Batch processing `cfa_analysis_simple.R`      | `./run_batch_r_analysis.sh` |
| `human_factor_analysis.R` | Human data baseline       | Run interactively |

## Results

All results in `results_r/`:
- **Loadings**: `loadings/[format]__[model]__*_loadings.csv`
- **CFA**: `study_[2,3,4]/[format]/[model]_R_factor_analysis.csv`
- **Comparison**: `cross_format_comparison/R_factor_analysis_summary.csv`

## Quick Start

```bash
# Single file
Rscript scripts/cfa_analysis_simple.R study_2.json results_r/study_2/

# All files
./scripts/run_batch_r_analysis.sh

# Comparison
Rscript scripts/create_r_cross_format_comparison.R
```

## Dependencies

```r
install.packages(c("lavaan", "psych", "semTools", "jsonlite", "dplyr", "readr"))
```