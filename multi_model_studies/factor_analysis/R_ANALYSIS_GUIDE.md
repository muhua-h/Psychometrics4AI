# R-Based Factor Analysis Guide

## Overview
This guide documents the R-based confirmatory factor analysis (CFA) system created to replace the Python implementation, providing proper fit indices (omega, RMSEA, SRMR, CFI) and reliability measures.

## What Was Accomplished

### ✅ Created Robust R-Based CFA System
- **Proper CFA implementation** using `lavaan` package
- **Comprehensive fit indices**: CFI, TLI, RMSEA, SRMR
- **Reliability measures**: Cronbach's Alpha, McDonald's Omega
- **Automatic data processing**: reverse coding, scale detection
- **Separate results storage**: No overwriting of existing Python results

### ✅ Key Files Created
- `cfa_analysis_simple.R` - Main CFA analysis script
- `run_batch_r_analysis.sh` - Batch processing for all studies
- New directory structure: `results_r/` for all R outputs

## Quick Start

### Single File Analysis
```bash
# Run on one JSON file
Rscript cfa_analysis_simple.R \
  /path/to/simulation_results.json \
  /output/directory/
```

### Batch Analysis (All Studies)
```bash
# Run comprehensive analysis across all studies
./run_batch_r_analysis.sh
```

## Results Structure

All R analysis results are saved in:
```
/multi_model_studies/factor_analysis/results_r/
├── study_2/
│   ├── binary_simple_format/
│   ├── binary_elaborated_format/
│   ├── expanded_format/
│   └── likert_format/
├── study_3/
│   └── [format directories]/
└── study_4/
    └── [format directories]/
```

Each analysis produces:
- `*_R_factor_analysis.csv` - Summary with all fit indices and reliability
- Console output with domain-by-domain results

## Sample Output

| Study | Format | Model | Factor_Domain | Alpha | CFI | TLI | RMSEA | SRMR |
|-------|--------|-------|---------------|-------|-----|-----|-------|------|
| STUDY_2 | binary_elaborated | gpt_4o | Extraversion | 0.995 | 0.942 | 0.918 | 0.271 | 0.011 |
| STUDY_2 | binary_elaborated | gpt_4o | Agreeableness | 0.978 | 0.829 | 0.760 | 0.348 | 0.063 |

## Dependencies

### R Packages Required
```r
install.packages(c("lavaan", "psych", "jsonlite", "dplyr"))
```

## Troubleshooting

### Common Issues
1. **Scale range detection**: Script automatically detects 1-5, 1-7, or 1-9 scales
2. **Reverse coding**: Automatically handles negative items
3. **Missing data**: Uses listwise deletion (na.omit)

### Error Messages
- "Insufficient items": Domain has < 3 items available
- "CFA failed": Likely due to data issues - check sample size and item correlations

## Migration from Python
- **No changes needed** to existing Python results
- **New results** saved in separate `results_r/` directory
- **Backward compatible** - can run both systems independently