# Binary Dichotomization Analysis

This directory contains the **NEW** binary dichotomization analysis framework that analyzes the correlation between dichotomized BFI-2 scores and Mini-Marker output across both Study 2a and Study 2b.

## Overview

The analysis extends the existing convergent validity framework by:
1. **Dichotomizing continuous BFI-2 scores** using median/mean split
2. **Analyzing correlation** between binary personality classifications and Mini-Marker ratings
3. **Comparing performance** across binary formats (simple vs elaborated)
4. **Validating results** across both empirical (Study 2a) and simulated (Study 2b) data

## Purpose

- **Primary Goal**: Investigate how well LLM models assign Mini-Marker personality ratings when given binary (high/low) personality classifications
- **Methodological Innovation**: Use point-biserial correlation instead of Pearson correlation for binary vs continuous data
- **Research Question**: Do LLMs maintain personality understanding accuracy when personality is presented as categorical rather than continuous?

## Methodology

### Data Processing
1. **Dichotomization**: Split continuous BFI-2 domain scores using median split
   - High group: scores ≥ median
   - Low group: scores < median
2. **Format Specification**: Focus on binary formats only (simple/elaborated)
3. **Data Sources**: 
   - Study 2a: 438 participants from empirical data
   - Study 2b: 200 participants from facet-level simulation

### Statistical Analysis
- **Correlation Method**: Point-biserial correlation (`scipy.stats.pointbiserialr`)
- **Significance Testing**: Two-tailed p-values for each correlation
- **Aggregation**: Average correlations across Big Five domains
- **Cross-validation**: Compare results between empirical and simulated data

## Files Structure

```
binary_dichotomization_analysis/
├── unified_study_analysis.py            # Main analysis script
├── results/                             # Analysis results
│   ├── unified_binary_dichotomized_correlations_median.csv  # Detailed correlations
│   ├── model_wise_stats_median.csv      # Model-level statistics
│   ├── study_wise_stats_median.csv      # Study-level comparison
│   └── format_wise_stats_median.csv     # Format-level comparison
└── README.md                            # This documentation
```

## Usage

### Running the Analysis
```bash
cd multi_model_studies/binary_dichotomization_analysis
python unified_study_analysis.py
```

### Configuration Options
The script supports different cutoff types for dichotomization:
- `median` (default): Median split
- `mean`: Mean split
- Numeric values: Custom cutoff points

Modify the `cutoff_type` parameter in `unified_study_analysis.py`:
```python
cutoff_type = 'median'  # Options: 'median', 'mean', or numeric value
```

## Results Structure

### Detailed Results (`unified_binary_dichotomized_correlations_median.csv`)
Contains individual correlations for each:
- **Study** (study_2a, study_2b)
- **Model** (gpt_4, gpt_4o, llama, deepseek, gpt_3.5_turbo)
- **Format** (binary_simple, binary_elaborated)
- **Domain correlations** (E, A, C, N, O)
- **Significance tests** (p-values for each correlation)
- **Average correlation** across domains

### Summary Statistics
- **Model-wise**: Performance comparison across LLM models
- **Study-wise**: Empirical vs simulated data comparison
- **Format-wise**: Binary simple vs elaborated comparison

## Key Findings (Example)

Based on median split analysis:
- **Overall correlation**: ~0.35-0.37 across models and formats
- **Best performer**: GPT-4o with elaborated format (~0.39)
- **Format comparison**: Elaborated slightly better than simple format
- **Study comparison**: Similar performance between empirical and simulated data
- **Domain variations**: Extraversion shows strongest correlations

## Technical Notes

### Data Requirements
- **Study 2a**: Requires `/multi_model_studies/study_2a/shared_data/study2_preprocessed_data.csv`
- **Study 2b**: Requires `/multi_model_studies/study_2b/facet_lvl_simulated_data.csv`
- **Simulation Results**: Expects JSON files from binary format simulations

### Error Handling
- Automatic detection of missing data files
- Graceful handling of missing simulation results
- Comprehensive logging for debugging

### Output Files
Results are automatically saved to:
- `results/unified_binary_dichotomized_correlations_[cutoff_type].csv`
- `results/model_wise_stats_[cutoff_type].csv`
- `results/study_wise_stats_[cutoff_type].csv`
- `results/format_wise_stats_[cutoff_type].csv`

## Integration with Existing Framework

This analysis complements the existing multi-model studies by:
1. **Extending binary format analysis** beyond basic convergent validity
2. **Providing categorical personality validation** 
3. **Enabling cross-study validation** between empirical and simulated data
4. **Supporting robustness testing** of LLM personality assignment

## References

- **Original Framework**: Built upon the multi-model studies infrastructure
- **Statistical Method**: Point-biserial correlation for binary vs continuous data
- **Psychometric Validation**: Extends traditional convergent validity approaches

## Future Extensions

- **Alternative cutoffs**: Test different dichotomization thresholds
- **Multi-level analysis**: Explore within-domain vs between-domain effects
- **Cross-format validation**: Extend to expanded and likert formats
- **Factor analysis integration**: Incorporate structural validity testing
