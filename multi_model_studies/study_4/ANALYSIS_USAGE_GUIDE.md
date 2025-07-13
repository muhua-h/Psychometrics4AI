# Study 4 Analysis Usage Guide

## Overview
This guide explains how to use the Study 4 multi-model behavioral analysis system to process simulation data and generate comprehensive reports.

## Fixed Issues
The following issues were resolved to make the analysis system functional:

1. **File Loading Logic**: Updated to handle actual simulation file naming patterns (`temp1` instead of `temp0.0`)
2. **Model Names**: Added support for all available models including `openai-gpt-3.5-turbo-0125`
3. **File Path Resolution**: Fixed path references to work with current directory structure

## Available Analysis Scripts

### 1. Risk Behavioral Analysis
**File**: `study_4_risk_behavioral_analysis.py`
**Purpose**: Analyzes risk-taking behavior simulation results across all models

**Usage**:
```bash
python study_4_risk_behavioral_analysis.py
```

**Outputs**:
- `study_4_risk_results/risk_regression_results.csv` - Detailed regression results
- `study_4_risk_results/risk_coefficients_heatmap.png` - Coefficient visualization
- `study_4_risk_results/risk_significance_counts.png` - Significance patterns
- Console output with detailed statistical analysis

### 2. Moral Behavioral Analysis
**File**: `study_4_moral_behavioral_analysis.py`
**Purpose**: Analyzes moral reasoning simulation results across all models

**Usage**:
```bash
python study_4_moral_behavioral_analysis.py
```

**Outputs**:
- `study_4_moral_results/moral_regression_results.csv` - Detailed regression results
- `study_4_moral_results/moral_coefficients_heatmap.png` - Coefficient visualization
- `study_4_moral_results/moral_significance_counts.png` - Significance patterns
- Console output with detailed statistical analysis

### 3. Unified Behavioral Analysis
**File**: `unified_behavioral_analysis.py`
**Purpose**: Combines moral and risk analysis results for comprehensive model comparison

**Usage**:
```bash
python unified_behavioral_analysis.py
```

**Outputs**:
- `unified_behavioral_analysis_results/unified_analysis_report.txt` - Comprehensive report
- `unified_behavioral_analysis_results/model_performance_rankings.csv` - Model rankings
- `unified_behavioral_analysis_results/personality_trait_patterns.csv` - Trait patterns
- Multiple visualization files in `unified_behavioral_analysis_results/`

## Data Requirements

### Input Files Required:
1. **Human Data**: `../../study_4/simulation/data_w_simulation.csv` (from original study)
2. **Simulation Results**: JSON files in appropriate results directories:
   - `study_4_risk_results/risk_[model]_temp1.json`
   - `study_4_moral_results/moral_[model]_temp1.json`

### Supported Models:
- `gpt-4`
- `gpt-4o`
- `llama`
- `deepseek`
- `openai-gpt-3.5-turbo-0125`

## Analysis Workflow

### Step 1: Individual Analysis
Run both individual analyses first:
```bash
python study_4_risk_behavioral_analysis.py
python study_4_moral_behavioral_analysis.py
```

### Step 2: Unified Analysis
Run the unified analysis to get comprehensive results:
```bash
python unified_behavioral_analysis.py
```

## Key Metrics Analyzed

### Personality Traits (Predictors):
- Openness
- Conscientiousness
- Extraversion
- Agreeableness
- Neuroticism

### Risk Scenarios:
- Investment decisions
- Extreme sports participation
- Entrepreneurial ventures
- Confessing feelings
- Studying overseas

### Moral Scenarios:
- Confidential information handling
- Underage drinking
- Exam cheating
- Honest feedback
- Workplace theft

## Output Interpretation

### Regression Results:
- **Coefficient**: Effect size and direction
- **P-value**: Statistical significance (< 0.05 considered significant)
- **R-squared**: Proportion of variance explained
- **N**: Sample size for analysis

### Model Performance Metrics:
- **Overall Score**: Composite performance measure
- **Significant Associations**: Number and percentage of significant relationships
- **Average R²**: Mean R-squared for significant associations
- **Consistency Score**: Measure of consistent performance across scenarios

### Visualization Files:
- **Heatmaps**: Show coefficient patterns across personality traits and scenarios
- **Significance Counts**: Bar charts showing number of significant associations
- **Performance Rankings**: Model comparison visualizations

## Troubleshooting

### Common Issues:
1. **File Not Found**: Ensure simulation JSON files exist in correct directories
2. **Data Shape Mismatch**: Check that human data file has expected structure
3. **Missing Dependencies**: Install required packages (pandas, numpy, matplotlib, seaborn, statsmodels)

### Data Quality Checks:
- Participants with `Finished == 1` only
- English proficiency level 5 only
- No missing values in key columns
- Valid JSON format for simulation results

## Example Results Summary

From a typical run:
- **Total Participants**: ~276 (filtered to ~225-226 per model)
- **Significant Associations**: 4.3% of all tests
- **Best Model**: openai-gpt-3.5-turbo-0125 (overall score: 0.321)
- **Most Predictive Trait**: neuroticism (10.0% significance rate)

## Next Steps

After running analyses:
1. Review comprehensive report in `unified_behavioral_analysis_results/`
2. Examine visualization files for patterns
3. Use model rankings to select best-performing models
4. Consider trait patterns for future prompt engineering
5. Compare results with human baseline for validation

## Contact & Support

For issues or questions about the analysis system, refer to the main project documentation or create an issue in the project repository. 