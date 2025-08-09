# Factor Analysis Results - Organized by Model

This directory contains factor analysis results organized with **one model per table**, grouped by format.

## New Directory Structure

```
factor_analysis_results/
├── study_2/
│   ├── binary_format/
│   │   ├── gpt_4_factor_loadings.csv       # All factors/domains for GPT-4 in binary format
│   │   ├── gpt_4_factor_summary.csv        # Summary statistics for GPT-4 in binary format
│   │   ├── gpt_4o_factor_loadings.csv      # All factors/domains for GPT-4o in binary format
│   │   ├── gpt_4o_factor_summary.csv       # Summary statistics for GPT-4o in binary format
│   │   ├── llama_3.3_70b_factor_loadings.csv
│   │   ├── llama_3.3_70b_factor_summary.csv
│   │   ├── deepseek_v3_factor_loadings.csv
│   │   ├── deepseek_v3_factor_summary.csv
│   │   ├── gpt_3.5_turbo_factor_loadings.csv
│   │   └── gpt_3.5_turbo_factor_summary.csv
│   ├── expanded_format/
│   │   └── ... (same model files as binary)
│   └── likert_format/
│       └── ... (same model files as binary)
├── study_3/
│   └── ... (same structure as study_2)
└── cross_format_comparison/
    ├── comprehensive_model_format_comparison.csv  # All models across all formats
    └── format_model_summary.csv                  # Aggregated statistics

```

## File Contents

### Factor Loadings Files (`*_factor_loadings.csv`)
Each file contains **all factor loadings for one model** with columns:
- **Study**: Study name (Study_2, Study_3)
- **Format**: Response format (binary, expanded, likert)
- **Model**: LLM model name
- **Structure_Type**: Original (Big Five) or Modified (empirical structure)
- **Factor_Domain**: Factor or domain name
- **Item**: Mini-Marker personality item
- **Loading**: Factor loading value
- **Loading_Abs**: Absolute loading value
- **Alpha**: Cronbach's Alpha reliability
- **Omega**: McDonald's Omega reliability
- **Eigenvalue**: Factor eigenvalue
- **Variance_Explained**: Proportion of variance explained
- **N_Items**: Number of items in factor/domain
- **N_Participants**: Number of participants
- **RMSEA, CFI, TLI, SRMR**: Model fit indices
- **File_Source**: Original result file name

### Factor Summary Files (`*_factor_summary.csv`)
Each file contains **summary statistics for one model** with columns:
- **Study, Format, Model, Structure_Type, Factor_Domain**: Identifiers
- **N_Items, N_Participants**: Scale composition
- **Alpha, Omega**: Reliability measures
- **Eigenvalue, Variance_Explained**: Factor strength
- **Mean_Loading_Abs, Max_Loading_Abs, Min_Loading_Abs**: Loading statistics
- **RMSEA, CFI, TLI, SRMR**: Model fit indices
- **Total_Variance_Explained, N_Factors_Total**: Overall structure info (modified only)
- **File_Source**: Original result file name

## Advantages of Model-Based Organization

### 1. **Easy Model Comparison**
- Each model's complete factor structure in one table
- Direct comparison of original vs. modified structures
- All reliability and fit statistics together

### 2. **Format-Based Analysis**
- Clear separation by response format (binary, expanded, likert)
- Easy to see format effects on factor structure
- Straightforward format comparison

### 3. **Comprehensive Model Profiles**
- Complete personality structure for each model
- Both Big Five and empirical structures included
- All psychometric properties in one place

### 4. **Research-Friendly Format**
- One model = one table (standard analysis format)
- Easy import into statistical software
- Direct visualization and modeling

## Usage Examples

### Compare All Models in Expanded Format (Study 2)

```python
import pandas as pd

# Load all expanded format models
gpt4_expanded = pd.read_csv('../factor_analysis_results_new/study_2/expanded_format/gpt_4_factor_loadings.csv')
gpt4o_expanded = pd.read_csv('../factor_analysis_results_new/study_2/expanded_format/gpt_4o_factor_loadings.csv')
llama_expanded = pd.read_csv('../factor_analysis_results_new/study_2/expanded_format/llama_3.3_70b_factor_loadings.csv')

# Compare original structure reliability
models = [gpt4_expanded, gpt4o_expanded, llama_expanded]
for model_df in models:
    original_alpha = model_df[model_df['Structure_Type'] == 'Original']['Alpha'].mean()
    print(f"{model_df['Model'].iloc[0]}: Mean Alpha = {original_alpha:.3f}")
```

### Analyze Format Effects for One Model
```python
# Load GPT-4 across all formats
gpt4_binary = pd.read_csv('study_2/binary_format/gpt_4_factor_loadings.csv')
gpt4_expanded = pd.read_csv('study_2/expanded_format/gpt_4_factor_loadings.csv')
gpt4_likert = pd.read_csv('study_2/likert_format/gpt_4_factor_loadings.csv')

# Compare reliability across formats
for df in [gpt4_binary, gpt4_expanded, gpt4_likert]:
    format_name = df['Format'].iloc[0]
    mean_alpha = df[df['Structure_Type'] == 'Original']['Alpha'].mean()
    print(f"GPT-4 {format_name}: Mean Alpha = {mean_alpha:.3f}")
```

### Cross-Format Model Ranking
```python
# Load comprehensive comparison
comparison = pd.read_csv('cross_format_comparison/comprehensive_model_format_comparison.csv')

# Rank models by reliability across all formats
model_ranking = comparison[comparison['Structure_Type'] == 'Original'].groupby('Model')['Mean_Alpha'].mean().sort_values(ascending=False)
print("Model Ranking by Mean Alpha Reliability:")
print(model_ranking)
```

## Key Improvements

1. **One Model Per Table**: Complete factor structure for each model in single file
2. **Format Organization**: Clear separation by response format type
3. **Comprehensive Data**: All loadings, reliability, and fit statistics together
4. **Easy Comparison**: Direct model-to-model and format-to-format comparison
5. **Research Ready**: Standard format for statistical analysis and visualization

This organization makes it much easier to analyze individual model performance, compare models within formats, and examine format effects on personality structure quality.
