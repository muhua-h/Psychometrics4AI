# Study 3 Multi-Format Convergent Analysis

This document describes the convergent validity analysis for Study 3, which has been expanded to support multiple response formats (Likert, Binary, Expanded) and uses the original Study 3 facet-level data generation approach.

## Key Differences from Study 2

### Data Source
- **Study 2**: Uses empirical BFI-2 and TDA data from Soto's study
- **Study 3**: Uses facet-level statistically simulated BFI-2 data as the baseline

### Data Generation Approach
- **Study 2**: Direct use of empirical data
- **Study 3**: Sophisticated facet-level simulation that preserves correlation structure from original Study 3

### Analysis Scope
- **Study 2**: Single format analysis (primarily Likert)
- **Study 3**: Multi-format analysis (Likert, Binary, Expanded)

### Analysis Focus
- **Study 2**: Compares empirical BFI-2 vs empirical TDA vs LLM-simulated Mini-Marker
- **Study 3**: Compares facet-level simulated BFI-2 vs LLM-simulated Mini-Marker across formats

### Metrics
- **Primary Metric**: Correlation between simulated BFI-2 domain scores and LLM-generated Mini-Marker domain scores
- **Secondary Metrics**: Cross-format performance comparison, cross-model consistency
- **Interpretation**: Higher correlations indicate better LLM personality simulation performance

## File Structure

```
study_3/
├── unified_convergent_analysis.py              # Main analysis script (supports all formats)
├── bfi2_facet_level_parameter_extraction_and_simulation.py  # Data generation script
├── facet_lvl_simulated_data.csv               # Facet-level simulated BFI-2 data (primary)
├── study3_simulated_data.csv                  # Alternative simulated data (legacy)
├── study_3_likert_results/                    # Likert format results
│   ├── bfi_to_minimarker_[model]_temp1.json   # Raw LLM responses per model
│   └── study3_likert_preprocessed_data.csv    # Processed input data
├── study_3_binary_results/                    # Binary format results
│   ├── bfi_to_minimarker_binary_[model]_temp1.json
│   └── study3_binary_preprocessed_data.csv
├── study_3_expanded_results/                  # Expanded format results
│   ├── bfi_to_minimarker_expanded_[model]_temp1.json
│   └── study3_expanded_preprocessed_data.csv
└── unified_analysis_results/                  # Cross-format analysis outputs
    ├── unified_analysis_log.txt               # Detailed analysis log
    ├── unified_convergent_results.csv         # Summary results table
    ├── model_wise_stats.csv                  # Statistics by model
    ├── condition_wise_stats.csv              # Statistics by condition (format)
    ├── model_condition_stats.csv             # Combined statistics
    └── format_comparison_analysis.csv         # Cross-format comparison
```

## Usage

### Prerequisites
1. Generate facet-level data: `python bfi2_facet_level_parameter_extraction_and_simulation.py`
2. Run LLM simulations for desired formats:
   - Likert: Execute `study_3_likert_multi_model_simulation.ipynb`
   - Binary: Execute `study_3_binary_multi_model_simulation.ipynb`
   - Expanded: Execute `study_3_expanded_multi_model_simulation.ipynb`

### Running the Analysis
```bash
cd multi_model_studies/study_3
python unified_convergent_analysis.py
```

The analysis will automatically detect available formats and run comprehensive cross-format analysis.

## Analysis Process

### 1. Data Loading and Validation
- Loads facet-level simulated BFI-2 data (`facet_lvl_simulated_data.csv`)
- Validates data structure and correlation patterns
- Generates BFI-2 domain scores using proper aggregation
- Loads LLM simulation results from all available formats

### 2. Format-Specific Processing
Each format is processed with format-appropriate methods:

#### Likert Format Processing
- Uses standard Mini-Marker aggregation with reverse coding
- Computes domain scores as averages of trait ratings
- Handles missing responses gracefully

#### Binary Format Processing  
- Converts binary responses to numeric scale (Yes=1, No=0)
- Applies appropriate scaling for domain score computation
- Accounts for binary response limitations

#### Expanded Format Processing
- Uses enhanced response parsing for detailed descriptions
- Applies sophisticated text analysis for trait extraction
- Handles variable response lengths and formats

### 3. Domain Score Generation
The script automatically generates BFI-2 domain scores from the facet-level data:

```python
# Uses original Study 3 column structure (bfi1-bfi60)
domain_items = {
    'bfi_e': ['bfi1', 'bfi6', 'bfi11', 'bfi16', 'bfi21', 'bfi26', 'bfi31', 'bfi36', 'bfi41', 'bfi46', 'bfi51', 'bfi56'],
    'bfi_a': ['bfi2', 'bfi7', 'bfi12', 'bfi17', 'bfi22', 'bfi27', 'bfi32', 'bfi37', 'bfi42', 'bfi47', 'bfi52', 'bfi57'],
    'bfi_c': ['bfi3', 'bfi8', 'bfi13', 'bfi18', 'bfi23', 'bfi28', 'bfi33', 'bfi38', 'bfi43', 'bfi48', 'bfi53', 'bfi58'],
    'bfi_n': ['bfi4', 'bfi9', 'bfi14', 'bfi19', 'bfi24', 'bfi29', 'bfi34', 'bfi39', 'bfi44', 'bfi49', 'bfi54', 'bfi59'],
    'bfi_o': ['bfi5', 'bfi10', 'bfi15', 'bfi20', 'bfi25', 'bfi30', 'bfi35', 'bfi40', 'bfi45', 'bfi50', 'bfi55', 'bfi60']
}
```

### 4. Multi-Format Correlation Analysis
- Computes Pearson correlations between BFI-2 and Mini-Marker domain scores for each format
- Generates format-specific and cross-format comparison metrics
- Calculates domain-level and average correlations for each model-format combination

### 5. Cross-Format Validation
- Compares performance across formats for each model
- Identifies format-specific advantages and limitations
- Generates format ranking and consistency metrics

## Output Interpretation

### Primary Metrics
- **BFI-2 vs Mini-Marker Correlation**: Core convergent validity measure
- **Range**: -1.0 to 1.0 (higher is better)
- **Interpretation**: Measures how well LLMs simulate personality structure in each format

### Format-Specific Benchmarks
- **Likert Format**: Expected baseline performance (r ≈ 0.6-0.7 based on original Study 3)
- **Binary Format**: Expected slightly lower performance due to reduced information
- **Expanded Format**: Expected higher performance due to richer context

### Cross-Format Metrics
- **Format Consistency**: Correlation between model rankings across formats
- **Format Advantage**: Relative performance improvement of each format
- **Model-Format Interaction**: Which models perform better with which formats

### Domain-Level Analysis
Analysis provides detailed breakdown for each Big Five domain:
- **Extraversion**: Social and assertive behaviors
- **Agreeableness**: Cooperative and trusting behaviors  
- **Conscientiousness**: Organized and responsible behaviors
- **Neuroticism**: Emotional stability and anxiety
- **Openness**: Intellectual and creative behaviors

## Model-Format Performance Matrix

The analysis generates a comprehensive performance matrix:

| Model | Likert | Binary | Expanded | Average | Rank |
|-------|--------|--------|----------|---------|------|
| GPT-4 | 0.65 | 0.58 | 0.72 | 0.65 | 1 |
| GPT-4o | 0.63 | 0.56 | 0.70 | 0.63 | 2 |
| GPT-3.5 | 0.61 | 0.54 | 0.68 | 0.61 | 3 |
| Llama | 0.58 | 0.51 | 0.65 | 0.58 | 4 |
| DeepSeek | 0.55 | 0.48 | 0.62 | 0.55 | 5 |

*Note: Values shown are hypothetical examples*

## Advanced Analysis Features

### Statistical Validation
- **Correlation Significance**: Tests statistical significance of correlations
- **Effect Size Analysis**: Interprets correlation magnitudes using Cohen's conventions
- **Confidence Intervals**: Provides uncertainty estimates for correlation coefficients

### Format Comparison Tests
- **Paired t-tests**: Compares format performance within models
- **ANOVA**: Tests overall format differences across models
- **Post-hoc Analysis**: Identifies specific format advantages

### Data Quality Checks
- **Missing Data Analysis**: Reports missing responses by format and model
- **Response Validity**: Checks for invalid or nonsensical responses
- **Correlation Sanity Checks**: Validates that correlations are within expected ranges

## Limitations

### Study 3 Specific Limitations
1. **Simulated Baseline**: BFI-2 data is simulated, not empirical (though based on empirical parameters)
2. **Sample Size**: Limited to 200 participants (vs. larger empirical studies)
3. **Format Dependence**: Results may be influenced by specific format implementations

### General Limitations
1. **Correlation vs Understanding**: High correlations don't prove genuine personality understanding
2. **Domain Assumptions**: Assumes Big Five structure applies to LLM responses
3. **Prompt Sensitivity**: Results may vary with different prompt formulations

### Format-Specific Limitations
1. **Binary Format**: Reduced information may limit performance ceiling
2. **Expanded Format**: Longer responses may introduce parsing errors
3. **Likert Format**: May not capture full personality complexity

## Future Enhancements

### Analysis Improvements
1. **Factor Analysis**: Examine whether LLM responses show expected Big Five structure
2. **Reliability Analysis**: Assess internal consistency within domains and formats
3. **Temporal Stability**: Test consistency across multiple simulation runs

### Format Extensions
1. **Hybrid Formats**: Combine elements from different formats
2. **Adaptive Formats**: Adjust format based on model capabilities
3. **Custom Formats**: Develop format-specific optimizations for each model

### Cross-Study Integration
1. **Study 2 Comparison**: Compare with empirical baseline from Study 2
2. **Meta-Analysis**: Combine results across studies for broader insights
3. **Validation Studies**: Test findings with additional personality measures

## Troubleshooting

### Common Issues

1. **Data Generation Errors**
   ```
   Error: Original Study 3 data not found
   ```
   **Solution**: Ensure `../../study_3/likert_format/data.csv` exists

2. **Format Detection Issues**
   ```
   Warning: No Binary format results found
   ```
   **Solution**: Run binary simulation notebook or skip binary analysis

3. **Correlation Computation Errors**
   ```
   Error: Insufficient data for correlation
   ```
   **Solution**: Check for missing responses or invalid data

### Validation Checks

The script includes comprehensive validation:
- **Data Structure Validation**: Ensures proper column names and data types
- **Correlation Range Checks**: Validates correlations are within [-1, 1]
- **Missing Data Reporting**: Identifies and reports missing responses
- **Format Consistency Checks**: Ensures consistent data structure across formats

## Technical Notes

### Data Processing Pipeline
1. **Data Loading**: Loads facet-level simulated data with proper structure
2. **Domain Aggregation**: Computes BFI-2 domain scores from individual items
3. **Format Detection**: Automatically detects available simulation formats
4. **Response Processing**: Parses LLM responses using format-specific methods
5. **Correlation Computation**: Calculates correlations using appropriate methods
6. **Results Integration**: Combines results across formats for comprehensive analysis

### Statistical Methods
- **Correlation**: Pearson product-moment correlation (primary metric)
- **Significance Testing**: Two-tailed tests with Bonferroni correction
- **Effect Size**: Cohen's conventions for correlation interpretation
- **Missing Data**: Pairwise deletion for correlations

### Performance Considerations
- **Memory Usage**: Scales with number of participants, models, and formats
- **Computation Time**: Typically completes in under 2 minutes for full analysis
- **Storage Requirements**: Results saved in multiple formats for flexibility

The analysis provides comprehensive insights into LLM personality simulation performance across multiple formats, enabling detailed comparison and validation of different approaches. 