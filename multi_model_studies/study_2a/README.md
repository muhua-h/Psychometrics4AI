# Study 2a Multi-Model Simulation

This directory contains the **COMPLETED** multi-model replication of Study 2a, which validates BFI-2 to Mini-Marker personality assignment across multiple LLM models and response formats.

## Overview

Study 2a has been fully implemented with comprehensive multi-model testing across three response formats:

1. **Binary Format**: Yes/No personality descriptions
2. **Expanded Format**: Detailed personality descriptions with full context  
3. **Likert Format**: Concise personality descriptions with rating scales

## Implementation Status

✅ **COMPLETED**: Full multi-model simulation and analysis framework
🔄 **PENDING**: Factor analysis implementation (R script adaptation needed)

## Files

### Core Implementation
- `study_2_binary_baseline_simulation.ipynb` - Binary format simulation notebook
- `study_2_expanded_multi_model_simulation.ipynb` - Expanded format simulation notebook
- `study_2_likert_multi_model_simulation.ipynb` - Likert format simulation notebook
- `unified_convergent_analysis.py` - Unified analysis script for all formats
- `recover_missing_participants.py` - Recovery script for failed simulations

### Data
- `shared_data/` - Preprocessed participant data
  - `study2_preprocessed_data.csv` - Main dataset (438 participants)
  - `study2_likert_preprocessed_data.csv` - Likert-specific data

### Results Directories
- `study_2_simple_binary_results/` - Simple binary format results
- `study_2_elaborated_binary_results/` - Elaborated binary format results
- `study_2_expanded_results_i_am/` - Expanded format with "I am" prompts
- `study_2_expanded_results_you_are/` - Expanded format with "You are" prompts
- `study_2_likert_results/` - Likert format results
- `unified_analysis_results/` - Cross-format unified analysis results

## Models Tested

- **GPT-4** (OpenAI)
- **GPT-4o** (OpenAI)
- **Llama-3.3-70B** (Meta)
- **DeepSeek-V3** (DeepSeek)
- **GPT-3.5-Turbo** (OpenAI)

## Data Source

- **Empirical Data**: Soto's BFI-2 dataset (438 participants)
- **Personality Measures**: BFI-2 domain scores (Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness)
- **Target Measure**: Mini-Marker personality scale responses

## Implementation Pipeline

### 1. Multi-Format Simulation

**Binary Format**:
```bash
jupyter notebook study_2_binary_baseline_simulation.ipynb
```

**Expanded Format**:
```bash
jupyter notebook study_2_expanded_multi_model_simulation.ipynb
```

**Likert Format**:
```bash
jupyter notebook study_2_likert_multi_model_simulation.ipynb
```

Each notebook:
- Loads preprocessed participant data
- Converts BFI-2 scores to appropriate format descriptions
- Runs simulations across 5 models with temperature=1.0
- Saves results to respective directories

### 2. Unified Convergent Analysis

```bash
python unified_convergent_analysis.py
```

This comprehensive analysis:
- Processes all available simulation results
- Calculates convergent validity correlations (BFI-2 → Mini-Marker)
- Compares performance across formats and models
- Generates detailed statistical reports and visualizations

## Key Features

### Multi-Format Support
- **Binary Format**: Yes/No personality descriptions (e.g., "Is outgoing, sociable: Yes")
- **Expanded Format**: Detailed personality descriptions with full context and examples
- **Likert Format**: Concise personality descriptions with rating scales (e.g., "Is outgoing, sociable: 5")

### Multi-Model Testing
- **GPT-4**: OpenAI's flagship model
- **GPT-4o**: OpenAI's optimized model
- **Llama-3.3-70B**: Meta's large language model
- **DeepSeek-V3**: DeepSeek's advanced model
- **GPT-3.5-Turbo**: OpenAI's efficient model

### Comprehensive Analysis
- **Convergent Validity**: Correlation between BFI-2 and Mini-Marker domain scores
- **Cross-Format Comparison**: Performance differences across Binary/Expanded/Likert formats
- **Cross-Model Comparison**: Performance differences across LLM models
- **Domain-Level Analysis**: Detailed analysis for each Big Five domain

## Expected Results

### Convergent Validity Benchmarks
- **Excellent Performance**: r > 0.6 (matches original Study 2a performance)
- **Good Performance**: r = 0.4-0.6 (acceptable convergent validity)
- **Moderate Performance**: r = 0.2-0.4 (limited convergent validity)
- **Poor Performance**: r < 0.2 (inadequate convergent validity)

### Format Comparisons
- **Expanded vs Others**: Expected higher performance due to richer context
- **Binary vs Likert**: Expected similar performance with possible slight advantage to Likert
- **Cross-Model Consistency**: Expected consistent ranking across formats

## Output Structure

### Simulation Results
```
study_2_*_results/
├── bfi_to_minimarker_gpt_4_temp1_0.json
├── bfi_to_minimarker_gpt_4o_temp1_0.json
├── bfi_to_minimarker_llama_temp1_0.json
├── bfi_to_minimarker_deepseek_temp1_0.json
└── bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1_0.json
```

### Unified Analysis Results
```
unified_analysis_results/
├── condition_wise_stats.csv
├── model_condition_stats.csv
├── model_wise_stats.csv
├── unified_analysis_log.txt
└── unified_convergent_results.csv
```

## Technical Details

### Data Processing
1. **Load Empirical Data**: Load Soto's BFI-2 data with proper reverse coding
2. **Format Conversion**: Convert to appropriate description format (Binary/Expanded/Likert)
3. **Prompt Generation**: Create personality prompts for each participant
4. **Multi-Model Execution**: Submit prompts to multiple models in parallel
5. **Response Processing**: Validate and store responses with comprehensive metadata
6. **Results Storage**: Save in consistent format for cross-format analysis

### Analysis Pipeline
1. **Data Loading**: Load all simulation results across formats and models
2. **Response Validation**: Filter valid responses and handle missing data
3. **Correlation Analysis**: Calculate BFI-2 → Mini-Marker correlations for each domain
4. **Statistical Testing**: Perform significance testing and effect size calculations
5. **Cross-Comparison**: Compare performance across formats and models
6. **Visualization**: Generate comprehensive charts and summary statistics

### Correlation Measures Explained

The unified analysis calculates three key correlation measures for convergent validity:

- **BFI-Orig**: Correlation between **BFI-2 scores** and **Original Mini-Marker scores** (human responses). This is the empirical baseline correlation showing how well the original human data correlates.

- **BFI-Sim**: Correlation between **BFI-2 scores** and **simulated Mini-Marker scores** (LLM responses). This measures **simulation validity** - how well the LLM simulation matches the target personality test.

- **Orig-Sim**: Correlation between **Original Mini-Marker scores** (human responses) and **simulated Mini-Marker scores** (LLM responses). This provides a **direct comparison** between human and LLM responses on the same personality measure.

**Note**: "Orig" consistently refers to **Original Mini-Marker scores from human participants**, not original BFI-2 scores.

## Pending Implementation

### Factor Analysis
- **Status**: 🔄 Pending implementation
- **Requirement**: Adapt original `study_2a/factor_analysis.R` for multi-model results
- **Scope**: Validate personality structure across formats and models
- **Expected Output**: Factor loadings, model fit indices, and structural validity metrics

## Usage Examples

### Quick Start
```bash
# Run all simulations
jupyter notebook study_2_binary_baseline_simulation.ipynb
jupyter notebook study_2_expanded_multi_model_simulation.ipynb
jupyter notebook study_2_likert_multi_model_simulation.ipynb

# Run unified analysis
python unified_convergent_analysis.py
```

### Individual Format Testing
```python
# Test single format for specific model
from study_2_expanded_multi_model_simulation import run_expanded_simulation

results = run_expanded_simulation('gpt-4', 1.0, 'test_output/')
```

## Performance Considerations

- **Runtime**: ~20-30 minutes per format for full simulation (438 participants × 5 models)
- **API Costs**: ~$15-25 per format depending on provider
- **Memory**: Minimal requirements, results saved incrementally
- **Parallelization**: Batch processing with configurable concurrency

## Troubleshooting

### Common Issues
1. **API Rate Limits**: Implemented exponential backoff and retry logic
2. **JSON Parsing Errors**: Enhanced response validation and extraction
3. **Missing Data**: Robust filtering and NaN handling
4. **Model Timeouts**: Configurable retry attempts with increasing delays

### Error Recovery
- Failed participants automatically retried with exponential backoff
- Partial results saved to prevent data loss
- Detailed error logging for debugging

## Integration with Other Studies

Study 2a serves as the foundation for the multi-model framework:
- **Shared Utilities**: Uses `simulation_utils.py` for common functions
- **Portal Integration**: Unified model interface via `portal.py`
- **Analysis Patterns**: Consistent regression and visualization approaches
- **Directory Structure**: Parallel organization for easy navigation

## Future Extensions

Potential enhancements for future research:
1. **Temperature Variation**: Test different creativity levels (temp 0.0, 0.5, 1.0)
2. **Prompt Engineering**: Alternative personality description formats
3. **Scale Expansion**: Test with additional personality measures
4. **Cultural Validation**: Cross-cultural personality assessment
5. **Longitudinal Analysis**: Stability of LLM personality patterns over time
