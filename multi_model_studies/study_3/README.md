# Study 3 Multi-Model Simulation

This directory contains the **COMPLETED** multi-model replication of Study 3, which implements facet-level parameter extraction and personality simulation using multiple LLM models across different response formats.

## Overview

Study 3 has been fully implemented with three main components:

1. **Statistical Data Generation**: Extract parameters from Soto's study and generate synthetic BFI-2 data using facet-level modeling
2. **Multi-Model LLM Simulation**: Use generated personality data to simulate Mini-Marker responses across multiple models and formats
3. **Convergent Validity Analysis**: Compare results across models, formats, and analyze convergent validity

## Implementation Status

✅ **COMPLETED**: Full multi-model simulation and analysis framework
🔄 **PENDING**: Factor analysis implementation (R script adaptation needed)

## Files

### Core Implementation
- `bfi2_facet_level_parameter_extraction_and_simulation.py` - Core data generation script
- `study_3_likert_multi_model_simulation.ipynb` - Multi-model simulation notebook for Likert format
- `study_3_binary_multi_model_simulation.ipynb` - Multi-model simulation notebook for Binary format  
- `study_3_expanded_multi_model_simulation.ipynb` - Multi-model simulation notebook for Expanded format
- `unified_convergent_analysis.py` - Unified analysis script for all formats
- `recover_missing_participants.py` - Recovery script for failed simulations

### Data Generation
- `facet_lvl_simulated_data.csv` - Generated BFI-2 data with proper correlation structure (200 participants)
- `study3_simulated_data.csv` - Alternative simulated data (legacy, for comparison)

### Results Directories
- `study_3_likert_results/` - Likert format simulation results
- `study_3_binary_simple_results/` - Simple binary format simulation results
- `study_3_binary_expanded_results/` - Expanded binary format simulation results
- `study_3_expanded_results/` - Expanded format simulation results
- `unified_analysis_results/` - Cross-format analysis results

## Usage

### Step 1: Generate Synthetic BFI-2 Data

Run the facet-level data generation script:

```bash
cd multi_model_studies/study_3
python bfi2_facet_level_parameter_extraction_and_simulation.py
```

This will:
- Load Soto's original data from `../../study_3/likert_format/data.csv`
- Apply reverse coding to BFI-2 items
- Extract domain and facet-level parameters (means, SDs, correlations)
- Simulate 200 participants using multivariate normal distributions
- Preserve intra-domain and inter-domain correlation structures
- Save results to `facet_lvl_simulated_data.csv`

### Step 2: Multi-Model LLM Simulation

Run simulations for each format:

#### Likert Format
```bash
jupyter notebook study_3_likert_multi_model_simulation.ipynb
```

#### Binary Format  
```bash
jupyter notebook study_3_binary_multi_model_simulation.ipynb
```

#### Expanded Format
```bash
jupyter notebook study_3_expanded_multi_model_simulation.ipynb
```

Each notebook will:
- Load the generated BFI-2 data (`facet_lvl_simulated_data.csv`)
- Convert to appropriate format (Likert/Binary/Expanded descriptions)
- Run simulations across 5 models (GPT-3.5, GPT-4, GPT-4o, Llama, DeepSeek)
- Save results to respective directories

### Step 3: Convergent Validity Analysis

Run the unified analysis:

```bash
python unified_convergent_analysis.py
```

This will analyze all available simulation results and generate comprehensive convergent validity reports.

## Key Features

### Advanced Data Generation
- **Facet-Level Modeling**: Simulates individual BFI-2 items within each domain using empirical parameters
- **Correlation Preservation**: Maintains realistic inter-domain (r≈0.3) and intra-domain (r≈0.44) correlations
- **Multivariate Simulation**: Uses multivariate normal distributions for group-level scores
- **Item-Level Noise**: Adds appropriate noise while preserving correlation structure
- **Proper Reverse Coding**: Handles BFI-2 reverse-coded items correctly

### Multi-Format Support
- **Likert Format**: Concise personality descriptions (e.g., "Is outgoing, sociable: 5")
- **Binary Format**: Yes/No personality descriptions (e.g., "Is outgoing, sociable: Yes")
- **Expanded Format**: Detailed personality descriptions with full context

### Multi-Model Testing
- **GPT-3.5-Turbo**: OpenAI's efficient model
- **GPT-4**: OpenAI's flagship model  
- **GPT-4o**: OpenAI's optimized model
- **Llama-3.1-405B**: Meta's large language model
- **DeepSeek-Chat**: DeepSeek's advanced model

### Comprehensive Analysis
- **Convergent Validity**: Correlation between BFI-2 and Mini-Marker domain scores
- **Cross-Format Comparison**: Performance differences across Likert/Binary/Expanded formats
- **Cross-Model Comparison**: Performance differences across LLM models
- **Domain-Level Analysis**: Detailed analysis for each Big Five domain

## Technical Details

### Data Generation Algorithm
1. **Load Empirical Data**: Load Soto's BFI-2 data and apply reverse coding
2. **Extract Parameters**: Calculate domain means, SDs, and correlation matrices
3. **Compute Intra-Domain Correlations**: Calculate average correlations within each domain
4. **Generate Group Scores**: Use multivariate normal distribution with empirical correlation matrix
5. **Add Item Noise**: Generate item-level responses with appropriate within-domain correlation
6. **Apply Constraints**: Clip values to valid BFI-2 range (1-5) and convert to integers

### Simulation Pipeline
1. **Data Loading**: Load generated BFI-2 data with proper column structure
2. **Format Conversion**: Convert to appropriate description format (Likert/Binary/Expanded)
3. **Prompt Generation**: Create personality prompts for each participant
4. **Multi-Model Execution**: Submit prompts to multiple models in parallel
5. **Response Processing**: Validate and store responses with comprehensive metadata
6. **Results Storage**: Save in consistent format for cross-format analysis

## Expected Results

### Convergent Validity Benchmarks
- **Excellent Performance**: r > 0.6 (matches original Study 3 performance)
- **Good Performance**: r = 0.4-0.6 (acceptable convergent validity)
- **Moderate Performance**: r = 0.2-0.4 (limited convergent validity)
- **Poor Performance**: r < 0.2 (inadequate convergent validity)

### Format Comparisons
- **Likert vs Binary**: Expected similar performance with possible slight advantage to Likert
- **Expanded vs Others**: Expected higher performance due to richer context
- **Cross-Model Consistency**: Expected consistent ranking across formats

## Differences from Study 2

1. **Data Source**: Uses sophisticated statistical simulation instead of empirical data
2. **Sample Size**: 200 participants (vs. 438 in Study 2)  
3. **Data Quality**: Preserves realistic correlation structures through facet-level modeling
4. **Format Coverage**: Supports all three formats (Likert, Binary, Expanded)
5. **Validation**: Tests whether LLMs can replicate personality structure with synthetic data

## Dependencies

- pandas
- numpy
- jupyter
- scipy
- concurrent.futures
- Custom modules from `../shared/`

## Troubleshooting

### Common Issues

1. **Data Generation Errors**
   ```
   Error: Original Study 3 data not found
   ```
   **Solution**: Ensure `../../study_3/likert_format/data.csv` exists

2. **Simulation Errors**
   ```
   Error: facet_lvl_simulated_data.csv not found
   ```
   **Solution**: Run data generation script first

3. **API Errors**
   ```
   Error: Model access denied
   ```
   **Solution**: Verify model access through portal.py system

### Performance Tips

- Generate data once and reuse for all formats
- Run simulations in parallel but respect API rate limits
- Use checkpointing to resume interrupted simulations
- Monitor correlation results to validate data quality

## Pending Implementation

### Factor Analysis
- **Status**: 🔄 Pending implementation
- **Requirement**: Adapt original `study_3/factor_analysis.R` for multi-model results
- **Scope**: Validate personality structure across formats and models
- **Expected Output**: Factor loadings, model fit indices, and structural validity metrics

## Next Steps

1. **Factor Analysis Implementation**: Complete psychometric validation with R scripts
2. **Cross-Study Validation**: Compare Study 3 results with Study 2 empirical baselines
3. **Format Optimization**: Identify optimal prompt formats for each model
4. **Scale Expansion**: Test with larger sample sizes and additional personality measures
5. **Methodological Validation**: Validate statistical simulation approach against other methods 