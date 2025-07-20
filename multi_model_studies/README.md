# Multi-Model Psychometrics Studies

This directory contains refactored versions of the original psychometrics studies (Studies 2, 3, and 4) that have been updated to work with multiple LLM models using a unified API interface.

## Overview

The original studies were designed to work with OpenAI's GPT-3.5-turbo model. This refactored version extends the research to test the same workflows across multiple state-of-the-art LLM models:

- **GPT-4** (via Azure OpenAI)
- **GPT-4o** (via Azure OpenAI)  
- **Llama-3.3-70B-Instruct** (via Azure AI Inference)
- **DeepSeek-V3** (via Azure AI Inference)

## Directory Structure

```
multi_model_studies/
├── shared/
│   ├── simulation_utils.py          # Unified simulation functions
│   ├── schema_bfi2.py              # BFI-2 expanded scale mapping
│   ├── mini_marker_prompt.py       # Mini-Marker prompt generation
│   ├── moral_stories.py            # Moral reasoning scenarios
│   ├── risk_taking.py              # Risk-taking scenarios
│   └── schema_tda.py               # TDA scale mapping
├── study_2/
│   └── study_2_multi_model_simulation.ipynb
├── study_3/
│   └── study_3_multi_model_simulation.ipynb
├── study_4/
│   └── study_4_multi_model_simulation.ipynb
└── README.md                       # This file
```

## Key Improvements

### 1. Unified API Interface
- All studies now use the `portal.py` module for unified access to multiple LLM APIs
- Consistent error handling and retry logic across all models
- Standardized response processing

### 2. Streamlined Code
- Removed repetitive OpenAI client initialization code
- Centralized batch processing and concurrent execution
- Unified JSON response parsing

### 3. Enhanced Configuration
- Configurable model, temperature, batch size, and worker settings
- Easy switching between models for comparative analysis
- Systematic result storage with model and temperature information

### 4. Better Organization
- All shared utilities moved to `shared/` directory
- Clear separation between study-specific notebooks and common code
- Consistent file naming and result storage

## Studies Description

### Study 2: ✅ **COMPLETED** - Personality Assignment Validation
- **Purpose**: Validate BFI-2 to Mini-Marker personality assignment
- **Models**: Tests binary, expanded, and likert formats across 5 models
- **Output**: Mini-Marker trait ratings from personality descriptions
- **Status**: Full implementation with unified analysis framework
- **Pending**: Factor analysis implementation (R script adaptation needed)

### Study 3: ✅ **COMPLETED** - Facet-Level Parameter Extraction
- **Purpose**: Extract and validate personality at the facet level (12 facets vs 5 domains)
- **Features**: Advanced psychometric analysis with facet-level correlations
- **Comparison**: Binary, expanded, and likert format effectiveness across 5 models
- **Status**: Full implementation with statistical data generation and multi-format analysis
- **Pending**: Factor analysis implementation (R script adaptation needed)

### Study 4: ✅ **COMPLETED** - Behavioral Validation
- **Purpose**: Test personality effects on moral reasoning and risk-taking behavior
- **Scenarios**: 5 moral dilemmas + risk assessment tasks across multiple formats
- **Validation**: Compare LLM responses to empirical human behavioral data
- **Status**: Full implementation with generalized framework and unified analysis

## Usage

### Prerequisites
1. Ensure `portal.py` and `.env` are configured with valid API keys
2. Have the required data files in the project root:
   - `raw_data/Soto_data.xlsx` (for Studies 2 & 3)
   - `study_4/data/york_data_clean.csv` (for Study 4)

### Running a Study
1. Navigate to the appropriate study directory
2. Open the Jupyter notebook
3. Run cells in order
4. Results will be saved in `study_X_results/` directories

### Configuration
Modify the `SimulationConfig` parameters in each notebook:
```python
config = SimulationConfig(
    model="gpt-4",           # Model to use
    temperature=0.0,         # Response randomness
    batch_size=20,          # Participants per batch
    max_workers=8           # Concurrent API calls
)
```

## Output Structure

Each study creates organized output directories:

### Study 2 Results
```
study_2_*_results/
├── bfi_to_minimarker_gpt_4_temp1_0.json
├── bfi_to_minimarker_gpt_4o_temp1_0.json
├── bfi_to_minimarker_llama_temp1_0.json
├── bfi_to_minimarker_deepseek_temp1_0.json
└── bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1_0.json

unified_analysis_results/
├── condition_wise_stats.csv
├── model_condition_stats.csv
├── model_wise_stats.csv
└── unified_convergent_results.csv
```

### Study 3 Results
```
study_3_*_results/
├── bfi_to_minimarker_gpt_4_temp1_0.json
├── bfi_to_minimarker_gpt_4o_temp1_0.json
├── bfi_to_minimarker_llama_temp1_0.json
├── bfi_to_minimarker_deepseek_temp1_0.json
└── bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1_0.json

unified_analysis_results/
├── format_wise_stats.csv
├── model_wise_stats.csv
└── unified_convergent_results.csv
```

### Study 4 Results
```
study_4_*_results/
├── moral_{model}_temp0_0.json
├── risk_{model}_temp0_0.json
└── *_retried.json

study_4_generalized_results/
├── bfi_binary_elaborated_format/
├── bfi_binary_simple_format/
├── bfi_expanded_format/
└── bfi_likert_format/

unified_behavioral_analysis_results/
├── model_performance_rankings.csv
├── personality_trait_patterns.csv
└── coefficient_heatmap_unified.png
```

## Analysis Workflow

After running simulations:

1. **Data Processing**: Use the unified analysis scripts to process JSON results
2. **Statistical Analysis**: Run the unified analysis scripts for comprehensive statistical comparisons
3. **Cross-Model Comparison**: Compare results across models to assess consistency
4. **Factor Analysis**: 🔄 Pending - adapt original R scripts (`factor_analysis.R`) for multi-model results

## Error Handling

The refactored code includes robust error handling:
- Automatic retry with exponential backoff for API failures
- Failed participant tracking and selective retry
- Comprehensive logging of success/failure rates
- Graceful handling of malformed responses

## Performance Considerations

- **Batch Processing**: Configurable batch sizes to balance speed and API rate limits
- **Concurrent Execution**: Multi-threaded processing within batches
- **Progressive Saving**: Results saved incrementally to prevent data loss
- **Memory Management**: Efficient handling of large participant datasets

## Compatibility

- **Backward Compatibility**: Results are compatible with existing analysis notebooks
- **JSON Format**: Maintains the same JSON structure as original studies
- **Data Schema**: Preserves all original data fields and formats

## Current Status

### Completed Studies
- **Study 2**: ✅ Full multi-model personality assignment validation
- **Study 3**: ✅ Full multi-model facet-level parameter extraction
- **Study 4**: ✅ Full multi-model behavioral validation

### Pending Implementation
- **Factor Analysis**: 🔄 R script adaptation for Studies 2 and 3 multi-model results

## Future Extensions

The modular design makes it easy to:
- Add new LLM models by updating `portal.py`
- Implement new personality scales by adding schema files
- Create new study types using the shared utilities
- Extend to other psychometric constructs
- Complete factor analysis implementation for comprehensive validation

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure `sys.path.append('../shared')` is run before imports
2. **Data File Not Found**: Check that data files exist at expected paths
3. **API Key Issues**: Verify `.env` file contains valid API keys
4. **Memory Errors**: Reduce `batch_size` or `max_workers` in configuration

### Getting Help
- Check the original study notebooks for reference implementations
- Review `simulation_utils.py` for detailed function documentation
- Examine `portal.py` for API configuration examples