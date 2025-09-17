# Multi-Model Psychometrics Studies

This directory contains the **COMPLETED** multi-model replication of the original psychometrics studies (Studies 2, 3, and 4) that have been updated to work with multiple LLM models using a unified API interface.

## Overview

The original studies were designed to work with OpenAI's GPT-3.5-turbo model. This refactored version extends the research to test the same workflows across multiple state-of-the-art LLM models:

- **GPT-4** (via Azure OpenAI)
- **GPT-4o** (via Azure OpenAI)  
- **Llama-3.3-70B-Instruct** (via Azure AI Inference)
- **DeepSeek-V3** (via Azure AI Inference)
- **GPT-3.5-Turbo** (via OpenAI)

## Current Implementation Status

✅ **ALL STUDIES COMPLETED** - Full multi-model implementation with comprehensive analysis frameworks
🔄 **PENDING**: Factor analysis implementation (R script adaptation needed for Studies 2 and 3)

## Directory Structure

```
multi_model_studies/
├── shared/                                    # Common utilities and modules
│   ├── simulation_utils.py                   # Unified simulation framework
│   ├── enhanced_simulation_utils.py          # Enhanced simulation utilities
│   ├── schema_bfi2.py                       # BFI-2 expanded scale mapping
│   ├── schema_tda.py                        # TDA scale mapping
│   ├── mini_marker_prompt.py                # Mini-Marker prompt generation
│   ├── binary_baseline_prompt.py            # Binary format prompt generation
│   ├── moral_stories.py                     # Moral reasoning scenarios
│   ├── risk_taking.py                       # Risk-taking scenarios
│   └── participant_validation.py            # Participant data validation
├── study_2a/                                  # ✅ COMPLETED - Personality Assignment Validation
│   ├── README.md                            # Study 2a documentation
│   ├── study_2_binary_baseline_simulation.ipynb
│   ├── study_2_expanded_multi_model_simulation.ipynb
│   ├── study_2_likert_multi_model_simulation.ipynb
│   ├── unified_convergent_analysis.py       # Unified analysis script
│   ├── recover_missing_participants.py      # Recovery script
│   ├── shared_data/                         # Preprocessed participant data
│   ├── study_2_simple_binary_results/       # Simple binary format results
│   ├── study_2_elaborated_binary_results/   # Elaborated binary format results
│   ├── study_2_expanded_results_i_am/       # Expanded format (I am) results
│   ├── study_2_expanded_results_you_are/    # Expanded format (You are) results
│   ├── study_2_likert_results/              # Likert format results
│   └── unified_analysis_results/            # Cross-format analysis results
├── study_2b/                                  # ✅ COMPLETED - Facet-Level Parameter Extraction
│   ├── README.md                            # Study 2b documentation
│   ├── bfi2_facet_level_parameter_extraction_and_simulation.py
│   ├── study_3_binary_multi_model_simulation.ipynb
│   ├── study_3_expanded_multi_model_simulation.ipynb
│   ├── study_3_likert_multi_model_simulation.ipynb
│   ├── unified_convergent_analysis.py       # Unified analysis script
│   ├── recover_missing_participants.py      # Recovery script
│   ├── facet_lvl_simulated_data.csv         # Generated BFI-2 data (200 participants)
│   ├── study_3_binary_simple_results/       # Simple binary format results
│   ├── study_3_binary_expanded_results/     # Expanded binary format results
│   ├── study_3_expanded_results/            # Expanded format results
│   ├── study_3_likert_results/              # Likert format results
│   └── unified_analysis_results/            # Cross-format analysis results
├── study_2b/                                  # ✅ COMPLETED - Behavioral Validation
│   ├── README.md                            # Study 2b documentation
│   ├── study_4_moral_multi_model_simulation.py
│   ├── study_4_risk_multi_model_simulation.py
│   ├── study_4_moral_behavioral_analysis.py
│   ├── study_4_risk_behavioral_analysis.py
│   ├── study_4_generalized_combined_simulation.py
│   ├── study_4_generalized_behavioral_analysis.py
│   ├── unified_behavioral_analysis.py       # Unified analysis script
│   ├── recover_failed_responses.py          # Recovery script
│   ├── study_4_moral_results/               # Moral scenario results
│   ├── study_4_risk_results/                # Risk scenario results
│   ├── study_4_generalized_results/         # Generalized framework results
│   ├── study_4_generalized_analysis_results/ # Generalized analysis results
│   └── unified_behavioral_analysis_results/ # Cross-scenario analysis results
└── README.md                                # This file
```

## Studies Description

### Study 2a: ✅ **COMPLETED** - Personality Assignment Validation
- **Purpose**: Validate BFI-2 to Mini-Marker personality assignment across multiple formats
- **Data Source**: Empirical BFI-2 data from Soto's study (438 participants)
- **Formats**: Binary (simple/elaborated), Expanded (I am/You are), Likert
- **Models**: 5 models (GPT-4, GPT-4o, Llama, DeepSeek, GPT-3.5-Turbo)
- **Output**: Mini-Marker trait ratings from personality descriptions
- **Analysis**: Unified convergent validity analysis across all formats and models
- **Status**: Full implementation with comprehensive analysis framework

### Study 2b: ✅ **COMPLETED** - Facet-Level Parameter Extraction
- **Purpose**: Extract and validate personality at the facet level using statistical simulation
- **Data Source**: Facet-level statistically simulated BFI-2 data (200 participants)
- **Formats**: Binary (simple/expanded), Expanded, Likert
- **Models**: 5 models (GPT-4, GPT-4o, Llama, DeepSeek, GPT-3.5-Turbo)
- **Features**: Advanced psychometric analysis with facet-level correlations
- **Analysis**: Unified convergent validity analysis with cross-format comparison
- **Status**: Full implementation with statistical data generation and multi-format analysis

### Study 2b: ✅ **COMPLETED** - Behavioral Validation
- **Purpose**: Test personality effects on moral reasoning and risk-taking behavior
- **Data Source**: York behavioral dataset with 4 personality formats
- **Scenarios**: 5 moral dilemmas + 5 risk assessment tasks
- **Formats**: Binary (simple/elaborated), Expanded, Likert
- **Models**: 4 models (GPT-4, GPT-4o, Llama, DeepSeek)
- **Analysis**: Individual scenario analysis + unified behavioral analysis
- **Status**: Full implementation with generalized framework and comprehensive analysis

## Key Features

### 1. Unified API Interface
- All studies use the `portal.py` module for unified access to multiple LLM APIs
- Consistent error handling and retry logic across all models
- Standardized response processing and validation

### 2. Multi-Format Support
- **Binary Format**: Yes/No personality descriptions with simple and elaborated variants
- **Expanded Format**: Detailed personality descriptions with full context and examples
- **Likert Format**: Concise personality descriptions with rating scales

### 3. Comprehensive Analysis Framework
- **Individual Format Analysis**: Separate analysis for each format and model combination
- **Unified Analysis**: Cross-format and cross-model comparison
- **Convergent Validity**: Correlation between personality measures
- **Behavioral Validation**: Personality effects on decision-making scenarios

### 4. Robust Error Handling
- Automatic retry with exponential backoff for API failures
- Failed participant tracking and selective recovery
- Comprehensive logging of success/failure rates
- Graceful handling of malformed responses

### 5. Parallel Processing
- Concurrent execution across models for improved performance
- Configurable batch sizes and worker settings
- Progressive saving to prevent data loss

## Usage

### Prerequisites
1. Ensure `portal.py` and `.env` are configured with valid API keys
2. Have the required data files in the project root:
   - `raw_data/Soto_data.xlsx` (for Studies 2 & 3)
   - `raw_data/york_data_clean.csv` (for Study 2b)

### Running Studies

#### Study 2a: Personality Assignment Validation
```bash
cd multi_model_studies/study_2a

# Run simulations for each format
jupyter notebook study_2_binary_baseline_simulation.ipynb
jupyter notebook study_2_expanded_multi_model_simulation.ipynb
jupyter notebook study_2_likert_multi_model_simulation.ipynb

# Run unified analysis
python unified_convergent_analysis.py
```

#### Study 2b: Facet-Level Parameter Extraction
```bash
cd multi_model_studies/study_2b

# Generate synthetic data (if needed)
python bfi2_facet_level_parameter_extraction_and_simulation.py

# Run simulations for each format
jupyter notebook study_3_binary_multi_model_simulation.ipynb
jupyter notebook study_3_expanded_multi_model_simulation.ipynb
jupyter notebook study_3_likert_multi_model_simulation.ipynb

# Run unified analysis
python unified_convergent_analysis.py
```

#### Study 2b: Behavioral Validation
```bash
cd multi_model_studies/study_2b

# Run individual scenario simulations
python study_4_moral_multi_model_simulation.py
python study_4_risk_multi_model_simulation.py

# Run generalized framework (recommended)
python study_4_generalized_combined_simulation.py

# Run analysis
python study_4_moral_behavioral_analysis.py
python study_4_risk_behavioral_analysis.py
python study_4_generalized_behavioral_analysis.py
python unified_behavioral_analysis.py
```

### Configuration
Modify the `SimulationConfig` parameters in each script:
```python
config = SimulationConfig(
    model="gpt-4",           # Model to use
    temperature=1.0,         # Response randomness
    batch_size=25,          # Participants per batch
    max_workers=4           # Concurrent API calls
)
```

## Output Structure

### Study 2a Results
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

### Study 2b Results
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

### Study 2b Results
```
study_4_*_results/
├── moral_{model}_temp1_0.json
├── risk_{model}_temp1_0.json
└── *_retried.json

study_4_generalized_results/
├── bfi_binary_elaborated_format/
│   ├── moral/
│   └── risk/
├── bfi_binary_simple_format/
│   ├── moral/
│   └── risk/
├── bfi_expanded_format/
│   ├── moral/
│   └── risk/
└── bfi_likert_format/
    ├── moral/
    └── risk/

unified_behavioral_analysis_results/
├── model_performance_rankings.csv
├── personality_trait_patterns.csv
└── coefficient_heatmap_unified.png
```

## Analysis Workflow

### Convergent Validity Analysis (Studies 2 & 3)
1. **Data Processing**: Load simulation results and validate responses
2. **Trait Extraction**: Extract Mini-Marker traits from LLM responses
3. **Domain Aggregation**: Compute domain scores using proper reverse coding
4. **Correlation Analysis**: Calculate BFI-2 vs Mini-Marker correlations
5. **Cross-Format Comparison**: Compare performance across formats and models
6. **Statistical Testing**: Perform significance testing and effect size analysis

### Behavioral Validation Analysis (Study 2b)
1. **Response Processing**: Load and validate behavioral scenario responses
2. **Personality Integration**: Merge with human personality data
3. **Regression Analysis**: Test personality → behavior relationships
4. **Model Comparison**: Compare model performance across scenarios
5. **Cross-Format Analysis**: Analyze format effects on behavioral predictions
6. **Human Baseline Comparison**: Validate against empirical human data

## Performance Considerations

- **Runtime**: ~20-30 minutes per format for full simulation (200-438 participants × 5 models)
- **API Costs**: ~$15-25 per format depending on provider
- **Memory**: Minimal requirements, results saved incrementally
- **Parallelization**: Batch processing with configurable concurrency

## Pending Implementation

### Factor Analysis
- **Status**: 🔄 Pending implementation
- **Requirement**: Adapt original `study_2a/factor_analysis.R` and `study_2b/factor_analysis.R` for multi-model results
- **Scope**: Validate personality structure across formats and models
- **Expected Output**: Factor loadings, model fit indices, and structural validity metrics

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure `sys.path.append('../shared')` is run before imports
2. **Data File Not Found**: Check that data files exist at expected paths
3. **API Key Issues**: Verify `.env` file contains valid API keys
4. **Memory Errors**: Reduce `batch_size` or `max_workers` in configuration

### Recovery Procedures
- **Failed Simulations**: Use recovery scripts to retry failed participants
- **Missing Data**: Recovery scripts automatically detect and fix missing responses
- **Validation Errors**: Check response format and validation logic

## Future Extensions

The modular design makes it easy to:
- Add new LLM models by updating `portal.py`
- Implement new personality scales by adding schema files
- Create new study types using the shared utilities
- Extend to other psychometric constructs
- Complete factor analysis implementation for comprehensive validation

## Citation

When using this implementation, please cite both the original studies and this multi-model extension.

---

**Status**: ✅ **FULLY IMPLEMENTED** - All studies completed with comprehensive analysis frameworks
**Next Steps**: Implement factor analysis for complete psychometric validation