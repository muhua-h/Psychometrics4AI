# Study 4 Multi-Model Behavioral Validation

This directory contains the **COMPLETED** multi-model replication of the original Study 4, which validates personality-driven behavioral responses in moral reasoning and risk-taking scenarios across multiple LLM models.

## Overview

Study 4 has been fully implemented and extends the original single-model behavioral validation to compare how different LLMs simulate human personality-driven decision-making in:
- **Moral reasoning scenarios**: Ethical dilemmas testing empathy vs. rule-following
- **Risk-taking scenarios**: Decision situations testing risk aversion vs. opportunity-seeking

## Implementation Status

✅ **COMPLETED**: Full multi-model behavioral validation framework

## Directory Structure

```
study_4/
├── README.md                                    # This documentation
├── simulation/                                  # Simulation scripts
│   ├── study_4_moral_multi_model_simulation.py     # Original moral simulation
│   ├── study_4_risk_multi_model_simulation.py      # Original risk simulation  
│   ├── study_4_generalized_combined_simulation.py  # Generalized simulation framework
│   ├── study_4_generalized_moral_simulation.py     # Generalized moral simulation
│   └── study_4_generalized_risk_simulation.py      # Generalized risk simulation
├── analysis/                                    # Analysis scripts
│   ├── study_4_moral_behavioral_analysis.py        # Moral validation analysis
│   ├── study_4_risk_behavioral_analysis.py         # Risk validation analysis
│   └── study_4_generalized_behavioral_analysis.py  # Generalized analysis framework
├── study_4_generalized_raw_data/               # All simulation results
│   ├── bfi_binary_simple_format/               # Binary simple format results
│   ├── bfi_binary_elaborated_format/           # Binary elaborated format results
│   ├── bfi_expanded_format/                    # Expanded format results
│   ├── bfi_likert_format/                      # Likert format results
│   └── simulation_metadata.json               # Run configuration tracking
├── study_4_generalized_analysis_results/       # Analysis outputs
│   ├── complete_regression_results.csv        # Full statistical analysis
│   ├── aggregated_measures_regression_results.csv  # Cross-format results
│   ├── significant_effects_summary.png        # Significance visualization
│   ├── human_vs_ai_comparison.png             # Human-AI comparison charts
│   └── complete_filled_table_fixed_ordered.tex  # LaTeX results table
└── recover_missing_participants.py            # Data recovery utility
```

## Models Tested

- **GPT-4** (OpenAI)
- **GPT-4o** (OpenAI) 
- **Llama-3.3-70B** (Meta)
- **DeepSeek-V3** (DeepSeek)

## Data Source

- **Human Data**: `../../raw_data/york_data_clean.csv`
  - **337 participants** (English comprehension ≥ 4, full York dataset)  
  - **337 simulation entries** per model/format (complete recovery achieved)
  - **325+ valid AI responses** per model/format (after response validation filtering)
  - BFI-2 personality profiles + behavioral responses
  - Moral and risk scenario ratings (1-10 scale)
- **Simulation Metadata**: `study_4_generalized_raw_data/simulation_metadata.json` - tracks configuration for each run

## Implementation Pipeline

### 1. Simulation Phase

**Original Simulations** (legacy format):
```bash
# Run original moral scenarios
python simulation/study_4_moral_multi_model_simulation.py

# Run original risk scenarios  
python simulation/study_4_risk_multi_model_simulation.py
```

**Generalized Framework** (recommended approach):
```bash
# Run comprehensive simulation across all formats and scenarios
python simulation/study_4_generalized_combined_simulation.py

# Individual scenario simulations
python simulation/study_4_generalized_moral_simulation.py
python simulation/study_4_generalized_risk_simulation.py
```

**Key Features**:
- **Multi-Format Support**: Binary (simple/elaborated), expanded, and likert formats
- **Temperature Variation**: Configurable temperature settings (0.0, 1.0+ supported)
- **Model Coverage**: GPT-4, GPT-4o, Llama-3.3-70B, DeepSeek-V3, GPT-3.5-turbo
- **Data Recovery**: Automatic retry with `recover_missing_participants.py`
- **Results Storage**: Organized by format in `study_4_generalized_raw_data/`

### 2. Validation Analysis

**Individual Scenario Analysis**:
```bash
# Analyze moral scenarios
python analysis/study_4_moral_behavioral_analysis.py

# Analyze risk scenarios
python analysis/study_4_risk_behavioral_analysis.py
```

**Generalized Analysis** (recommended):
```bash
# Comprehensive cross-format analysis
python analysis/study_4_generalized_behavioral_analysis.py
```

**Analysis Features**:
- **Cross-Format Validation**: Compares binary, expanded, and likert formats
- **Model Performance Ranking**: Statistical comparison across 5 models
- **Human-AI Comparison**: Direct comparison with York dataset patterns
- **Advanced Visualization**: Heatmaps, significance plots, and LaTeX tables
- **Results Storage**: `study_4_generalized_analysis_results/`

### 3. Data Recovery & Validation

**Missing Participant Recovery**:
```bash
python recover_missing_participants.py
```
- Identifies and recovers incomplete simulation runs
- Supports all formats and models
- Maintains data integrity across recovery attempts
- Updates simulation metadata automatically

## Key Analysis Metrics

### Model Performance
- **Significance Rate**: Proportion of personality-behavior associations reaching p < 0.05
- **Effect Size**: Average R² for significant associations
- **Consistency Score**: Reliability of effects across personality traits
- **Overall Score**: Weighted combination of above metrics

### Behavioral Validation
- **Matched Participant Analysis**: Human baseline uses full 337 participants from York dataset
- **Personality Regression**: Big Five traits predicting behavioral choices (n=325 valid AI, n=337 human)
- **Cross-Scenario Consistency**: Trait effects across moral vs. risk domains
- **Data Recovery**: Complete participant recovery achieved for all models and scenarios

## Scenario Details

### Moral Reasoning (5 scenarios)
1. **Confidential_Info**: Healthcare privacy vs. community support
2. **Underage_Drinking**: Legal compliance vs. family relationships  
3. **Exam_Cheating**: Academic integrity vs. friendship loyalty
4. **Honest_Feedback**: Truthfulness vs. relationship preservation
5. **Workplace_Theft**: Policy enforcement vs. empathy for struggling colleague

### Risk-Taking (5 scenarios)
1. **Investment**: High-risk stocks vs. safe government bonds
2. **Extreme_Sports**: Base jumping thrill vs. physical safety
3. **Entrepreneurial_Venture**: Startup opportunity vs. job security
4. **Confessing_Feelings**: Romantic confession vs. friendship preservation
5. **Study_Overseas**: International education vs. family/social stability

## Expected Findings

Based on the original Study 4 and psychological literature:

### Moral Reasoning
- **Conscientiousness** → Higher ethical rule-following
- **Agreeableness** → More empathetic responses  
- **Neuroticism** → Increased moral anxiety/conflict

### Risk-Taking
- **Openness** → Higher risk tolerance
- **Extraversion** → More social risk-taking
- **Neuroticism** → Risk aversion in uncertain situations

## Usage Examples

### Quick Start
```bash
# Run comprehensive multi-format analysis
python simulation/study_4_generalized_combined_simulation.py
python analysis/study_4_generalized_behavioral_analysis.py

# Legacy individual analyses (if needed)
python simulation/study_4_moral_multi_model_simulation.py
python simulation/study_4_risk_multi_model_simulation.py
python analysis/study_4_moral_behavioral_analysis.py  
python analysis/study_4_risk_behavioral_analysis.py

# Data recovery (if simulation incomplete)
python recover_missing_participants.py
```

### Individual Model Testing
```python
# Test single model for moral scenarios
from simulation.study_4_generalized_moral_simulation import run_moral_simulation, load_york_data

data = load_york_data()
participants = data.to_dict('records')
results = run_moral_simulation(participants, 'gpt-4', 0.0, 'bfi_expanded_format')
```

## Output Files

### Raw Simulation Data
- **Organized by Format**: Results stored in `study_4_generalized_raw_data/`
  - `bfi_binary_simple_format/`: Binary simple personality descriptions
  - `bfi_binary_elaborated_format/`: Binary elaborated personality descriptions  
  - `bfi_expanded_format/`: Full expanded personality descriptions
  - `bfi_likert_format/`: Likert-scale personality descriptions
  - `simulation_metadata.json`: Run configuration and tracking

### Analysis Results
- **Comprehensive Analysis**: `study_4_generalized_analysis_results/`
  - `complete_regression_results.csv`: Full statistical analysis across all formats
  - `aggregated_measures_regression_results.csv`: Cross-format comparison results
  - `complete_filled_table_fixed_ordered.tex`: Publication-ready LaTeX table
  - `human_vs_ai_comparison.png`: Direct human-AI behavioral pattern comparison
  - `significant_effects_summary.png`: Cross-format significance visualization
  - `moral_scenarios_coefficients.png`: Moral scenario effect sizes
  - `risk_scenarios_coefficients.png`: Risk scenario effect sizes

### Legacy Files (Original Implementation)
- `study_4_moral_results/`: Original moral simulation outputs (deprecated)
- `study_4_risk_results/`: Original risk simulation outputs (deprecated)
- `study_4_generalized_results/`: Original generalized results (deprecated)
- `unified_behavioral_analysis_results/`: Original unified analysis (deprecated)

## Validation Against Original Study

The multi-model implementation maintains compatibility with the original Study 4 methodology:

1. **Same Data Source**: York behavioral dataset with identical filtering
2. **Identical Scenarios**: Exact scenario texts and rating scales  
3. **Consistent Analysis**: Same regression approach and significance testing
4. **Extended Scope**: Adds cross-model comparison and enhanced validation

## Performance Considerations

- **Runtime**: ~30-45 minutes per model for full simulation (337 participants × 2 scenarios)
- **API Costs**: ~$12-25 per model depending on provider
- **Memory**: Minimal requirements, results saved incrementally
- **Parallelization**: Batch processing with configurable concurrency
- **Data Recovery**: Automated retry logic ensures complete participant coverage

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
- **Complete Recovery Achieved**: All models now have full 337 participant coverage
- Data validation ensures response quality and format compliance

## Integration with Study 2 Patterns

Study 4 follows the established multi-model framework from Study 2:
- **Shared Utilities**: Uses `simulation_utils.py` for common functions
- **Portal Integration**: Unified model interface via `portal.py`
- **Analysis Patterns**: Consistent regression and visualization approaches
- **Directory Structure**: Parallel organization for easy navigation

## Future Extensions

Potential enhancements for future research:
1. **Temperature Variation**: Test different creativity levels (temp 0.0, 0.5, 1.0)
2. **Prompt Engineering**: Alternative personality description formats
3. **Scenario Expansion**: Additional moral/risk domains
4. **Cultural Validation**: Cross-cultural personality-behavior relationships
5. **Longitudinal Analysis**: Stability of LLM behavioral patterns over time

## Citation

When using this implementation, please cite both the original Study 4 methodology and this multi-model extension.

---

**Status**: ✅ **REFACTORED AND ENHANCED** - Multi-format behavioral validation with advanced analysis

## Current Analysis Results (Multi-Format)

### Model Performance Rankings (Aggregated Across All Formats)
1. **GPT-4o**: Highest cross-format consistency across moral and risk scenarios
2. **GPT-4**: Strong performance with enhanced prompt variations
3. **Llama-3.3-70B**: Robust behavioral pattern replication
4. **DeepSeek-V3**: Reliable personality-behavior relationships
5. **GPT-3.5-turbo**: Baseline performance comparison

### Format-Specific Performance
- **Expanded Format**: Highest personality-behavior correlation fidelity
- **Likert Format**: Strong psychometric properties
- **Binary Elaborated**: Good balance of simplicity and detail
- **Binary Simple**: Most conservative personality interpretations

### Key Behavioral Patterns (Cross-Format Validation)
- **Neuroticism**: Most consistent predictor across all formats and scenarios
- **Conscientiousness**: Strong moral reasoning effects in all formats
- **Extraversion**: Distinct patterns for moral vs. risk scenarios
- **Openness**: Consistent risk-taking associations across formats
- **Agreeableness**: Empathy-driven moral responses across formats

### Advanced Analysis Features
- **Cross-Format Robustness**: Same personality effects validated across 4 formats
- **Temperature Sensitivity**: Behavioral patterns at temp=0.0 vs temp=1.0
- **Statistical Power**: Enhanced with 5 models × 4 formats × 337 participants
- **Publication Ready**: LaTeX tables and high-resolution visualizations

## Key Enhancements from Refactoring

### Directory Structure Improvements
- **Clear Organization**: `simulation/` and `analysis/` directories
- **Scalable Storage**: Hierarchical raw data structure by format
- **Metadata Tracking**: Simulation configuration preservation
- **Legacy Support**: Backward compatibility with original implementation

### Enhanced Framework Features
- **Multi-Format Support**: 4 personality description formats
- **Temperature Variation**: Configurable creativity levels
- **Model Expansion**: 5 models including GPT-3.5-turbo
- **Data Recovery**: Automated missing participant handling
- **Statistical Rigor**: Advanced regression modeling and validation

### Analysis Capabilities
- **Cross-Format Comparison**: Direct format performance evaluation
- **Human-AI Validation**: Direct comparison with York dataset
- **Publication Outputs**: LaTeX tables and publication-quality figures
- **Reproducibility**: Complete simulation metadata and configuration tracking