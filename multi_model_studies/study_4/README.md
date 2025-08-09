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
├── study_4_moral_multi_model_simulation.py     # Moral scenario simulation
├── study_4_risk_multi_model_simulation.py      # Risk scenario simulation  
├── study_4_moral_behavioral_analysis.py        # Moral validation analysis
├── study_4_risk_behavioral_analysis.py         # Risk validation analysis
├── unified_behavioral_analysis.py              # Cross-scenario unified analysis
├── study_4_generalized_combined_simulation.py  # Generalized simulation framework
├── study_4_generalized_behavioral_analysis.py  # Generalized analysis framework
├── study_4_moral_results/                      # Moral simulation outputs
├── study_4_risk_results/                       # Risk simulation outputs
├── study_4_generalized_results/                # Generalized simulation outputs
└── unified_behavioral_analysis_results/        # Combined analysis outputs
```

## Models Tested

- **GPT-4** (OpenAI)
- **GPT-4o** (OpenAI) 
- **Llama-3.3-70B** (Meta)
- **DeepSeek-V3** (DeepSeek)

## Data Source

- **Human Data**: `../../raw_data/york_data_clean.csv`
  - **337 participants** (English comprehension ≥ 4, full York dataset)  
  - **337 simulation entries** per model (complete recovery achieved)
  - **325 valid AI responses** per model (after response validation filtering)
  - BFI-2 personality profiles + behavioral responses
  - Moral and risk scenario ratings (1-10 scale)

## Implementation Pipeline

### 1. Simulation Phase

**Moral Scenarios** (`study_4_moral_multi_model_simulation.py`):
```bash
python study_4_moral_multi_model_simulation.py
```
- Loads York behavioral data
- Generates personality-driven prompts using `moral_stories.py`
- Runs simulations across 4 models with temperature=0.0
- Saves results to `study_4_moral_results/`

**Risk Scenarios** (`study_4_risk_multi_model_simulation.py`):
```bash
python study_4_risk_multi_model_simulation.py
```
- Uses same participant data
- Generates prompts using `risk_taking.py`
- Parallel processing with retry logic
- Saves results to `study_4_risk_results/`

**Generalized Framework** (`study_4_generalized_combined_simulation.py`):
```bash
python study_4_generalized_combined_simulation.py
```
- Unified framework for both moral and risk scenarios
- Supports multiple personality formats (binary, expanded, likert)
- Comprehensive multi-model testing across all formats
- Saves results to `study_4_generalized_results/`

### 2. Validation Analysis

**Moral Analysis** (`study_4_moral_behavioral_analysis.py`):
```bash
python study_4_moral_behavioral_analysis.py
```
- Correlates personality traits with simulated moral responses
- Regression analysis: Big Five → Moral scenario ratings
- Compares with human baseline patterns
- Generates visualizations and significance testing

**Risk Analysis** (`study_4_risk_behavioral_analysis.py`):
```bash
python study_4_risk_behavioral_analysis.py
```
- Same methodology applied to risk-taking scenarios
- Validates personality → risk behavior relationships
- Cross-model comparison of behavioral patterns

**Generalized Analysis** (`study_4_generalized_behavioral_analysis.py`):
```bash
python study_4_generalized_behavioral_analysis.py
```
- Unified analysis framework for both scenarios
- Cross-format comparison of behavioral patterns
- Comprehensive model performance evaluation
- Advanced statistical modeling and visualization

### 3. Unified Analysis

**Combined Analysis** (`unified_behavioral_analysis.py`):
```bash
python unified_behavioral_analysis.py
```
- Merges moral + risk validation results
- Model performance ranking and comparison
- Cross-scenario personality pattern analysis
- Comprehensive visualization dashboard
- Summary report with recommendations

**Generalized Analysis** (`study_4_generalized_behavioral_analysis.py`):
```bash
python study_4_generalized_behavioral_analysis.py
```
- Advanced unified analysis across all formats and scenarios
- Cross-format behavioral pattern comparison
- Comprehensive model performance evaluation
- Advanced statistical modeling and visualization

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
# Run full pipeline
python study_4_moral_multi_model_simulation.py
python study_4_risk_multi_model_simulation.py
python study_4_moral_behavioral_analysis.py  
python study_4_risk_behavioral_analysis.py
python unified_behavioral_analysis.py

# Or use generalized framework
python study_4_generalized_combined_simulation.py
python study_4_generalized_behavioral_analysis.py
```

### Individual Model Testing
```python
# Test single model for moral scenarios
from study_4_moral_multi_model_simulation import run_moral_simulation, load_york_data

data = load_york_data()
participants = data.to_dict('records')
results = run_moral_simulation(participants, 'gpt-4', 0.0, 'test_output/')
```

## Output Files

### Simulation Results
- `moral_{model}_temp0.0.json`: Raw moral scenario responses
- `risk_{model}_temp0.0.json`: Raw risk scenario responses
- `*_retried.json`: Results after failed participant retry

### Generalized Results
- `study_4_generalized_results/`: Comprehensive results across all formats and scenarios
  - `bfi_binary_elaborated_format/`: Binary elaborated format results
  - `bfi_binary_simple_format/`: Binary simple format results
  - `bfi_expanded_format/`: Expanded format results
  - `bfi_likert_format/`: Likert format results

### Analysis Results  
- `moral_regression_results.csv`: Personality→moral regression coefficients
- `risk_regression_results.csv`: Personality→risk regression coefficients
- `*_coefficients_heatmap.png`: Visualization of trait effects
- `*_significance_counts.png`: Model comparison charts

### Generalized Analysis Results
- `study_4_generalized_analysis_results/`: Comprehensive analysis results
  - `aggregated_measures_coefficients.png`: Cross-format coefficient visualization
  - `aggregated_measures_regression_results.csv`: Unified regression results
  - `complete_regression_results.csv`: Complete statistical analysis

### Unified Analysis
- `unified_analysis_report.txt`: Comprehensive findings summary
- `model_performance_rankings.csv`: Performance metrics by model
- `personality_trait_patterns.csv`: Cross-scenario trait effects
- `coefficient_heatmap_unified.png`: Full trait×model×scenario heatmap

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

**Status**: ✅ **FULLY IMPLEMENTED AND ANALYZED** - Complete with comprehensive results

## Current Analysis Results

### Model Performance Rankings
1. **GPT-4o**: Overall Score 0.540 (67.5% significant associations, avg R² 0.114)
2. **GPT-4**: Overall Score 0.510 (60.0% significant associations, avg R² 0.093)  
3. **Llama-3.3-70B**: Overall Score 0.493 (55.0% significant associations, avg R² 0.103)
4. **DeepSeek-V3**: Overall Score 0.474 (50.0% significant associations, avg R² 0.098)

### Key Behavioral Patterns
- **Most Predictive Trait**: Neuroticism (78.1% significance rate across scenarios)
- **Cross-Scenario Effects**: 496/960 total significant personality-behavior associations
- **Human vs AI**: Clear personality-driven behavioral patterns in both human and AI responses
- **Scenario Differences**: Distinct trait patterns for moral reasoning vs risk-taking behaviors

### Validated Findings
- **Moral Reasoning**: Conscientiousness predicts ethical rule-following (β=0.228, human data)  
- **Risk-Taking**: Extraversion shows negative association with risk preference (β=-0.461, human data)
- **Model Consistency**: All models show interpretable personality-behavior relationships
- **Cross-Model Validation**: Converging evidence across different LLM architectures

## Key Enhancements

### Generalized Framework
- **Multi-Format Support**: Binary (simple/elaborated), expanded, and likert formats
- **Unified Analysis**: Comprehensive cross-format behavioral pattern analysis
- **Advanced Statistics**: Enhanced regression modeling and visualization
- **Scalable Architecture**: Easy extension to additional scenarios and formats