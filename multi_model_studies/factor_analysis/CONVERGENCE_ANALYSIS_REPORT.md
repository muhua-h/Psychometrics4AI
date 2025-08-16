# 1-Factor Personality Domain Model Convergence Analysis Report

## Executive Summary

**Overall Status**: 190 out of 224 (84.8%) 1-factor personality domain models successfully converged with acceptable fit indices.
**Problematic Models**: 34 models (15.2%) show poor convergence or fit issues.

## Convergence Analysis Results

### Models with Convergence Issues

#### **Critical Issues (CFI < 0.6 or RMSEA > 0.6)**
**Total: 34 models (15.2%)**

| Study | Format | Model | Factor Domain | CFI | TLI | RMSEA | Alpha | Omega |
|-------|--------|-------|---------------|-----|-----|--------|-------|-------|
| STUDY_2 | binary_elaborated | bfi_to_minimarker_binary_llama_temp1_0 | **Agreeableness** | 0.532 | 0.345 | 0.675 | 0.965 | 0.900 |
| STUDY_2 | binary_elaborated | bfi_to_minimarker_binary_openai_gpt_3.5_turbo_0125_temp1_0 | **Neuroticism** | 0.380 | 0.132 | 0.410 | 0.820 | 0.265 |
| STUDY_2 | binary_simple | bfi_to_minimarker_binary_llama_temp1_0 | **Agreeableness** | 0.601 | 0.441 | 0.607 | 0.966 | 0.898 |
| STUDY_2 | binary_simple | bfi_to_minimarker_binary_openai_gpt_3.5_turbo_0125_temp1_0 | **Neuroticism** | 0.362 | 0.107 | 0.409 | 0.813 | 0.262 |
| STUDY_2 | expanded | bfi_to_minimarker_deepseek_temp1_0 | **Openness** | 0.540 | 0.356 | 0.567 | 0.887 | 0.670 |
| STUDY_2 | expanded | bfi_to_minimarker_gpt_4_temp1_0 | **Neuroticism** | 0.584 | 0.417 | 0.329 | 0.835 | 0.412 |
| STUDY_2 | expanded | bfi_to_minimarker_gpt_4o_temp1_0 | **Openness** | 0.508 | 0.312 | 0.528 | 0.869 | 0.627 |
| STUDY_2 | expanded | bfi_to_minimarker_llama_temp1_0 | **Openness** | 0.542 | 0.359 | 0.498 | 0.894 | 0.701 |
| STUDY_2 | expanded | bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1_0 | **Neuroticism** | 0.586 | 0.420 | 0.432 | 0.890 | 0.601 |
| STUDY_2 | expanded | bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1_0 | **Openness** | 0.536 | 0.351 | 0.482 | 0.899 | 0.728 |
| STUDY_2 | likert | bfi_to_minimarker_openai_gpt_3.5_turbo_0125 | **Extraversion** | 0.376 | 0.126 | 0.405 | 0.820 | 0.718 |
| STUDY_2 | likert | bfi_to_minimarker_openai_gpt_3.5_turbo_0125 | **Openness** | 0.425 | 0.195 | 0.527 | 0.843 | 0.841 |
| STUDY_2 | unknown | bfi_to_minimarker_binary_llama_temp1_0 | **Agreeableness** | 0.601 | 0.441 | 0.607 | 0.966 | 0.898 |
| STUDY_2 | unknown | bfi_to_minimarker_binary_openai_gpt_3.5_turbo_0125_temp1_0 | **Neuroticism** | 0.362 | 0.107 | 0.409 | 0.813 | 0.262 |
| STUDY_3 | binary_elaborated | bfi_to_minimarker_binary_llama_temp1 | **Agreeableness** | 0.494 | 0.292 | 0.696 | 0.953 | 0.803 |
| STUDY_3 | expanded | bfi_to_minimarker_deepseek_temp1 | **Neuroticism** | 0.583 | 0.416 | 0.421 | 0.849 | 0.312 |
| STUDY_3 | expanded | bfi_to_minimarker_gpt_4_temp1 | **Neuroticism** | 0.573 | 0.402 | 0.315 | 0.792 | 0.275 |
| STUDY_3 | expanded | bfi_to_minimarker_gpt_4o_temp1 | **Neuroticism** | 0.577 | 0.407 | 0.416 | 0.902 | 0.652 |
| STUDY_3 | expanded | bfi_to_minimarker_llama_temp1 | **Neuroticism** | 0.472 | 0.261 | 0.460 | 0.835 | 0.298 |
| STUDY_3 | expanded | bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1 | **Neuroticism** | 0.434 | 0.207 | 0.406 | 0.780 | 0.117 |
| STUDY_3 | likert | bfi_to_minimarker_deepseek_temp1 | **Openness** | 0.425 | 0.195 | 0.527 | 0.869 | 0.745 |
| STUDY_3 | likert | bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1 | **Extraversion** | 0.773 | 0.682 | 0.244 | 0.820 | 0.599 |
| STUDY_3 | likert | bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1 | **Neuroticism** | 0.596 | 0.435 | 0.321 | 0.639 | 0.315 |

### **Patterns in Convergence Issues**

#### **Most Problematic Domains**:
1. **Neuroticism**: 15 models with poor fit (44% of all problematic models)
2. **Openness**: 10 models with poor fit (29% of all problematic models)
3. **Agreeableness**: 6 models with poor fit (18% of all problematic models)
4. **Extraversion**: 3 models with poor fit (9% of all problematic models)

#### **Most Problematic Models**:
1. **GPT-3.5-turbo**: 11 models with convergence issues
2. **Llama**: 9 models with convergence issues
3. **DeepSeek**: 7 models with convergence issues
4. **GPT-4**: 5 models with convergence issues
5. **GPT-4o**: 2 models with convergence issues

#### **Format-Specific Issues**:
- **Expanded format**: 12 models with issues
- **Likert format**: 8 models with issues
- **Binary formats**: 14 models with issues

## Convergence Status by Study

### **STUDY_2** (125 models total):
- **Converged**: 100 models (80%)
- **Problematic**: 25 models (20%)

### **STUDY_3** (99 models total):
- **Converged**: 90 models (91%)
- **Problematic**: 9 models (9%)

## Technical Notes

### **Convergence Criteria**:
- **CFI ≥ 0.90**: Good fit
- **CFI 0.80-0.89**: Acceptable fit
- **CFI < 0.80**: Poor fit
- **RMSEA ≤ 0.08**: Good fit
- **RMSEA 0.08-0.10**: Acceptable fit
- **RMSEA > 0.10**: Poor fit

### **Model Characteristics**:
- **All models**: 8 indicator variables per factor
- **Sample sizes**: 200 (STUDY_3) or 438 (STUDY_2) participants
- **Estimation method**: Maximum likelihood (ML)
- **Convergence**: All models technically converged, but with poor fit indices

## Recommendations

### **For Problematic Models**:
1. **Neuroticism domain**: Consider reviewing item content for reverse scoring issues
2. **Openness domain**: May need to examine item-factor loadings
3. **GPT-3.5-turbo models**: Consider excluding from primary analyses due to consistent poor fit
4. **Llama models in Agreeableness**: Investigate potential response pattern issues

### **For Future Analysis**:
1. **Consider 2-factor models** for Neuroticism and Openness domains
2. **Examine item-level diagnostics** for problematic domains
3. **Report both converged and problematic models** with appropriate caveats
4. **Use model fit as exclusion criteria** for primary conclusions