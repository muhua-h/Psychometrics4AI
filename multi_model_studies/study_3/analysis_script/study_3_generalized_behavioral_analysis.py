#!/usr/bin/env python3
"""
Study 2b Generalized Behavioral Analysis - Regression Focus

This script analyzes simulation results from all 4 personality formats:
- expanded, likert, binary_elaborated, binary_simple

For both moral and risk-taking scenarios across multiple models.

Analysis focuses on:
1. Regression analysis: How personality traits predict behavioral outcomes
2. Standardized regression coefficients for better comparison
3. Aggregated behavioral measures (risk_sum, ethic_sum)
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_york_data():
    """Load and preprocess York behavioral data"""
    data_path = Path('../../../raw_data/york_data_clean.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path)
    print(f"Loaded York data shape: {data.shape}")

    # Filter for good English comprehension (value 5 = excellent)
    data = data[data['8) English language reading/comprehension ability:'] == 5]
    # Remove rows with null values in bfi6 column (index 17)
    data = data.dropna(subset=[data.columns[17]])

    print(f"Filtered data shape: {data.shape}")
    return data

def load_simulation_results(base_dir):
    """Load all simulation results from the generalized directory structure"""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Results directory not found: {base_path}")
    
    all_results = {}
    personality_formats = ['bfi_expanded', 'bfi_likert', 'bfi_binary_elaborated', 'bfi_binary_simple']
    scenario_types = ['moral', 'risk']
    models = ['gpt-4', 'gpt-4o', 'llama', 'deepseek', 'openai-gpt-3.5-turbo-0125']
    
    for format_name in personality_formats:
        format_dir = base_path / f"{format_name}_format"
        if not format_dir.exists():
            print(f"Warning: Format directory not found: {format_dir}")
            continue
            
        all_results[format_name] = {}
        
        for scenario_type in scenario_types:
            scenario_dir = format_dir / scenario_type
            if not scenario_dir.exists():
                print(f"Warning: Scenario directory not found: {scenario_dir}")
                continue
                
            all_results[format_name][scenario_type] = {}
            
            for model in models:
                # Look for the model's result file
                pattern = f"{scenario_type}_{model}_temp*.json"
                matching_files = list(scenario_dir.glob(pattern))
                
                if matching_files:
                    result_file = matching_files[0]  # Take the first match
                    try:
                        with open(result_file, 'r') as f:
                            results = json.load(f)
                        all_results[format_name][scenario_type][model] = results
                        print(f"Loaded: {format_name}/{scenario_type}/{model} ({len(results)} results)")
                    except Exception as e:
                        print(f"Error loading {result_file}: {e}")
                else:
                    print(f"No results found for {format_name}/{scenario_type}/{model}")
    
    return all_results

def extract_personality_traits(data):
    """Extract personality traits from York data"""
    # Extract personality trait scores directly by column name
    traits = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    
    personality_data = {}
    for trait in traits:
        if trait in data.columns:
            personality_data[trait] = data[trait].values
        else:
            print(f"Warning: {trait} column not found in data")
            personality_data[trait] = np.full(len(data), np.nan)
    
    return personality_data

def extract_behavioral_responses(simulation_results, scenario_type, model, format_name):
    """Extract behavioral responses for a specific scenario, model, and format"""
    if format_name not in simulation_results:
        return None
    
    if scenario_type not in simulation_results[format_name]:
        return None
    
    if model not in simulation_results[format_name][scenario_type]:
        return None
    
    results = simulation_results[format_name][scenario_type][model]
    if not isinstance(results, list):
        return None
    
    # Extract valid responses
    valid_responses = []
    for i, result in enumerate(results):
        if isinstance(result, dict) and 'error' not in result:
            result['participant_index'] = i
            valid_responses.append(result)
    
    return valid_responses

def perform_standardized_regression_analysis(personality_data, behavioral_data, predictor_name, target_name):
    """Perform standardized linear regression analysis"""
    X = personality_data[predictor_name]
    y = behavioral_data
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(X) | np.isnan(y))
    if np.sum(valid_mask) < 10:  # Need at least 10 valid pairs
        return None
    
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    # Standardize variables (z-score)
    X_std = (X_clean - np.mean(X_clean)) / np.std(X_clean)
    y_std = (y_clean - np.mean(y_clean)) / np.std(y_clean)
    
    # Add constant term
    X_with_const = sm.add_constant(X_std)
    
    # Fit model
    model = sm.OLS(y_std, X_with_const).fit()
    
    return {
        'standardized_coefficient': model.params[1],  # Index 1 is the predictor coefficient (0 is intercept)
        'p_value': model.pvalues[1],
        'r_squared': model.rsquared,
        'adj_r_squared': model.rsquared_adj,
        'std_error': model.bse[1],
        't_statistic': model.tvalues[1],
        'n_valid': len(X_clean)
    }

def calculate_aggregated_measures(responses, scenario_type):
    """Calculate aggregated measures (risk_sum or ethic_sum) from individual scenario responses"""
    if scenario_type == 'moral':
        scenarios = ['Confidential_Info', 'Underage_Drinking', 'Exam_Cheating', 'Honest_Feedback', 'Workplace_Theft']
        measure_name = 'ethic_sum'
    else:  # risk
        scenarios = ['Investment', 'Extreme_Sports', 'Entrepreneurial_Venture', 'Confessing_Feelings', 'Study_Overseas']
        measure_name = 'risk_sum'
    
    aggregated_scores = []
    
    for response in responses:
        if isinstance(response, dict) and 'participant_index' in response:
            # Sum up all valid scenario scores
            scenario_scores = []
            for scenario in scenarios:
                if scenario in response and isinstance(response[scenario], (int, float)):
                    scenario_scores.append(response[scenario])
            
            if scenario_scores:
                aggregated_scores.append(sum(scenario_scores))
            else:
                aggregated_scores.append(np.nan)
        else:
            aggregated_scores.append(np.nan)
    
    return np.array(aggregated_scores)

def analyze_human_baseline_regressions(human_data):
    """Perform regression analysis on human baseline data"""
    personality_data = extract_personality_traits(human_data)
    predictors = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    
    # Extract behavioral measures from human data
    ethic_sum = human_data['ethic_sum'].values if 'ethic_sum' in human_data.columns else None
    risk_sum = human_data['risk_sum'].values if 'risk_sum' in human_data.columns else None
    
    if ethic_sum is None or risk_sum is None:
        print("Warning: ethic_sum or risk_sum not found in human data")
        return pd.DataFrame()
    
    human_baseline_results = []
    
    # Analyze ethic_sum (moral behavior)
    for predictor in predictors:
        if predictor in personality_data:
            regression_result = perform_standardized_regression_analysis(
                personality_data, ethic_sum, predictor, 'ethic_sum'
            )
            
            if regression_result:
                result = {
                    'format': 'human_baseline',
                    'scenario_type': 'moral',
                    'target_measure': 'ethic_sum',
                    'model': 'human',
                    'predictor': predictor,
                    **regression_result
                }
                human_baseline_results.append(result)
    
    # Analyze risk_sum (risk behavior)
    for predictor in predictors:
        if predictor in personality_data:
            regression_result = perform_standardized_regression_analysis(
                personality_data, risk_sum, predictor, 'risk_sum'
            )
            
            if regression_result:
                result = {
                    'format': 'human_baseline',
                    'scenario_type': 'risk',
                    'target_measure': 'risk_sum',
                    'model': 'human',
                    'predictor': predictor,
                    **regression_result
                }
                human_baseline_results.append(result)
    
    return pd.DataFrame(human_baseline_results)

def analyze_behavioral_regressions(human_data, simulation_results):
    """Perform regression analysis for all combinations including aggregated measures and human baseline"""
    personality_data = extract_personality_traits(human_data)
    predictors = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    
    # Define scenario mappings
    moral_scenarios = ['Confidential_Info', 'Underage_Drinking', 'Exam_Cheating', 'Honest_Feedback', 'Workplace_Theft']
    risk_scenarios = ['Investment', 'Extreme_Sports', 'Entrepreneurial_Venture', 'Confessing_Feelings', 'Study_Overseas']
    
    all_regression_results = []
    
    # First, add human baseline results
    human_baseline_results = analyze_human_baseline_regressions(human_data)
    if not human_baseline_results.empty:
        all_regression_results.extend(human_baseline_results.to_dict('records'))
        print(f"Added {len(human_baseline_results)} human baseline regression results")
    
    for format_name in simulation_results.keys():
        for scenario_type in ['moral', 'risk']:
            scenarios = moral_scenarios if scenario_type == 'moral' else risk_scenarios
            
            for model in ['gpt-4', 'gpt-4o', 'llama', 'deepseek', 'openai-gpt-3.5-turbo-0125']:
                # Extract behavioral responses
                responses = extract_behavioral_responses(simulation_results, scenario_type, model, format_name)
                if not responses:
                    continue
                
                # First, analyze aggregated measures (risk_sum/ethic_sum) - keep at top
                aggregated_scores = calculate_aggregated_measures(responses, scenario_type)
                measure_name = 'ethic_sum' if scenario_type == 'moral' else 'risk_sum'
                
                # Perform regression for aggregated measures
                for predictor in predictors:
                    if predictor in personality_data:
                        regression_result = perform_standardized_regression_analysis(
                            personality_data, aggregated_scores, predictor, measure_name
                        )
                        
                        if regression_result:
                            result = {
                                'format': format_name,
                                'scenario_type': scenario_type,
                                'target_measure': measure_name,
                                'model': model,
                                'predictor': predictor,
                                **regression_result
                            }
                            all_regression_results.append(result)
                
                # Then analyze individual scenarios
                for scenario in scenarios:
                    # Extract behavioral ratings for this scenario
                    behavioral_ratings = []
                    for i in range(len(human_data)):
                        participant_response = next((r for r in responses if r['participant_index'] == i), None)
                        if participant_response and scenario in participant_response:
                            rating = participant_response[scenario]
                            if isinstance(rating, (int, float)):
                                behavioral_ratings.append(rating)
                            else:
                                behavioral_ratings.append(np.nan)
                        else:
                            behavioral_ratings.append(np.nan)
                    
                    behavioral_ratings = np.array(behavioral_ratings)
                    
                    # Perform regression for each personality trait
                    for predictor in predictors:
                        if predictor in personality_data:
                            regression_result = perform_standardized_regression_analysis(
                                personality_data, behavioral_ratings, predictor, scenario
                            )
                            
                            if regression_result:
                                result = {
                                    'format': format_name,
                                    'scenario_type': scenario_type,
                                    'target_measure': scenario,
                                    'model': model,
                                    'predictor': predictor,
                                    **regression_result
                                }
                                all_regression_results.append(result)
    
    return pd.DataFrame(all_regression_results)

def create_regression_visualizations(regression_results, output_dir):
    """Create visualizations focused on regression results including human baseline"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    if regression_results.empty:
        print("No regression results to visualize")
        return
    
    # 1. Standardized coefficients heatmap for aggregated measures (including human baseline)
    aggregated_results = regression_results[
        regression_results['target_measure'].isin(['risk_sum', 'ethic_sum'])
    ]
    
    if not aggregated_results.empty:
        plt.figure(figsize=(18, 12))
        
        # Create pivot table for standardized coefficients
        agg_pivot = aggregated_results.pivot_table(
            values='standardized_coefficient',
            index=['format', 'predictor'],
            columns=['model', 'target_measure'],
            aggfunc='mean'
        )
        
        sns.heatmap(agg_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
        plt.title('Standardized Regression Coefficients for Aggregated Measures (Including Human Baseline)')
        plt.tight_layout()
        plt.savefig(output_path / 'aggregated_measures_coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Human baseline vs AI models comparison for aggregated measures
    human_vs_ai_results = aggregated_results[
        aggregated_results['target_measure'].isin(['risk_sum', 'ethic_sum'])
    ]
    
    if not human_vs_ai_results.empty:
        # Create comparison plot
        plt.figure(figsize=(15, 10))
        
        # Separate human and AI results
        human_results = human_vs_ai_results[human_vs_ai_results['format'] == 'human_baseline']
        ai_results = human_vs_ai_results[human_vs_ai_results['format'] != 'human_baseline']
        
        if not human_results.empty and not ai_results.empty:
            # Create comparison data
            comparison_data = []
            
            for _, human_row in human_results.iterrows():
                predictor = human_row['predictor']
                target = human_row['target_measure']
                human_coef = human_row['standardized_coefficient']
                
                # Find corresponding AI results
                ai_subset = ai_results[
                    (ai_results['predictor'] == predictor) & 
                    (ai_results['target_measure'] == target)
                ]
                
                for _, ai_row in ai_subset.iterrows():
                    comparison_data.append({
                        'predictor': predictor,
                        'target_measure': target,
                        'format': ai_row['format'],
                        'model': ai_row['model'],
                        'human_coefficient': human_coef,
                        'ai_coefficient': ai_row['standardized_coefficient'],
                        'difference': ai_row['standardized_coefficient'] - human_coef
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Create heatmap of differences
                diff_pivot = comparison_df.pivot_table(
                    values='difference',
                    index=['format', 'predictor'],
                    columns=['model', 'target_measure'],
                    aggfunc='mean'
                )
                
                sns.heatmap(diff_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
                plt.title('Difference in Standardized Coefficients: AI Models - Human Baseline')
                plt.tight_layout()
                plt.savefig(output_path / 'human_vs_ai_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
    
    # 3. Standardized coefficients heatmap for moral scenarios (excluding aggregated measures)
    moral_results = regression_results[
        (regression_results['scenario_type'] == 'moral') & 
        (~regression_results['target_measure'].isin(['risk_sum', 'ethic_sum']))
    ]
    
    if not moral_results.empty:
        plt.figure(figsize=(15, 10))
        
        moral_pivot = moral_results.pivot_table(
            values='standardized_coefficient',
            index=['format', 'predictor'],
            columns='model',
            aggfunc='mean'
        )
        
        sns.heatmap(moral_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
        plt.title('Standardized Regression Coefficients for Moral Scenarios')
        plt.tight_layout()
        plt.savefig(output_path / 'moral_scenarios_coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Standardized coefficients heatmap for risk scenarios (excluding aggregated measures)
    risk_results = regression_results[
        (regression_results['scenario_type'] == 'risk') & 
        (~regression_results['target_measure'].isin(['risk_sum', 'ethic_sum']))
    ]
    
    if not risk_results.empty:
        plt.figure(figsize=(15, 10))
        
        risk_pivot = risk_results.pivot_table(
            values='standardized_coefficient',
            index=['format', 'predictor'],
            columns='model',
            aggfunc='mean'
        )
        
        sns.heatmap(risk_pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0)
        plt.title('Standardized Regression Coefficients for Risk Scenarios')
        plt.tight_layout()
        plt.savefig(output_path / 'risk_scenarios_coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Significant effects summary (including human baseline)
    significant_results = regression_results[regression_results['p_value'] < 0.05]
    if not significant_results.empty:
        plt.figure(figsize=(12, 8))
        
        # Count significant effects by format and predictor
        sig_counts = significant_results.groupby(['format', 'predictor']).size().reset_index(name='count')
        sig_pivot = sig_counts.pivot(index='format', columns='predictor', values='count')
        
        # Fill NaN values with 0 and convert to int
        sig_pivot = sig_pivot.fillna(0).astype(int)
        
        sns.heatmap(sig_pivot, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Number of Significant Effects (p < 0.05) by Format and Predictor')
        plt.tight_layout()
        plt.savefig(output_path / 'significant_effects_summary.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main analysis function"""
    print("=== Study 2b Generalized Behavioral Analysis - Regression Focus ===")
    
    # Load data
    print("Loading York human data...")
    human_data = load_york_data()
    
    print("Loading simulation results...")
    simulation_results = load_simulation_results('study_4_generalized_results')
    
    # Perform regression analysis
    print("Performing regression analysis...")
    regression_results = analyze_behavioral_regressions(human_data, simulation_results)
    
    # Create output directory
    output_dir = '../study_4_generalized_analysis_results'
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Split and save results
    print("Saving regression results...")
    
    # 1. Aggregated measures results (risk_sum, ethic_sum)
    aggregated_results = regression_results[
        regression_results['target_measure'].isin(['risk_sum', 'ethic_sum'])
    ]
    aggregated_results.to_csv(output_path / 'aggregated_measures_regression_results.csv', index=False)
    print(f"Saved {len(aggregated_results)} aggregated measures results to aggregated_measures_regression_results.csv")
    
    # 2. Specific scenarios results (individual moral and risk scenarios)
    specific_scenarios_results = regression_results[
        ~regression_results['target_measure'].isin(['risk_sum', 'ethic_sum'])
    ]
    specific_scenarios_results.to_csv(output_path / 'specific_scenarios_regression_results.csv', index=False)
    print(f"Saved {len(specific_scenarios_results)} specific scenarios results to specific_scenarios_regression_results.csv")
    
    # Also save the complete results for reference
    regression_results.to_csv(output_path / 'complete_regression_results.csv', index=False)
    print(f"Saved {len(regression_results)} complete results to complete_regression_results.csv")
    
    # Create visualizations
    print("Creating visualizations...")
    create_regression_visualizations(regression_results, output_dir)
    
    # Print summary
    print("\n=== Regression Analysis Summary ===")
    if not regression_results.empty:
        print(f"Total regression analyses: {len(regression_results)}")
        print(f"Significant effects (p < 0.05): {len(regression_results[regression_results['p_value'] < 0.05])}")
        
        # Summary by format
        print(f"\nAverage R-squared by format:")
        print(regression_results.groupby('format')['r_squared'].mean())
        
        # Summary by predictor
        print(f"\nAverage standardized coefficients by predictor:")
        print(regression_results.groupby('predictor')['standardized_coefficient'].mean())
        
        # Summary for aggregated measures
        aggregated_results = regression_results[
            regression_results['target_measure'].isin(['risk_sum', 'ethic_sum'])
        ]
        if not aggregated_results.empty:
            print(f"\nAggregated measures analysis:")
            print(f"Number of aggregated measure analyses: {len(aggregated_results)}")
            print(f"Significant aggregated effects (p < 0.05): {len(aggregated_results[aggregated_results['p_value'] < 0.05])}")
            
            # Human baseline summary
            human_baseline = aggregated_results[aggregated_results['format'] == 'human_baseline']
            if not human_baseline.empty:
                print(f"\nHuman baseline results:")
                print(f"Number of human baseline analyses: {len(human_baseline)}")
                print(f"Significant human baseline effects (p < 0.05): {len(human_baseline[human_baseline['p_value'] < 0.05])}")
                print(f"Human baseline standardized coefficients:")
                for _, row in human_baseline.iterrows():
                    print(f"  {row['predictor']} -> {row['target_measure']}: {row['standardized_coefficient']:.3f} (p={row['p_value']:.3f})")
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main() 