#!/usr/bin/env python3
"""
Study 4 Multi-Model Risk-Taking Behavioral Validation Analysis

This script analyzes the correlation between human personality traits and 
LLM-simulated risk-taking responses across multiple models.

Analysis includes:
1. Load human personality data and LLM simulation results
2. Process risk-taking scenario responses (individual and aggregate)
3. Regression analysis: personality traits predicting risk responses
4. Cross-model comparison of behavioral patterns
5. Validation against original single-model findings
"""

import pandas as pd
import numpy as np
import json
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

def load_human_data():
    """Load and preprocess human personality and risk response data"""
    data_path = Path('../../study_4/simulation/data_w_simulation.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"Loaded human data shape: {data.shape}")
    
    # Apply same filtering as simulation
    data = data[data['Finished'] == 1]
    data = data[data['8) English language reading/comprehension ability:'] == 5]
    data = data.dropna(subset=[data.columns[17]])
    
    print(f"Filtered human data shape: {data.shape}")
    return data

def load_simulation_results(model, results_dir="study_4_risk_results"):
    """Load LLM simulation results for a specific model"""
    results_path = Path(results_dir)
    
    # Try different filename patterns
    possible_files = [
        f"risk_{model}_temp0.0.json",
        f"risk_{model}_temp0.0_retried.json",
        f"risk_{model}.json"
    ]
    
    for filename in possible_files:
        file_path = results_path / filename
        if file_path.exists():
            with open(file_path, 'r') as f:
                results = json.load(f)
            print(f"Loaded {model} results: {len(results)} participants")
            return results
    
    raise FileNotFoundError(f"No results file found for {model} in {results_path}")

def process_risk_responses(results):
    """Process risk-taking scenario responses into numeric format"""
    # Risk scenarios in order: Investment, Extreme_Sports, Entrepreneurial_Venture, Confessing_Feelings, Study_Overseas
    risk_scenarios = ["Investment", "Extreme_Sports", "Entrepreneurial_Venture", "Confessing_Feelings", "Study_Overseas"]
    
    processed = []
    for result in results:
        if isinstance(result, dict) and 'error' not in result:
            risk_responses = {}
            valid_responses = 0
            
            # Extract individual scenario responses
            for scenario in risk_scenarios:
                if scenario in result and isinstance(result[scenario], (int, float)):
                    risk_responses[scenario] = result[scenario]
                    valid_responses += 1
                else:
                    risk_responses[scenario] = np.nan
            
            # Calculate sum if we have valid responses
            if valid_responses >= 3:  # Require at least 3 valid responses
                risk_responses['risk_sum'] = sum(v for v in risk_responses.values() if not np.isnan(v))
            else:
                risk_responses['risk_sum'] = np.nan
                
            processed.append(risk_responses)
        else:
            # Failed simulation - fill with NaN
            processed.append({scenario: np.nan for scenario in risk_scenarios + ['risk_sum']})
    
    return pd.DataFrame(processed)

def run_regression_analysis(human_data, sim_data, model_name):
    """Run regression analysis comparing human personality traits to simulated risk responses"""
    # Personality predictors
    predictors = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    
    # Risk targets (individual scenarios + sum)
    risk_scenarios = ["Investment", "Extreme_Sports", "Entrepreneurial_Venture", "Confessing_Feelings", "Study_Overseas"]
    targets = risk_scenarios + ['risk_sum']
    
    # Combine human and simulation data
    combined_data = human_data.copy()
    for target in targets:
        combined_data[f'sim_{target}'] = sim_data[target]
    
    # Remove rows with missing simulation data
    valid_mask = combined_data[[f'sim_{target}' for target in targets]].notna().all(axis=1)
    analysis_data = combined_data[valid_mask]
    
    print(f"\n{model_name} Analysis - Valid participants: {len(analysis_data)}")
    
    results = []
    
    # Human vs Simulation correlation
    print(f"\n{model_name} - Human vs Simulation Correlations:")
    for target in targets:
        if f'risk_{target.split("_")[-1]}_1' in human_data.columns or target == 'risk_sum':
            if target == 'risk_sum':
                human_col = 'risk_sum'
            else:
                # Map scenario names to human data columns
                scenario_map = {
                    'Investment': 'risk_1_1',
                    'Extreme_Sports': 'risk_2_1', 
                    'Entrepreneurial_Venture': 'risk_3_1',
                    'Confessing_Feelings': 'risk_4_1',
                    'Study_Overseas': 'risk_5_1'
                }
                human_col = scenario_map.get(target)
            
            if human_col and human_col in analysis_data.columns:
                human_vals = analysis_data[human_col].dropna()
                sim_vals = analysis_data[f'sim_{target}'].dropna()
                
                if len(human_vals) > 10 and len(sim_vals) > 10:
                    # Align the data
                    common_idx = human_vals.index.intersection(sim_vals.index)
                    if len(common_idx) > 10:
                        corr, p_val = pearsonr(human_vals[common_idx], sim_vals[common_idx])
                        print(f"  {target}: r = {corr:.3f}, p = {p_val:.3f}")
    
    # Regression analysis for each target
    for target in targets:
        print(f"\n{model_name} - Regression Analysis for {target}")
        print("-" * 60)
        
        for predictor in predictors:
            # Prepare data
            X = analysis_data[predictor].dropna()
            y = analysis_data[f'sim_{target}'].dropna()
            
            # Align data
            common_idx = X.index.intersection(y.index)
            if len(common_idx) < 10:
                print(f"  {predictor}: Insufficient data")
                continue
                
            X_aligned = X[common_idx]
            y_aligned = y[common_idx]
            
            # Add constant for intercept
            X_with_const = sm.add_constant(X_aligned)
            
            try:
                # Fit regression model
                model = sm.OLS(y_aligned, X_with_const).fit()
                
                # Store results
                result = {
                    'model': model_name,
                    'target': target,
                    'predictor': predictor,
                    'coefficient': model.params.iloc[1],
                    'p_value': model.pvalues.iloc[1],
                    'r_squared': model.rsquared,
                    'n_observations': len(common_idx)
                }
                results.append(result)
                
                # Print summary
                print(f"  {predictor}:")
                print(f"    Coefficient: {model.params.iloc[1]:.4f}")
                print(f"    P-value: {model.pvalues.iloc[1]:.4f}")
                print(f"    R-squared: {model.rsquared:.4f}")
                print(f"    N: {len(common_idx)}")
                
            except Exception as e:
                print(f"  {predictor}: Error in regression: {str(e)}")
    
    return pd.DataFrame(results)

def compare_with_human_baseline(human_data):
    """Run regression analysis on human risk data for comparison"""
    print("\nHuman Baseline Analysis:")
    print("=" * 60)
    
    predictors = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']
    targets = ['risk_sum']  # Focus on aggregate measure
    
    results = []
    
    for target in targets:
        print(f"\nHuman Analysis for {target}")
        print("-" * 40)
        
        for predictor in predictors:
            X = human_data[predictor].dropna()
            y = human_data[target].dropna()
            
            # Align data
            common_idx = X.index.intersection(y.index)
            if len(common_idx) < 10:
                continue
                
            X_aligned = X[common_idx]
            y_aligned = y[common_idx]
            X_with_const = sm.add_constant(X_aligned)
            
            try:
                model = sm.OLS(y_aligned, X_with_const).fit()
                
                result = {
                    'model': 'Human',
                    'target': target,
                    'predictor': predictor,
                    'coefficient': model.params.iloc[1],
                    'p_value': model.pvalues.iloc[1],
                    'r_squared': model.rsquared,
                    'n_observations': len(common_idx)
                }
                results.append(result)
                
                print(f"  {predictor}:")
                print(f"    Coefficient: {model.params.iloc[1]:.4f}")
                print(f"    P-value: {model.pvalues.iloc[1]:.4f}")
                print(f"    R-squared: {model.rsquared:.4f}")
                
            except Exception as e:
                print(f"  {predictor}: Error: {str(e)}")
    
    return pd.DataFrame(results)

def visualize_results(all_results, output_dir="study_4_risk_results"):
    """Create visualizations of the regression results"""
    output_path = Path(output_dir)
    
    # Filter for risk_sum results
    sum_results = all_results[all_results['target'] == 'risk_sum'].copy()
    
    if len(sum_results) == 0:
        print("No risk_sum results to visualize")
        return
    
    # Create coefficient comparison plot
    plt.figure(figsize=(12, 8))
    
    # Pivot data for plotting
    pivot_data = sum_results.pivot(index='predictor', columns='model', values='coefficient')
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, 
                fmt='.3f', cbar_kws={'label': 'Regression Coefficient'})
    plt.title('Risk-Taking: Personality Trait Coefficients by Model')
    plt.xlabel('Model')
    plt.ylabel('Personality Trait')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'risk_coefficients_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create significance plot
    plt.figure(figsize=(10, 6))
    sig_data = sum_results.copy()
    sig_data['significant'] = sig_data['p_value'] < 0.05
    
    # Count significant results by model
    sig_counts = sig_data.groupby('model')['significant'].sum()
    
    plt.bar(sig_counts.index, sig_counts.values)
    plt.title('Significant Personality-Risk Taking Associations by Model')
    plt.xlabel('Model')
    plt.ylabel('Number of Significant Associations (p < 0.05)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / 'risk_significance_counts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_path}/")

def main():
    """Main analysis execution"""
    print("="*80)
    print("Study 4 Multi-Model Risk-Taking Behavioral Validation Analysis")
    print("="*80)
    
    # Load human data
    human_data = load_human_data()
    
    # Models to analyze
    models = ['gpt-4', 'gpt-4o', 'llama', 'deepseek']
    
    # Run human baseline analysis
    human_results = compare_with_human_baseline(human_data)
    
    # Analyze each model
    all_results = []
    
    for model in models:
        try:
            print(f"\n{'='*60}")
            print(f"Analyzing {model}")
            print(f"{'='*60}")
            
            # Load simulation results
            sim_results = load_simulation_results(model)
            
            # Process risk responses
            sim_data = process_risk_responses(sim_results)
            
            # Run regression analysis
            model_results = run_regression_analysis(human_data, sim_data, model)
            all_results.append(model_results)
            
        except Exception as e:
            print(f"Error analyzing {model}: {str(e)}")
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results + [human_results], ignore_index=True)
        
        # Save results
        output_path = Path("study_4_risk_results")
        output_path.mkdir(exist_ok=True)
        combined_results.to_csv(output_path / 'risk_regression_results.csv', index=False)
        
        # Create visualizations
        visualize_results(combined_results)
        
        # Summary
        print("\n" + "="*80)
        print("RISK BEHAVIORAL ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Results saved to: {output_path / 'risk_regression_results.csv'}")
        print("Key findings:")
        
        # Show significant results
        sig_results = combined_results[combined_results['p_value'] < 0.05]
        if len(sig_results) > 0:
            print("\nSignificant associations (p < 0.05):")
            for _, row in sig_results.iterrows():
                print(f"  {row['model']}: {row['predictor']} -> {row['target']} "
                      f"(β = {row['coefficient']:.3f}, p = {row['p_value']:.3f})")
        else:
            print("No significant associations found")
    
    else:
        print("No successful model analyses completed")

if __name__ == "__main__":
    main()