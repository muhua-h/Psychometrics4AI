#!/usr/bin/env python3
"""
Study 4 Unified Multi-Model Behavioral Analysis

This script provides a comprehensive analysis combining moral reasoning and 
risk-taking behavioral validation across multiple LLM models.

Unified Analysis includes:
1. Load and merge moral + risk simulation results across all models
2. Cross-scenario behavioral pattern analysis
3. Model comparison and ranking for behavioral validity
4. Comprehensive visualization dashboard
5. Summary report with key findings and recommendations
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def load_regression_results():
    """Load regression results from both moral and risk analyses"""
    moral_path = Path("study_4_moral_results/moral_regression_results.csv")
    risk_path = Path("study_4_risk_results/risk_regression_results.csv")
    
    results = {}
    
    if moral_path.exists():
        moral_results = pd.read_csv(moral_path)
        moral_results['scenario_type'] = 'moral'
        results['moral'] = moral_results
        print(f"Loaded moral regression results: {len(moral_results)} entries")
    else:
        print("Warning: Moral regression results not found")
        
    if risk_path.exists():
        risk_results = pd.read_csv(risk_path)
        risk_results['scenario_type'] = 'risk'
        results['risk'] = risk_results
        print(f"Loaded risk regression results: {len(risk_results)} entries")
    else:
        print("Warning: Risk regression results not found")
    
    return results

def load_simulation_data():
    """Load all simulation data for cross-scenario analysis"""
    moral_dir = Path("study_4_moral_results")
    risk_dir = Path("study_4_risk_results")
    
    models = ['gpt-4', 'gpt-4o', 'llama', 'deepseek']
    simulation_data = {}
    
    for model in models:
        model_data = {}
        
        # Load moral data
        moral_files = [f"moral_{model}_temp0.0.json", f"moral_{model}_temp0.0_retried.json"]
        for filename in moral_files:
            moral_path = moral_dir / filename
            if moral_path.exists():
                with open(moral_path, 'r') as f:
                    model_data['moral'] = json.load(f)
                break
        
        # Load risk data
        risk_files = [f"risk_{model}_temp0.0.json", f"risk_{model}_temp0.0_retried.json"]
        for filename in risk_files:
            risk_path = risk_dir / filename
            if risk_path.exists():
                with open(risk_path, 'r') as f:
                    model_data['risk'] = json.load(f)
                break
        
        if model_data:
            simulation_data[model] = model_data
            print(f"Loaded simulation data for {model}")
    
    return simulation_data

def analyze_model_performance(regression_results):
    """Analyze and rank model performance across scenarios"""
    if not regression_results:
        return pd.DataFrame()
    
    # Combine all regression results
    all_results = []
    for scenario_type, results in regression_results.items():
        all_results.append(results)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Focus on aggregate measures (sum variables)
    sum_results = combined_results[combined_results['target'].str.contains('_sum')].copy()
    
    # Calculate performance metrics by model
    performance_metrics = []
    
    for model in sum_results['model'].unique():
        if model == 'Human':
            continue  # Skip human baseline for model comparison
            
        model_results = sum_results[sum_results['model'] == model]
        
        # Count significant associations
        significant_count = (model_results['p_value'] < 0.05).sum()
        total_tests = len(model_results)
        
        # Average R-squared for significant results
        sig_results = model_results[model_results['p_value'] < 0.05]
        avg_r2_significant = sig_results['r_squared'].mean() if len(sig_results) > 0 else 0
        
        # Average coefficient magnitude (absolute value)
        avg_coef_magnitude = model_results['coefficient'].abs().mean()
        
        # Calculate model consistency (lower coefficient variance = more consistent)
        coef_std = model_results['coefficient'].std()
        consistency_score = 1 / (1 + coef_std) if coef_std > 0 else 1
        
        performance_metrics.append({
            'model': model,
            'significant_associations': significant_count,
            'total_tests': total_tests,
            'significance_rate': significant_count / total_tests if total_tests > 0 else 0,
            'avg_r2_significant': avg_r2_significant,
            'avg_coefficient_magnitude': avg_coef_magnitude,
            'consistency_score': consistency_score,
            'overall_score': (significant_count / total_tests * 0.4 + 
                             avg_r2_significant * 0.3 + 
                             consistency_score * 0.3) if total_tests > 0 else 0
        })
    
    return pd.DataFrame(performance_metrics).sort_values('overall_score', ascending=False)

def analyze_personality_patterns(regression_results):
    """Analyze personality trait patterns across scenarios and models"""
    if not regression_results:
        return pd.DataFrame()
    
    # Combine results
    all_results = []
    for scenario_type, results in regression_results.items():
        all_results.append(results)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Focus on sum variables and exclude human baseline for pattern analysis
    sum_results = combined_results[
        (combined_results['target'].str.contains('_sum')) & 
        (combined_results['model'] != 'Human')
    ].copy()
    
    # Analyze patterns by personality trait
    trait_patterns = []
    
    for trait in sum_results['predictor'].unique():
        trait_results = sum_results[sum_results['predictor'] == trait]
        
        # Overall significance rate for this trait
        sig_rate = (trait_results['p_value'] < 0.05).mean()
        
        # Average coefficient (direction of effect)
        avg_coefficient = trait_results['coefficient'].mean()
        
        # Consistency across models (standard deviation of coefficients)
        coef_consistency = 1 / (1 + trait_results['coefficient'].std()) if len(trait_results) > 1 else 1
        
        # Scenario-specific patterns
        moral_coef = trait_results[trait_results['scenario_type'] == 'moral']['coefficient'].mean()
        risk_coef = trait_results[trait_results['scenario_type'] == 'risk']['coefficient'].mean()
        
        trait_patterns.append({
            'personality_trait': trait,
            'significance_rate': sig_rate,
            'avg_coefficient': avg_coefficient,
            'consistency_score': coef_consistency,
            'moral_effect': moral_coef,
            'risk_effect': risk_coef,
            'cross_scenario_consistency': 1 - abs(moral_coef - risk_coef) / (abs(moral_coef) + abs(risk_coef) + 0.001)
        })
    
    return pd.DataFrame(trait_patterns).sort_values('significance_rate', ascending=False)

def create_comprehensive_visualizations(regression_results, model_performance, trait_patterns, output_dir):
    """Create comprehensive visualization dashboard"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not regression_results:
        print("No regression results available for visualization")
        return
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Model Performance Comparison
    if not model_performance.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison - Study 4 Behavioral Validation', fontsize=16, fontweight='bold')
        
        # Significance rate
        axes[0,0].bar(model_performance['model'], model_performance['significance_rate'])
        axes[0,0].set_title('Significance Rate by Model')
        axes[0,0].set_ylabel('Proportion of Significant Associations')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Average R-squared for significant results
        axes[0,1].bar(model_performance['model'], model_performance['avg_r2_significant'])
        axes[0,1].set_title('Average R² for Significant Results')
        axes[0,1].set_ylabel('Average R²')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Consistency score
        axes[1,0].bar(model_performance['model'], model_performance['consistency_score'])
        axes[1,0].set_title('Model Consistency Score')
        axes[1,0].set_ylabel('Consistency Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Overall score
        axes[1,1].bar(model_performance['model'], model_performance['overall_score'])
        axes[1,1].set_title('Overall Performance Score')
        axes[1,1].set_ylabel('Overall Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Personality Trait Patterns
    if not trait_patterns.empty:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Personality Trait Patterns Across Scenarios', fontsize=16, fontweight='bold')
        
        # Significance rate by trait
        axes[0,0].barh(trait_patterns['personality_trait'], trait_patterns['significance_rate'])
        axes[0,0].set_title('Significance Rate by Personality Trait')
        axes[0,0].set_xlabel('Significance Rate')
        
        # Cross-scenario comparison
        x = np.arange(len(trait_patterns))
        width = 0.35
        axes[0,1].bar(x - width/2, trait_patterns['moral_effect'], width, label='Moral', alpha=0.8)
        axes[0,1].bar(x + width/2, trait_patterns['risk_effect'], width, label='Risk', alpha=0.8)
        axes[0,1].set_title('Average Effects by Scenario Type')
        axes[0,1].set_ylabel('Average Coefficient')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(trait_patterns['personality_trait'], rotation=45)
        axes[0,1].legend()
        
        # Consistency scores
        axes[1,0].barh(trait_patterns['personality_trait'], trait_patterns['consistency_score'])
        axes[1,0].set_title('Trait Consistency Across Models')
        axes[1,0].set_xlabel('Consistency Score')
        
        # Cross-scenario consistency
        axes[1,1].barh(trait_patterns['personality_trait'], trait_patterns['cross_scenario_consistency'])
        axes[1,1].set_title('Cross-Scenario Consistency')
        axes[1,1].set_xlabel('Consistency Score')
        
        plt.tight_layout()
        plt.savefig(output_path / 'personality_trait_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Coefficient Heatmap
    all_results = []
    for scenario_type, results in regression_results.items():
        all_results.append(results)
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        sum_results = combined_results[
            (combined_results['target'].str.contains('_sum')) & 
            (combined_results['model'] != 'Human')
        ].copy()
        
        if not sum_results.empty:
            # Create pivot table for heatmap
            pivot_data = sum_results.pivot_table(
                index='predictor', 
                columns=['model', 'scenario_type'], 
                values='coefficient',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot_data, annot=True, cmap='RdBu_r', center=0, 
                       fmt='.3f', cbar_kws={'label': 'Regression Coefficient'})
            plt.title('Personality Trait Coefficients: Models × Scenarios')
            plt.ylabel('Personality Trait')
            plt.xlabel('Model × Scenario')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / 'coefficient_heatmap_unified.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Comprehensive visualizations saved to {output_path}/")

def generate_summary_report(model_performance, trait_patterns, regression_results, output_dir):
    """Generate comprehensive summary report"""
    output_path = Path(output_dir)
    
    report = []
    report.append("="*80)
    report.append("STUDY 4 UNIFIED MULTI-MODEL BEHAVIORAL ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    # Model Performance Summary
    if not model_performance.empty:
        report.append("MODEL PERFORMANCE RANKING:")
        report.append("-" * 40)
        for i, (_, row) in enumerate(model_performance.iterrows(), 1):
            report.append(f"{i}. {row['model']}")
            report.append(f"   Overall Score: {row['overall_score']:.3f}")
            report.append(f"   Significant Associations: {row['significant_associations']}/{row['total_tests']} ({row['significance_rate']:.1%})")
            report.append(f"   Avg R² (significant): {row['avg_r2_significant']:.3f}")
            report.append(f"   Consistency Score: {row['consistency_score']:.3f}")
            report.append("")
    
    # Personality Trait Insights
    if not trait_patterns.empty:
        report.append("PERSONALITY TRAIT INSIGHTS:")
        report.append("-" * 40)
        
        # Most predictive traits
        top_traits = trait_patterns.head(3)
        report.append("Most Predictive Traits:")
        for _, trait in top_traits.iterrows():
            report.append(f"  • {trait['personality_trait']}: {trait['significance_rate']:.1%} significance rate")
        
        report.append("")
        
        # Cross-scenario patterns
        report.append("Cross-Scenario Patterns:")
        for _, trait in trait_patterns.iterrows():
            moral_dir = "→" if trait['moral_effect'] > 0 else "←" if trait['moral_effect'] < 0 else "—"
            risk_dir = "→" if trait['risk_effect'] > 0 else "←" if trait['risk_effect'] < 0 else "—"
            report.append(f"  • {trait['personality_trait']}: Moral {moral_dir} ({trait['moral_effect']:.3f}), Risk {risk_dir} ({trait['risk_effect']:.3f})")
        
        report.append("")
    
    # Key Findings
    report.append("KEY FINDINGS:")
    report.append("-" * 40)
    
    if regression_results:
        # Count total significant associations
        all_results = pd.concat([results for results in regression_results.values()])
        total_sig = (all_results['p_value'] < 0.05).sum()
        total_tests = len(all_results[all_results['model'] != 'Human'])
        
        report.append(f"• Total significant associations found: {total_sig}/{total_tests} ({total_sig/total_tests:.1%})")
        
        # Best performing model
        if not model_performance.empty:
            best_model = model_performance.iloc[0]
            report.append(f"• Best performing model: {best_model['model']} (overall score: {best_model['overall_score']:.3f})")
        
        # Most predictive trait
        if not trait_patterns.empty:
            best_trait = trait_patterns.iloc[0]
            report.append(f"• Most predictive personality trait: {best_trait['personality_trait']} ({best_trait['significance_rate']:.1%} significance)")
    
    report.append("")
    report.append("RECOMMENDATIONS:")
    report.append("-" * 40)
    if not model_performance.empty:
        top_model = model_performance.iloc[0]['model']
        report.append(f"• Use {top_model} for future behavioral personality simulations")
        
        if model_performance.iloc[0]['significance_rate'] < 0.3:
            report.append("• Consider prompt engineering improvements to increase behavioral validity")
        
        if len(model_performance) > 1:
            score_range = model_performance['overall_score'].max() - model_performance['overall_score'].min()
            if score_range < 0.2:
                report.append("• Models show similar performance - ensemble approaches may be beneficial")
            else:
                report.append("• Significant performance differences detected - model selection is critical")
    
    report.append("")
    report.append("="*80)
    
    # Save report
    with open(output_path / 'unified_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    # Print to console
    print('\n'.join(report))

def main():
    """Main unified analysis execution"""
    print("="*80)
    print("Study 4 Unified Multi-Model Behavioral Analysis")
    print("="*80)
    
    output_dir = "unified_behavioral_analysis_results"
    
    # Load regression results
    regression_results = load_regression_results()
    
    if not regression_results:
        print("No regression results found. Please run individual analysis scripts first:")
        print("1. python study_4_moral_behavioral_analysis.py")
        print("2. python study_4_risk_behavioral_analysis.py")
        return
    
    # Analyze model performance
    print("\nAnalyzing model performance...")
    model_performance = analyze_model_performance(regression_results)
    
    # Analyze personality patterns
    print("Analyzing personality trait patterns...")
    trait_patterns = analyze_personality_patterns(regression_results)
    
    # Create visualizations
    print("Creating comprehensive visualizations...")
    create_comprehensive_visualizations(regression_results, model_performance, trait_patterns, output_dir)
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(model_performance, trait_patterns, regression_results, output_dir)
    
    # Save detailed results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if not model_performance.empty:
        model_performance.to_csv(output_path / 'model_performance_rankings.csv', index=False)
    
    if not trait_patterns.empty:
        trait_patterns.to_csv(output_path / 'personality_trait_patterns.csv', index=False)
    
    print(f"\n✅ Unified analysis complete! Results saved to {output_dir}/")
    print("\nKey outputs:")
    print(f"  • Comprehensive report: {output_dir}/unified_analysis_report.txt")
    print(f"  • Model rankings: {output_dir}/model_performance_rankings.csv")
    print(f"  • Trait patterns: {output_dir}/personality_trait_patterns.csv")
    print(f"  • Visualizations: {output_dir}/*.png")

if __name__ == "__main__":
    main()