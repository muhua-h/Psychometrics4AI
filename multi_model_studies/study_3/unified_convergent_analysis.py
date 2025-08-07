#!/usr/bin/env python3
"""
Unified Convergent Analysis for Study 3 - Multi-Model Multi-Format Simulation

This script analyzes convergent validity between BFI-2 and Mini-Marker scales
across both likert and expanded formats using multiple LLM models.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_original_study3_data():
    """Load the original Study 3 simulated data."""
    # Use the facet-level simulated data that's used for both formats
    data_path = Path(__file__).parent / 'facet_lvl_simulated_data.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Study 3 simulated data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # The Study 3 data already has the correct domain scores (bfi_e, bfi_a, etc.)
    # Use these directly as they already have proper reverse coding applied
    
    # Map them to the expected column names for analysis
    df['bfi2_e'] = df['bfi_e']
    df['bfi2_a'] = df['bfi_a'] 
    df['bfi2_c'] = df['bfi_c']
    df['bfi2_n'] = df['bfi_n']
    df['bfi2_o'] = df['bfi_o']
    
    # For Study 3, we use the same BFI-2 values as baseline since this is simulated data
    # testing convergent validity between BFI-2 and Mini-Marker
    df['tda_e'] = df['bfi_e']
    df['tda_a'] = df['bfi_a']
    df['tda_c'] = df['bfi_c'] 
    df['tda_n'] = df['bfi_n']
    df['tda_o'] = df['bfi_o']
    
    logger.info(f"Loaded Study 3 simulated data: {df.shape}")
    return df

def aggregate_minimarker_responses(df, format_type='likert'):
    """Aggregate Mini-Marker trait ratings to Big Five domain scores."""
    # Use the exact same approach as the original Study 3
    def reverse_score(score):
        return 10 - score

    def calculate_big_five_scores(df):
        # Exact mapping from original Study 3 process_json_bfi_miniMarker.ipynb
        dimensions = {
            'E': [('Bashful', True), ('Bold', False), ('Energetic', False), ('Extraverted', False), 
                  ('Quiet', True), ('Shy', True), ('Talkative', False), ('Withdrawn', True)],
            
            'A': [('Cold', True), ('Cooperative', False), ('Harsh', True), ('Kind', False), 
                  ('Rude', True), ('Sympathetic', False), ('Unsympathetic', True), ('Warm', False)],
            
            'C': [('Careless', True), ('Disorganized', True), ('Efficient', False), ('Inefficient', True), 
                  ('Organized', False), ('Sloppy', True), ('Systematic', False)],
            
            'N': [('Fretful', False), ('Jealous', False), ('Moody', False), ('Relaxed', True), 
                  ('Temperamental', False), ('Touchy', False), ('Envious', False)],
            
            'O': [('Complex', False), ('Creative', False), ('Deep', False), ('Imaginative', False), 
                  ('Intellectual', False), ('Philosophical', False), ('Practical', True), 
                  ('Uncreative', True), ('Unintellectual', True)]
        }
        
        results = {}
        
        for dimension, traits in dimensions.items():
            scores = []
            for trait, reverse in traits:
                if trait in df.columns:
                    trait_scores = df[trait].astype(float)
                    if reverse:
                        trait_scores = trait_scores.apply(reverse_score)
                    scores.append(trait_scores)
                else:
                    logger.warning(f"Trait '{trait}' not found in data for {format_type} format")
            
            if scores:
                # Average the scores for this dimension
                dimension_scores = pd.concat(scores, axis=1).mean(axis=1)
                results[f'miniMarker_simulated_{dimension}'] = dimension_scores
            else:
                logger.warning(f"No traits found for dimension {dimension} in {format_type} format")
        
        return pd.DataFrame(results)
    
    # Calculate Big Five scores
    big_five_df = calculate_big_five_scores(df)
    
    # Merge with original dataframe
    result_df = df.copy()
    for col in big_five_df.columns:
        result_df[col] = big_five_df[col]
    
    logger.info(f"Aggregated Mini-Marker scores for {format_type} format: {big_five_df.shape}")
    return result_df

def load_simulation_results(results_dir, format_type):
    """Load Mini-Marker simulation results for a specific format."""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return {}
    
    simulation_results = {}
    
    # Look for JSON files with simulation results
    for json_file in results_dir.glob("*.json"):
        if json_file.name.startswith("bfi_to_minimarker"):
            model_name = json_file.stem.replace("bfi_to_minimarker_", "").replace("bfi_to_minimarker_binary_", "").replace("_temp1.0", "").replace("_temp1", "")
            
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Parse responses
                responses = []
                for item in data:
                    if isinstance(item, dict):
                        if 'choices' in item:
                            # OpenAI format
                            for choice in item['choices']:
                                content = choice['message']['content']
                                if content.startswith('```json\n') and content.endswith('\n```'):
                                    content = content[7:-4]
                                try:
                                    response_data = json.loads(content)
                                    responses.append(response_data)
                                except json.JSONDecodeError:
                                    logger.warning(f"Could not parse JSON response in {json_file}")
                        else:
                            # Direct response format
                            responses.append(item)
                
                if responses:
                    simulation_results[model_name] = pd.DataFrame(responses)
                    logger.info(f"Loaded {len(responses)} responses for {model_name} ({format_type} format)")
                else:
                    logger.warning(f"No valid responses found in {json_file}")
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
    
    return simulation_results

def calculate_convergent_validity(simulated_data, simulation_results, format_type):
    """Calculate convergent validity correlations for a specific format."""
    results = []
    
    for model_name, sim_df in simulation_results.items():
        logger.info(f"Analyzing {model_name} ({format_type} format)...")
        
        # Ensure we have the same number of participants
        n_participants = min(len(simulated_data), len(sim_df))
        data_subset = simulated_data.head(n_participants).copy()
        sim_subset = sim_df.head(n_participants).copy()
        
        # Aggregate Mini-Marker responses
        sim_aggregated = aggregate_minimarker_responses(sim_subset, format_type)
        
        # Calculate correlations between BFI-2 and simulated Mini-Marker
        domain_correlations = {}
        
        bfi_minimarker_pairs = [
            ('bfi2_e', 'miniMarker_simulated_E'),
            ('bfi2_a', 'miniMarker_simulated_A'),
            ('bfi2_c', 'miniMarker_simulated_C'),
            ('bfi2_n', 'miniMarker_simulated_N'),
            ('bfi2_o', 'miniMarker_simulated_O')
        ]
        
        for bfi_col, mm_col in bfi_minimarker_pairs:
            if bfi_col in data_subset.columns and mm_col in sim_aggregated.columns:
                corr, p_value = pearsonr(data_subset[bfi_col], sim_aggregated[mm_col])
                domain = bfi_col.split('_')[1].upper()
                domain_correlations[f'{domain}_correlation'] = corr
                domain_correlations[f'{domain}_p_value'] = p_value
                logger.info(f"  {domain}: r = {corr:.3f}, p = {p_value:.3f}")
        
        # Calculate average correlation
        correlations = [v for k, v in domain_correlations.items() if k.endswith('_correlation')]
        avg_correlation = np.mean(correlations) if correlations else 0
        
        result = {
            'model': model_name,
            'format': format_type,
            'n_participants': n_participants,
            'avg_correlation': avg_correlation,
            **domain_correlations
        }
        
        results.append(result)
        logger.info(f"  Average correlation: {avg_correlation:.3f}")
    
    return results

def analyze_format_differences(results_df):
    """Analyze differences between all formats."""
    format_stats = {}
    
    for format_type in results_df['format'].unique():
        format_data = results_df[results_df['format'] == format_type]
        if not format_data.empty:
            format_stats[format_type] = {
                'mean_correlation': format_data['avg_correlation'].mean(),
                'std_correlation': format_data['avg_correlation'].std(),
                'min_correlation': format_data['avg_correlation'].min(),
                'max_correlation': format_data['avg_correlation'].max(),
                'n_models': len(format_data['model'].unique())
            }
    
    return format_stats

def main():
    """Main analysis function."""
    logger.info("Starting unified convergent analysis for Study 3 (multi-format)...")
    
    # Load Study 3 simulated data
    try:
        simulated_data = load_original_study3_data()
    except FileNotFoundError as e:
        logger.error(f"Could not load Study 3 data: {e}")
        return
    
    all_results = []
    
    # Define format configurations
    format_configs = [
        {
            'name': 'likert',
            'results_dir': Path(__file__).parent / 'study_3_likert_results'
        },
        {
            'name': 'expanded', 
            'results_dir': Path(__file__).parent / 'study_3_expanded_results'
        },
        {
            'name': 'binary_simple',
            'results_dir': Path(__file__).parent / 'study_3_binary_simple_results'
        },
        {
            'name': 'binary_elaborated',
            'results_dir': Path(__file__).parent / 'study_3_binary_elaborated_results'
        }
    ]
    
    # Analyze each format
    for config in format_configs:
        format_name = config['name']
        results_dir = config['results_dir']
        
        logger.info(f"\n{'='*50}")
        logger.info(f"ANALYZING {format_name.upper()} FORMAT")
        logger.info(f"{'='*50}")
        
        # Load simulation results for this format
        simulation_results = load_simulation_results(results_dir, format_name)
        
        if simulation_results:
            # Calculate convergent validity
            format_results = calculate_convergent_validity(simulated_data, simulation_results, format_name)
            all_results.extend(format_results)
            
            logger.info(f"Completed analysis for {format_name} format: {len(format_results)} model results")
        else:
            logger.warning(f"No simulation results found for {format_name} format in {results_dir}")
    
    if not all_results:
        logger.error("No results to analyze")
        return
    
    # Save results
    output_dir = Path(__file__).parent / 'unified_analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_file = output_dir / 'unified_convergent_results.csv'
    results_df.to_csv(results_file, index=False)
    
    # Generate summary statistics
    summary_stats = {
        'model_wise_stats': results_df.groupby(['model', 'format']).agg({
            'avg_correlation': ['mean', 'std', 'min', 'max'],
            'n_participants': 'first'
        }).round(3).to_dict(),
        
        'format_wise_stats': results_df.groupby('format').agg({
            'avg_correlation': ['mean', 'std', 'min', 'max'],
            'n_participants': 'first'
        }).round(3).to_dict(),
        
        'overall_stats': {
            'mean_correlation': results_df['avg_correlation'].mean(),
            'std_correlation': results_df['avg_correlation'].std(),
            'min_correlation': results_df['avg_correlation'].min(),
            'max_correlation': results_df['avg_correlation'].max(),
            'n_models': len(results_df['model'].unique()),
            'n_formats': len(results_df['format'].unique())
        }
    }
    
    # Analyze format differences
    format_comparison = analyze_format_differences(results_df)
    
    # Save summary files
    model_wise_file = output_dir / 'model_wise_stats.csv'
    results_df.groupby(['model', 'format'])['avg_correlation'].agg(['mean', 'std', 'count']).to_csv(model_wise_file)
    
    format_wise_file = output_dir / 'format_wise_stats.csv'
    results_df.groupby('format')['avg_correlation'].agg(['mean', 'std', 'count']).to_csv(format_wise_file)
    
    # Log comprehensive summary
    logger.info("\n" + "="*70)
    logger.info("STUDY 3 CONVERGENT VALIDITY ANALYSIS SUMMARY")
    logger.info("="*70)
    logger.info(f"Overall mean correlation: {summary_stats['overall_stats']['mean_correlation']:.3f}")
    logger.info(f"Standard deviation: {summary_stats['overall_stats']['std_correlation']:.3f}")
    logger.info(f"Range: {summary_stats['overall_stats']['min_correlation']:.3f} - {summary_stats['overall_stats']['max_correlation']:.3f}")
    logger.info(f"Number of models: {summary_stats['overall_stats']['n_models']}")
    logger.info(f"Number of formats: {summary_stats['overall_stats']['n_formats']}")
    
    # Log format comparison
    logger.info("\nFormat Comparison:")
    for format_name, stats in format_comparison.items():
        logger.info(f"  {format_name.capitalize()} format:")
        logger.info(f"    Mean correlation: {stats['mean_correlation']:.3f}")
        logger.info(f"    Std deviation: {stats['std_correlation']:.3f}")
        logger.info(f"    Range: {stats['min_correlation']:.3f} - {stats['max_correlation']:.3f}")
        logger.info(f"    Models tested: {stats['n_models']}")
    
    # Log individual model-format results
    logger.info("\nDetailed Results by Model and Format:")
    for _, row in results_df.iterrows():
        logger.info(f"  {row['model']} ({row['format']}): {row['avg_correlation']:.3f}")
    
    # Find best performing combinations
    best_overall = results_df.loc[results_df['avg_correlation'].idxmax()]
    logger.info(f"\nBest performing combination:")
    logger.info(f"  {best_overall['model']} ({best_overall['format']}): {best_overall['avg_correlation']:.3f}")
    
    # Compare formats if both are available
    if len(format_comparison) == 2:
        likert_mean = format_comparison.get('likert', {}).get('mean_correlation', 0)
        expanded_mean = format_comparison.get('expanded', {}).get('mean_correlation', 0)
        difference = expanded_mean - likert_mean
        logger.info(f"\nFormat Comparison:")
        logger.info(f"  Expanded format advantage: {difference:+.3f}")
        if abs(difference) > 0.01:
            better_format = "expanded" if difference > 0 else "likert"
            logger.info(f"  {better_format.capitalize()} format performs better")
        else:
            logger.info(f"  Formats perform similarly")
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  Detailed results: {results_file}")
    logger.info(f"  Model-wise stats: {model_wise_file}")
    logger.info(f"  Format-wise stats: {format_wise_file}")
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
