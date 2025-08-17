#!/usr/bin/env python3
"""
Unified Binary Dichotomized Analysis for Study 2 and Study 3

This script analyzes the correlation between dichotomized BFI-2 scores and Mini-Marker output
for both Study 2 (empirical data) and Study 3 (simulated data) across binary simple and binary elaborated formats.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr, pointbiserialr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_study2_data():
    """Load the original Study 2 empirical data."""
    data_path = Path(__file__).parent / '..' / 'study_2' / 'shared_data' / 'study2_preprocessed_data.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Study 2 data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Study 2 has pre-computed BFI-2 domain scores
    df['bfi2_e'] = df['bfi2_e']
    df['bfi2_a'] = df['bfi2_a']
    df['bfi2_c'] = df['bfi2_c']
    df['bfi2_n'] = df['bfi2_n']
    df['bfi2_o'] = df['bfi2_o']
    
    logger.info(f"Loaded Study 2 empirical data: {df.shape}")
    return df

def load_study3_data():
    """Load the original Study 3 simulated data."""
    data_path = Path(__file__).parent / '..' / 'study_3' / 'facet_lvl_simulated_data.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Study 3 data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Map to expected column names
    df['bfi2_e'] = df['bfi_e']
    df['bfi2_a'] = df['bfi_a']
    df['bfi2_c'] = df['bfi_c']
    df['bfi2_n'] = df['bfi_n']
    df['bfi2_o'] = df['bfi_o']
    
    logger.info(f"Loaded Study 3 simulated data: {df.shape}")
    return df


def dichotomize_scores(df, cutoff_value=2.5):
    """Dichotomize continuous BFI-2 scores using a fixed cutoff (default=2.5)."""
    df_dichotomized = df.copy()

    # Define BFI-2 domains
    bfi_domains = ['e', 'a', 'c', 'n', 'o']

    for domain in bfi_domains:
        bfi_col = f'bfi2_{domain}'

        # Always use the fixed cutoff
        cutoff = cutoff_value

        # Create dichotomized version (0 = low, 1 = high)
        df_dichotomized[f'{bfi_col}_dichotomized'] = (
                    df[bfi_col] >= cutoff).astype(int)

        logger.info(f"Dichotomized {domain}: cutoff={cutoff:.3f}, "
                    f"low={sum(df_dichotomized[f'{bfi_col}_dichotomized'] == 0)}, "
                    f"high={sum(df_dichotomized[f'{bfi_col}_dichotomized'] == 1)}")

    return df_dichotomized

def aggregate_minimarker_responses(df, format_type='binary'):
    """Aggregate Mini-Marker trait ratings to Big Five domain scores."""
    def reverse_score(score):
        return 10 - score

    def calculate_big_five_scores(df):
        # Exact mapping from original studies
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

def load_simulation_results(results_dir, format_type, study_name):
    """Load Mini-Marker simulation results for a specific format."""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
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
                    logger.info(f"Loaded {len(responses)} responses for {model_name} ({format_type} format, {study_name})")
                else:
                    logger.warning(f"No valid responses found in {json_file}")
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
    
    return simulation_results

def calculate_dichotomized_correlations(dichotomized_data, simulation_results, format_type, study_name, cutoff_type='median'):
    """Calculate correlations between dichotomized BFI-2 scores and Mini-Marker output."""
    results = []
    
    for model_name, sim_df in simulation_results.items():
        logger.info(f"Analyzing {model_name} ({format_type} format, {study_name})...")
        
        # Ensure we have the same number of participants
        n_participants = min(len(dichotomized_data), len(sim_df))
        data_subset = dichotomized_data.head(n_participants).copy()
        sim_subset = sim_df.head(n_participants).copy()
        
        # Aggregate Mini-Marker responses
        sim_aggregated = aggregate_minimarker_responses(sim_subset, format_type)
        
        # Calculate correlations between dichotomized BFI-2 and Mini-Marker
        domain_correlations = {}
        
        bfi_minimarker_pairs = [
            (f'bfi2_e_dichotomized', 'miniMarker_simulated_E'),
            (f'bfi2_a_dichotomized', 'miniMarker_simulated_A'),
            (f'bfi2_c_dichotomized', 'miniMarker_simulated_C'),
            (f'bfi2_n_dichotomized', 'miniMarker_simulated_N'),
            (f'bfi2_o_dichotomized', 'miniMarker_simulated_O')
        ]
        
        for bfi_col, mm_col in bfi_minimarker_pairs:
            if bfi_col in data_subset.columns and mm_col in sim_aggregated.columns:
                # Use point-biserial correlation for dichotomous vs continuous
                corr, p_value = pointbiserialr(data_subset[bfi_col], sim_aggregated[mm_col])
                domain = bfi_col.split('_')[1].upper()
                domain_correlations[f'{domain}_correlation'] = corr
                domain_correlations[f'{domain}_p_value'] = p_value
                logger.info(f"  {domain}: r = {corr:.3f}, p = {p_value:.3f}")
        
        # Calculate average correlation
        correlations = [v for k, v in domain_correlations.items() if k.endswith('_correlation')]
        avg_correlation = np.mean(correlations) if correlations else 0
        
        result = {
            'study': study_name,
            'model': model_name,
            'format': format_type,
            'cutoff_type': cutoff_type,
            'n_participants': n_participants,
            'avg_correlation': avg_correlation,
            **domain_correlations
        }
        
        results.append(result)
        logger.info(f"  Average correlation: {avg_correlation:.3f}")
    
    return results

def analyze_study_differences(results_df):
    """Analyze differences between studies."""
    study_stats = {}
    
    for study_name in results_df['study'].unique():
        study_data = results_df[results_df['study'] == study_name]
        if not study_data.empty:
            study_stats[study_name] = {
                'mean_correlation': study_data['avg_correlation'].mean(),
                'std_correlation': study_data['avg_correlation'].std(),
                'min_correlation': study_data['avg_correlation'].min(),
                'max_correlation': study_data['avg_correlation'].max(),
                'n_models': len(study_data['model'].unique()),
                'n_formats': len(study_data['format'].unique())
            }
    
    return study_stats

def analyze_format_differences(results_df):
    """Analyze differences between binary formats."""
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
    logger.info("Starting unified binary dichotomized analysis for Study 2 and Study 3...")
    
    all_results = []
    cutoff_type = 2.5
    
    # Define study configurations
    study_configs = [
        {
            'name': 'study_2',
            'data_loader': load_study2_data,
            'binary_simple_dir': Path(__file__).parent / '..' / 'study_2' / 'study_2_simple_binary_results',
            'binary_elaborated_dir': Path(__file__).parent / '..' / 'study_2' / 'study_2_elaborated_binary_results'
        },
        {
            'name': 'study_3',
            'data_loader': load_study3_data,
            'binary_simple_dir': Path(__file__).parent / '..' / 'study_3' / 'study_3_binary_simple_results',
            'binary_elaborated_dir': Path(__file__).parent / '..' / 'study_3' / 'study_3_binary_elaborated_results'
        }
    ]
    
    # Process each study
    for study_config in study_configs:
        study_name = study_config['name']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING {study_name.upper()}")
        logger.info(f"{'='*60}")
        
        try:
            # Load data for this study
            original_data = study_config['data_loader']()
            
            # Dichotomize BFI-2 scores
            dichotomized_data = dichotomize_scores(original_data, cutoff_type)
            
            # Define format configurations for this study
            format_configs = [
                {
                    'name': 'binary_simple',
                    'results_dir': study_config['binary_simple_dir']
                },
                {
                    'name': 'binary_elaborated',
                    'results_dir': study_config['binary_elaborated_dir']
                }
            ]
            
            # Analyze each format
            for format_config in format_configs:
                format_name = format_config['name']
                results_dir = format_config['results_dir']
                
                logger.info(f"\n{'='*50}")
                logger.info(f"ANALYZING {study_name.upper()} - {format_name.upper()}")
                logger.info(f"{'='*50}")
                
                # Load simulation results for this format
                simulation_results = load_simulation_results(results_dir, format_name, study_name)
                
                if simulation_results:
                    # Calculate correlations with dichotomized scores
                    format_results = calculate_dichotomized_correlations(
                        dichotomized_data, simulation_results, format_name, study_name, cutoff_type
                    )
                    all_results.extend(format_results)
                    
                    logger.info(f"Completed analysis for {study_name} {format_name}: {len(format_results)} model results")
                else:
                    logger.warning(f"No simulation results found for {study_name} {format_name}")
        
        except FileNotFoundError as e:
            logger.error(f"Could not load {study_name} data: {e}")
            continue
    
    if not all_results:
        logger.error("No results to analyze")
        return
    
    # Save results
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(all_results)
    results_file = output_dir / f'unified_binary_dichotomized_correlations_{cutoff_type}.csv'
    results_df.to_csv(results_file, index=False)
    
    # Generate summary statistics
    summary_stats = {
        'study_wise_stats': results_df.groupby(['study', 'format']).agg({
            'avg_correlation': ['mean', 'std', 'min', 'max'],
            'n_participants': 'first'
        }).round(3).to_dict(),
        
        'model_wise_stats': results_df.groupby(['study', 'model', 'format'])['avg_correlation'].agg(['mean', 'std', 'count']).to_csv(
            output_dir / f'model_wise_stats_{cutoff_type}.csv'
        ),
        
        'format_wise_stats': results_df.groupby(['study', 'format']).agg({
            'avg_correlation': ['mean', 'std', 'min', 'max'],
            'n_participants': 'first'
        }).round(3).to_dict(),
        
        'overall_stats': {
            'mean_correlation': results_df['avg_correlation'].mean(),
            'std_correlation': results_df['avg_correlation'].std(),
            'min_correlation': results_df['avg_correlation'].min(),
            'max_correlation': results_df['avg_correlation'].max(),
            'n_models': len(results_df['model'].unique()),
            'n_studies': len(results_df['study'].unique()),
            'n_formats': len(results_df['format'].unique()),
            'cutoff_type': cutoff_type
        }
    }
    
    # Analyze differences
    study_comparison = analyze_study_differences(results_df)
    format_comparison = analyze_format_differences(results_df)
    
    # Save summary files
    study_wise_file = output_dir / f'study_wise_stats_{cutoff_type}.csv'
    results_df.groupby(['study', 'format'])['avg_correlation'].agg(['mean', 'std', 'count']).to_csv(study_wise_file)
    
    # Log comprehensive summary
    logger.info("\n" + "="*80)
    logger.info("UNIFIED BINARY DICHOTOMIZED ANALYSIS SUMMARY")
    logger.info("="*80)
    logger.info(f"Cutoff type: {cutoff_type}")
    logger.info(f"Overall mean correlation: {summary_stats['overall_stats']['mean_correlation']:.3f}")
    logger.info(f"Standard deviation: {summary_stats['overall_stats']['std_correlation']:.3f}")
    logger.info(f"Range: {summary_stats['overall_stats']['min_correlation']:.3f} - {summary_stats['overall_stats']['max_correlation']:.3f}")
    logger.info(f"Number of studies: {summary_stats['overall_stats']['n_studies']}")
    logger.info(f"Number of models: {summary_stats['overall_stats']['n_models']}")
    logger.info(f"Number of formats: {summary_stats['overall_stats']['n_formats']}")
    
    # Log study comparison
    logger.info("\nStudy Comparison:")
    for study_name, stats in study_comparison.items():
        logger.info(f"  {study_name}:")
        logger.info(f"    Mean correlation: {stats['mean_correlation']:.3f}")
        logger.info(f"    Std deviation: {stats['std_correlation']:.3f}")
        logger.info(f"    Range: {stats['min_correlation']:.3f} - {stats['max_correlation']:.3f}")
        logger.info(f"    Models tested: {stats['n_models']}")
    
    # Log format comparison
    logger.info("\nFormat Comparison:")
    for format_name, stats in format_comparison.items():
        logger.info(f"  {format_name}:")
        logger.info(f"    Mean correlation: {stats['mean_correlation']:.3f}")
        logger.info(f"    Std deviation: {stats['std_correlation']:.3f}")
        logger.info(f"    Range: {stats['min_correlation']:.3f} - {stats['max_correlation']:.3f}")
        logger.info(f"    Models tested: {stats['n_models']}")
    
    # Log individual results
    logger.info("\nDetailed Results by Study, Model, and Format:")
    for _, row in results_df.iterrows():
        logger.info(f"  {row['study']} - {row['model']} ({row['format']}): {row['avg_correlation']:.3f}")
    
    # Find best performing combinations
    best_overall = results_df.loc[results_df['avg_correlation'].idxmax()]
    logger.info(f"\nBest performing combination:")
    logger.info(f"  {best_overall['study']} - {best_overall['model']} ({best_overall['format']}): {best_overall['avg_correlation']:.3f}")
    
    # Compare studies
    if len(study_comparison) == 2:
        study2_mean = study_comparison.get('study_2', {}).get('mean_correlation', 0)
        study3_mean = study_comparison.get('study_3', {}).get('mean_correlation', 0)
        difference = study3_mean - study2_mean
        logger.info(f"\nStudy Comparison:")
        logger.info(f"  Study 3 vs Study 2 difference: {difference:+.3f}")
        if abs(difference) > 0.01:
            better_study = "study_3" if difference > 0 else "study_2"
            logger.info(f"  {better_study} performs better")
        else:
            logger.info(f"  Studies perform similarly")
    
    # Compare binary formats
    if len(format_comparison) == 2:
        simple_mean = format_comparison.get('binary_simple', {}).get('mean_correlation', 0)
        elaborated_mean = format_comparison.get('binary_elaborated', {}).get('mean_correlation', 0)
        difference = elaborated_mean - simple_mean
        logger.info(f"\nBinary Format Comparison:")
        logger.info(f"  Elaborated vs Simple difference: {difference:+.3f}")
        if abs(difference) > 0.01:
            better_format = "elaborated" if difference > 0 else "simple"
            logger.info(f"  {better_format.capitalize()} format performs better")
        else:
            logger.info(f"  Formats perform similarly")
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  Detailed results: {results_file}")
    logger.info(f"  Study-wise stats: {study_wise_file}")
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()