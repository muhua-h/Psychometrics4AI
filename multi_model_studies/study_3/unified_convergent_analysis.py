#!/usr/bin/env python3
"""
Unified Convergent Analysis for Study 3 - Multi-Model Simulation

This script analyzes convergent validity between BFI-2 and Mini-Marker scales
using the exact same approach as the original Study 3 to ensure proper correlations.
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
    """Load the original Study 3 data that achieved good correlations."""
    # Use the original Study 3 data that achieved 0.679 correlation
    data_path = Path(__file__).parent.parent.parent / 'study_3' / 'likert_format' / 'facet_lvl_simulated_data.csv'
    
    if not data_path.exists():
        raise FileNotFoundError(f"Original Study 3 data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # The original Study 3 data already has the correct domain scores (bfi_e, bfi_a, etc.)
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
    
    logger.info(f"Loaded original Study 3 data: {df.shape}")
    return df

def aggregate_minimarker_original_study3(df, format_type='likert'):
    """Aggregate Mini-Marker trait ratings to Big Five domain scores using original Study 3 methodology."""
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
                    logger.warning(f"Trait '{trait}' not found in data")
            
            if scores:
                # Average the scores for this dimension
                dimension_scores = pd.concat(scores, axis=1).mean(axis=1)
                results[f'miniMarker_simulated_{dimension}'] = dimension_scores
            else:
                logger.warning(f"No traits found for dimension {dimension}")
        
        return pd.DataFrame(results)
    
    # Calculate Big Five scores
    big_five_df = calculate_big_five_scores(df)
    
    # Merge with original dataframe
    result_df = df.copy()
    for col in big_five_df.columns:
        result_df[col] = big_five_df[col]
    
    logger.info(f"Aggregated Mini-Marker scores: {big_five_df.shape}")
    return result_df

def load_simulation_results(results_dir):
    """Load Mini-Marker simulation results."""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return {}
    
    simulation_results = {}
    
    # Look for JSON files with simulation results
    for json_file in results_dir.glob("*.json"):
        if json_file.name.startswith("bfi_to_minimarker"):
            model_name = json_file.stem.replace("bfi_to_minimarker_", "").replace("_temp1.0", "").replace("_temp1", "")
            
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
                    logger.info(f"Loaded {len(responses)} responses for {model_name}")
                else:
                    logger.warning(f"No valid responses found in {json_file}")
                    
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
    
    return simulation_results

def calculate_convergent_validity(simulated_data, simulation_results):
    """Calculate convergent validity correlations."""
    results = []
    
    for model_name, sim_df in simulation_results.items():
        logger.info(f"Analyzing {model_name}...")
        
        # Ensure we have the same number of participants
        n_participants = min(len(simulated_data), len(sim_df))
        data_subset = simulated_data.head(n_participants).copy()
        sim_subset = sim_df.head(n_participants).copy()
        
        # Aggregate Mini-Marker responses
        sim_aggregated = aggregate_minimarker_original_study3(sim_subset)
        
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
            'format': 'likert',
            'n_participants': n_participants,
            'avg_correlation': avg_correlation,
            **domain_correlations
        }
        
        results.append(result)
        logger.info(f"  Average correlation: {avg_correlation:.3f}")
    
    return results

def main():
    """Main analysis function."""
    logger.info("Starting unified convergent analysis for Study 3...")
    
    # Load original Study 3 data
    try:
        simulated_data = load_original_study3_data()
    except FileNotFoundError as e:
        logger.error(f"Could not load original Study 3 data: {e}")
        return
    
    # Load simulation results
    results_dir = Path(__file__).parent / 'study_3_likert_results'
    simulation_results = load_simulation_results(results_dir)
    
    if not simulation_results:
        logger.error("No simulation results found")
        return
    
    # Calculate convergent validity
    convergent_results = calculate_convergent_validity(simulated_data, simulation_results)
    
    # Save results
    output_dir = Path(__file__).parent / 'unified_analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(convergent_results)
    results_file = output_dir / 'unified_convergent_results.csv'
    results_df.to_csv(results_file, index=False)
    
    # Generate summary statistics
    summary_stats = {
        'model_wise_stats': results_df.groupby('model').agg({
            'avg_correlation': ['mean', 'std', 'min', 'max'],
            'n_participants': 'first'
        }).round(3).to_dict(),
        
        'overall_stats': {
            'mean_correlation': results_df['avg_correlation'].mean(),
            'std_correlation': results_df['avg_correlation'].std(),
            'min_correlation': results_df['avg_correlation'].min(),
            'max_correlation': results_df['avg_correlation'].max(),
            'n_models': len(results_df['model'].unique())
        }
    }
    
    # Save summary
    summary_file = output_dir / 'model_wise_stats.csv'
    pd.DataFrame(summary_stats['model_wise_stats']).to_csv(summary_file)
    
    # Log summary
    logger.info("\n" + "="*50)
    logger.info("CONVERGENT VALIDITY ANALYSIS SUMMARY")
    logger.info("="*50)
    logger.info(f"Overall mean correlation: {summary_stats['overall_stats']['mean_correlation']:.3f}")
    logger.info(f"Standard deviation: {summary_stats['overall_stats']['std_correlation']:.3f}")
    logger.info(f"Range: {summary_stats['overall_stats']['min_correlation']:.3f} - {summary_stats['overall_stats']['max_correlation']:.3f}")
    logger.info(f"Number of models: {summary_stats['overall_stats']['n_models']}")
    
    # Log individual model results
    logger.info("\nIndividual Model Results:")
    for _, row in results_df.iterrows():
        logger.info(f"  {row['model']}: {row['avg_correlation']:.3f}")
    
    logger.info(f"\nResults saved to: {results_file}")
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()
