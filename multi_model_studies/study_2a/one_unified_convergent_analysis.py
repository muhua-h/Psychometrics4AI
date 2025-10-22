#!/usr/bin/env python3
"""
Adapted Unified Convergent Analysis for Study 2a (DeepSeek Likert only)

This script analyzes convergent validity for Likert format with DeepSeek model.
It compares simulated Mini-Marker results with empirical BFI-2 data.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr
import warnings
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# Set up logging to capture all print output
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- Mini-Marker to Big Five domain mapping (Saucier, 1994) ---
minimarker_domain_mapping = {
    # Extraversion (E)
    'Bashful': 'E', 'Bold': 'E', 'Energetic': 'E', 'Extraverted': 'E',
    'Quiet': 'E', 'Shy': 'E', 'Talkative': 'E', 'Withdrawn': 'E',

    # Agreeableness (A)
    'Cold': 'A', 'Cooperative': 'A', 'Harsh': 'A', 'Kind': 'A',
    'Rude': 'A', 'Sympathetic': 'A', 'Unsympathetic': 'A', 'Warm': 'A',

    # Conscientiousness (C)
    'Careless': 'C', 'Disorganized': 'C', 'Efficient': 'C', 'Inefficient': 'C',
    'Organized': 'C', 'Practical': 'C', 'Sloppy': 'C', 'Systematic': 'C',

    # Neuroticism (N)
    'Envious': 'N', 'Fretful': 'N', 'Jealous': 'N', 'Moody': 'N',
    'Relaxed': 'N', 'Temperamental': 'N', 'Touchy': 'N', 'Unenvious': 'N',

    # Openness (O)
    'Complex': 'O', 'Deep': 'O', 'Creative': 'O', 'Imaginative': 'O',
    'Intellectual': 'O', 'Philosophical': 'O', 'Uncreative': 'O', 'Unintellectual': 'O'
}

# Items that need reverse coding (higher scores indicate LOWER levels of the trait)
reverse_coded_traits = {
    'Bashful', 'Quiet', 'Shy', 'Withdrawn',  # Extraversion (reverse)
    'Cold', 'Harsh', 'Rude', 'Unsympathetic',  # Agreeableness (reverse)
    'Careless', 'Disorganized', 'Inefficient', 'Sloppy',  # Conscientiousness (reverse)
    'Relaxed', 'Unenvious',  # Neuroticism (reverse for emotional stability)
    'Uncreative', 'Unintellectual'  # Openness (reverse)
}

def load_empirical_data():
    """Load empirical data for Likert format."""
    data_path = Path('test_study_2_likert_results') / 'study2_likert_preprocessed_data.csv'
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    return pd.read_csv(data_path)

def aggregate_minimarker(df):
    """Aggregate Mini-Marker trait ratings to Big Five domain scores."""
    domain_scores = {d: [] for d in ['E', 'A', 'C', 'N', 'O']}

    for idx, row in df.iterrows():
        trait_by_domain = {d: [] for d in ['E', 'A', 'C', 'N', 'O']}

        for trait, value in row.items():
            if trait not in minimarker_domain_mapping:
                continue

            # Convert string values to integers if needed
            try:
                value = float(value)
            except ValueError:
                continue

            domain = minimarker_domain_mapping[trait]
            if trait in reverse_coded_traits:
                value = 10 - value  # Reverse coding (assuming 1-9 scale as in Mini-Markers)

            trait_by_domain[domain].append(value)

        for domain in domain_scores:
            if trait_by_domain[domain]:
                domain_scores[domain].append(np.mean(trait_by_domain[domain]))
            else:
                domain_scores[domain].append(np.nan)

    return pd.DataFrame(domain_scores)

def load_simulation_results():
    """Load simulation results for DeepSeek Likert."""
    sim_path = Path('test_study_2_likert_results') / 'bfi_to_minimarker_deepseek_temp1.json'
    if not sim_path.exists():
        raise FileNotFoundError(f"Simulation file not found at {sim_path}")
    with open(sim_path) as f:
        return pd.DataFrame(json.load(f))

def get_bfi2_domain_scores(empirical_data):
    """Extract BFI-2 domain scores from empirical data."""
    domain_scores = {
        'E': empirical_data['bfi2_e'].values,
        'A': empirical_data['bfi2_a'].values,
        'C': empirical_data['bfi2_c'].values,
        'N': empirical_data['bfi2_n'].values,
        'O': empirical_data['bfi2_o'].values
    }
    return pd.DataFrame(domain_scores)

def main():
    output_dir = Path('test_study_2_likert_results/unified_analysis_results')
    output_dir.mkdir(exist_ok=True)
    
    log_filename = output_dir / 'unified_analysis_log.txt'
    original_stdout = sys.stdout
    sys.stdout = Logger(log_filename)
    
    print(f"Analysis for DeepSeek Likert")
    print(f"Timestamp: {datetime.now()}")
    
    empirical_data = load_empirical_data()
    simulated_data = load_simulation_results()
    simulated_domain_scores = aggregate_minimarker(simulated_data)
    bfi2_domain_scores = get_bfi2_domain_scores(empirical_data)
    
    domain_order = ['O', 'C', 'E', 'A', 'N']
    
    correlations = {}
    for domain in domain_order:
        bfi_scores = bfi2_domain_scores[domain]
        sim_scores = simulated_domain_scores[domain]
        r, p = pearsonr(bfi_scores, sim_scores)
        correlations[domain] = (r, p)
    
    print("Correlations:")
    print("  Domain  BFI2_vs_Simulated_r  BFI2_vs_Simulated_p")
    for domain, (r, p) in correlations.items():
        print(f"     {domain}             {r:.6f}          {p:.3e}")
    
    avg_bfi_sim = np.mean([r for r, p in correlations.values()])
    
    print(f"\nAverage BFI-Sim correlation: {avg_bfi_sim:.3f}")
    
    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Log file: {log_filename.name}")
    
    print("\nNote: All correlations are Pearson r values")
    print("BFI-Sim: BFI-2 vs Simulated Mini-Marker (Simulation Validity)")
    
    # Restore original stdout
    sys.stdout = original_stdout
    print(f"\nAnalysis completed! Log saved to: {log_filename}")
    
    # Save results to CSV
    csv_results = []
    for domain, (r, p) in correlations.items():
        csv_results.append({
            'Model': 'deepseek',
            'Format': 'likert',
            'Domain': domain,
            'BFI2_vs_Simulated_r': r,
            'BFI2_vs_Simulated_p': p
        })
    csv_df = pd.DataFrame(csv_results)
    csv_path = output_dir / 'unified_convergent_results.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to CSV: {csv_path}")
    
    # Generate stats tables
    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include='number').columns
    
    # Model-wise stats
    mean_df = df.groupby('Model')[num_cols].mean().add_suffix('_mean')
    std_df = df.groupby('Model')[num_cols].std().add_suffix('_std')
    count_df = df.groupby('Model')[num_cols].count().add_suffix('_count')
    model_stats = pd.concat([mean_df, std_df, count_df], axis=1)
    model_stats_path = output_dir / 'model_wise_stats.csv'
    model_stats.to_csv(model_stats_path)
    print(f"Model-wise table saved: {model_stats_path}")
    
    # Condition-wise stats (using Domain as condition proxy)
    mean_df = df.groupby('Domain')[num_cols].mean().add_suffix('_mean')
    std_df = df.groupby('Domain')[num_cols].std().add_suffix('_std')
    count_df = df.groupby('Domain')[num_cols].count().add_suffix('_count')
    condition_stats = pd.concat([mean_df, std_df, count_df], axis=1)
    condition_stats_path = output_dir / 'condition_wise_stats.csv'
    condition_stats.to_csv(condition_stats_path)
    print(f"Condition-wise table saved: {condition_stats_path}")
    
    # Model+Condition-wise stats
    mean_df = df.groupby(['Model', 'Domain'])[num_cols].mean().add_suffix('_mean')
    std_df = df.groupby(['Model', 'Domain'])[num_cols].std().add_suffix('_std')
    count_df = df.groupby(['Model', 'Domain'])[num_cols].count().add_suffix('_count')
    model_condition_stats = pd.concat([mean_df, std_df, count_df], axis=1)
    model_condition_stats_path = output_dir / 'model_condition_stats.csv'
    model_condition_stats.to_csv(model_condition_stats_path)
    print(f"Model+Condition-wise table saved: {model_condition_stats_path}")

if __name__ == "__main__":
    main()