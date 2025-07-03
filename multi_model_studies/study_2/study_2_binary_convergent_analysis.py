#!/usr/bin/env python3
"""
Study 2: Binary Baseline Convergent Analysis

This script compares the binary baseline results with the existing expanded and likert format results.
It evaluates which approach provides the best convergent validity with human personality data.
"""

import pandas as pd
import numpy as np
import json
import glob
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# --- Mini-Marker to Big Five domain mapping (Saucier, 1994) ---
minimarker_domain_mapping = {
    'Bashful': 'E', 'Bold': 'E', 'Energetic': 'E', 'Extraverted': 'E', 'Quiet': 'E',
    'Shy': 'E', 'Talkative': 'E', 'Withdrawn': 'E',
    'Cold': 'A', 'Cooperative': 'A', 'Harsh': 'A', 'Kind': 'A', 'Rude': 'A',
    'Sympathetic': 'A', 'Unsympathetic': 'A', 'Warm': 'A',
    'Careless': 'C', 'Disorganized': 'C', 'Efficient': 'C', 'Inefficient': 'C',
    'Organized': 'C', 'Sloppy': 'C', 'Systematic': 'C',
    'Envious': 'N', 'Fretful': 'N', 'Jealous': 'N', 'Moody': 'N', 'Relaxed': 'N',
    'Temperamental': 'N', 'Touchy': 'N', 'Unenvious': 'N',
    'Complex': 'O', 'Creative': 'O', 'Deep': 'O', 'Imaginative': 'O',
    'Intellectual': 'O', 'Philosophical': 'O', 'Practical': 'O', 'Uncreative': 'O',
    'Unintellectual': 'O'
}

# Reverse-coded traits
reverse_coded_traits = {
    'Bashful', 'Quiet', 'Shy', 'Withdrawn',
    'Cold', 'Harsh', 'Rude', 'Unsympathetic',
    'Careless', 'Disorganized', 'Inefficient', 'Sloppy',
    'Envious', 'Fretful', 'Jealous', 'Moody', 'Temperamental', 'Touchy',
    'Practical', 'Uncreative', 'Unintellectual'
}

print("=== Binary Baseline Convergent Analysis ===")

# Load original data
data_path = Path('../../raw_data/Soto_data.xlsx')
if not data_path.exists():
    print(f"Error: Data file not found at {data_path}")
    exit(1)

data = pd.read_excel(data_path)
print(f"Loaded {len(data)} participants")

def aggregate_minimarker(df):
    """Aggregate Mini-Marker trait ratings to Big Five domain scores."""
    domain_scores = {d: [] for d in ['E', 'A', 'C', 'N', 'O']}
    
    for idx, row in df.iterrows():
        trait_by_domain = {d: [] for d in ['E', 'A', 'C', 'N', 'O']}
        
        for trait, value in row.items():
            if trait not in minimarker_domain_mapping:
                continue
            domain = minimarker_domain_mapping[trait]
            
            if isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    continue
            
            if trait in reverse_coded_traits:
                value = 10 - value
            trait_by_domain[domain].append(value)
        
        for d in trait_by_domain:
            if trait_by_domain[d]:
                domain_scores[d].append(sum(trait_by_domain[d]))
            else:
                domain_scores[d].append(np.nan)
    
    return pd.DataFrame(domain_scores)

def calculate_convergent_validity(emp_bfi2, emp_tda, sim_tda, format_name):
    """Calculate convergent validity metrics."""
    bfi2_sim_correlations = []
    tda_sim_correlations = []
    
    for i, domain in enumerate(['E', 'A', 'C', 'N', 'O']):
        if not emp_bfi2.iloc[:, i].isna().all() and not sim_tda[domain].isna().all():
            corr, _ = pearsonr(emp_bfi2.iloc[:, i], sim_tda[domain])
            bfi2_sim_correlations.append(corr)
        else:
            bfi2_sim_correlations.append(np.nan)
        
        if not emp_tda.iloc[:, i].isna().all() and not sim_tda[domain].isna().all():
            corr, _ = pearsonr(emp_tda.iloc[:, i], sim_tda[domain])
            tda_sim_correlations.append(corr)
        else:
            tda_sim_correlations.append(np.nan)
    
    return {
        'format': format_name,
        'mean_bfi2_correlation': np.nanmean(bfi2_sim_correlations),
        'mean_tda_correlation': np.nanmean(tda_sim_correlations),
        'n_participants': len(emp_bfi2)
    }

# Process formats
formats_to_analyze = [
    {'name': 'Binary Baseline', 'results_dir': 'study_2_binary_results', 'file_pattern': 'bfi_to_minimarker_binary_*.json'},
    {'name': 'Expanded Format', 'results_dir': 'study_2_results_I_am', 'file_pattern': 'bfi_to_minimarker_*.json'},
    {'name': 'Likert Format', 'results_dir': 'study_2_likert_results_separate', 'file_pattern': 'bfi_to_minimarker_*.json'}
]

all_format_results = {}
bfi2_cols = ['bfi2_e', 'bfi2_a', 'bfi2_c', 'bfi2_n', 'bfi2_o']
tda_cols = ['tda_e', 'tda_a', 'tda_c', 'tda_n', 'tda_o']

for format_info in formats_to_analyze:
    format_name = format_info['name']
    results_dir = Path(__file__).parent / format_info['results_dir']
    
    print(f"\nProcessing {format_name}...")
    
    if not results_dir.exists():
        print(f"  Warning: Results directory not found. Skipping.")
        continue
    
    json_files = glob.glob(str(results_dir / format_info['file_pattern']))
    
    if len(json_files) == 0:
        print(f"  Warning: No JSON files found. Skipping.")
        continue
    
    print(f"  Found {len(json_files)} model files")
    format_results = {}
    
    for json_file in json_files:
        model_name = Path(json_file).stem.replace('bfi_to_minimarker_', '').replace('_temp1_0', '').replace('binary_', '')
        
        try:
            with open(json_file, 'r') as f:
                sim_json = json.load(f)
            
            # Clean keys
            cleaned_sim_json = []
            for item in sim_json:
                cleaned_item = {}
                for k, v in item.items():
                    k_clean = k.lstrip()
                    if '. ' in k_clean:
                        k_clean = k_clean.split('. ', 1)[1]
                    cleaned_item[k_clean] = v
                cleaned_sim_json.append(cleaned_item)
            
            sim_df = pd.DataFrame(cleaned_sim_json)
            sim_domains = aggregate_minimarker(sim_df)
            
            # Align data
            n = min(len(data), len(sim_domains))
            emp_bfi2 = data.loc[:n-1, bfi2_cols].reset_index(drop=True)
            emp_tda = data.loc[:n-1, tda_cols].reset_index(drop=True)
            sim_tda = sim_domains.loc[:n-1, ['E','A','C','N','O']].reset_index(drop=True)
            
            # Remove invalid data
            valid_mask = ~(sim_tda.isna().any(axis=1) | np.isinf(sim_tda.values).any(axis=1))
            if not valid_mask.all():
                emp_bfi2 = emp_bfi2[valid_mask].reset_index(drop=True)
                emp_tda = emp_tda[valid_mask].reset_index(drop=True)
                sim_tda = sim_tda[valid_mask].reset_index(drop=True)
            
            validity_results = calculate_convergent_validity(emp_bfi2, emp_tda, sim_tda, format_name)
            validity_results['model'] = model_name
            format_results[model_name] = validity_results
            
            print(f"    {model_name}: BFI-2 r={validity_results['mean_bfi2_correlation']:.3f}, TDA r={validity_results['mean_tda_correlation']:.3f}")
            
        except Exception as e:
            print(f"    Error with {model_name}: {str(e)}")
            
    all_format_results[format_name] = format_results

# Create summary
print("\n" + "="*60)
print("CONVERGENT VALIDITY COMPARISON")
print("="*60)

summary_data = []
for format_name, format_results in all_format_results.items():
    for model_name, results in format_results.items():
        summary_data.append({
            'Format': format_name,
            'Model': model_name,
            'BFI-2_Correlation': results['mean_bfi2_correlation'],
            'TDA_Correlation': results['mean_tda_correlation'],
            'N_Participants': results['n_participants']
        })

if summary_data:
    summary_df = pd.DataFrame(summary_data)
    print("\nDetailed Results:")
    print(summary_df.round(3).to_string(index=False))
    
    # Format averages
    format_averages = summary_df.groupby('Format').agg({
        'BFI-2_Correlation': ['mean', 'std'],
        'TDA_Correlation': ['mean', 'std']
    }).round(3)
    
    print("\nFormat Averages:")
    for format_name in format_averages.index:
        print(f"{format_name}:")
        print(f"  BFI-2: {format_averages.loc[format_name, ('BFI-2_Correlation', 'mean')]:.3f} ± {format_averages.loc[format_name, ('BFI-2_Correlation', 'std')]:.3f}")
        print(f"  TDA:   {format_averages.loc[format_name, ('TDA_Correlation', 'mean')]:.3f} ± {format_averages.loc[format_name, ('TDA_Correlation', 'std')]:.3f}")
    
    # Save results
    output_dir = Path(__file__).parent / 'format_comparison_results'
    output_dir.mkdir(exist_ok=True)
    summary_df.to_csv(output_dir / 'convergent_validity_comparison.csv', index=False)
    print(f"\nResults saved to: {output_dir}")

print("\nAnalysis complete!")
