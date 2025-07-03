#!/usr/bin/env python3
"""
Diagnostic script to detect where participants are being lost during convergent analysis.
This script traces through the same steps as the convergent analysis to identify
where participants are dropped.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import glob

# Add shared modules to path
import sys
sys.path.append('../shared')

print("="*80)
print("PARTICIPANT LOSS DETECTION SCRIPT")
print("="*80)

# --- 1. Load empirical data ---
data_path = Path('study_2_likert_results') / 'study2_likert_preprocessed_data.csv'
if not data_path.exists():
    print(f"Error: Preprocessed data not found at {data_path}")
    exit(1)

data = pd.read_csv(data_path)
print(f"1. Loaded empirical data: {data.shape[0]} participants")

# --- 2. Load simulation results ---
results_dir = Path('study_2_likert_results')
json_files = glob.glob(str(results_dir / 'bfi_to_minimarker_*.json'))

if len(json_files) == 0:
    print(f"Error: No simulation results found in {results_dir}")
    exit(1)

print(f"\n2. Found {len(json_files)} simulation files:")
for f in json_files:
    print(f"   - {Path(f).name}")

# --- 3. Define aggregation function (same as convergent analysis) ---
minimarker_domain_mapping = {
    # Extraversion (E)
    'Bashful': 'E', 'Bold': 'E', 'Energetic': 'E', 'Extraverted': 'E',
    'Quiet': 'E', 'Shy': 'E', 'Talkative': 'E', 'Withdrawn': 'E',
    # Agreeableness (A)
    'Cold': 'A', 'Cooperative': 'A', 'Envious': 'A', 'Harsh': 'A',
    'Jealous': 'A', 'Kind': 'A', 'Rude': 'A', 'Sympathetic': 'A', 
    'Unsympathetic': 'A', 'Warm': 'A',
    # Conscientiousness (C)
    'Careless': 'C', 'Disorganized': 'C', 'Efficient': 'C', 'Inefficient': 'C',
    'Organized': 'C', 'Practical': 'C', 'Sloppy': 'C', 'Systematic': 'C',
    # Neuroticism (N)
    'Fretful': 'N', 'Moody': 'N', 'Relaxed': 'N', 'Temperamental': 'N', 'Touchy': 'N',
    # Openness (O)
    'Complex': 'O', 'Creative': 'O', 'Deep': 'O', 'Imaginative': 'O',
    'Intellectual': 'O', 'Philosophical': 'O', 'Uncreative': 'O', 
    'Unenvious': 'O', 'Unintellectual': 'O'
}

reverse_coded_traits = {
    'Bashful', 'Quiet', 'Shy', 'Withdrawn',  # Extraversion (reverse)
    'Cold', 'Harsh', 'Rude', 'Unsympathetic',  # Agreeableness (reverse)
    'Careless', 'Disorganized', 'Inefficient', 'Sloppy',  # Conscientiousness (reverse)
    'Relaxed',  # Neuroticism (reverse)
    'Uncreative', 'Unintellectual'  # Openness (reverse)
}

def aggregate_minimarker(df):
    """Same aggregation function as convergent analysis"""
    domain_scores = {d: [] for d in ['E', 'A', 'C', 'N', 'O']}
    
    print(f"   Aggregating {len(df)} participants")
    
    parsing_errors = 0
    missing_traits = 0
    
    for idx, row in df.iterrows():
        trait_by_domain = {d: [] for d in ['E', 'A', 'C', 'N', 'O']}
        
        for trait, value in row.items():
            if trait not in minimarker_domain_mapping:
                continue
            domain = minimarker_domain_mapping[trait]

            # Cast value to int if it's a string
            if isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    print(f"   Warning: Non-integer value '{value}' for trait '{trait}' at participant {idx}")
                    parsing_errors += 1
                    continue
            
            # Check for NaN values
            if pd.isna(value):
                print(f"   Warning: NaN value for trait '{trait}' at participant {idx}")
                missing_traits += 1
                continue
                
            # Reverse code if needed
            if trait in reverse_coded_traits:
                value = 10 - value
            trait_by_domain[domain].append(value)
        
        # Aggregate by domain
        for d in trait_by_domain:
            if trait_by_domain[d]:
                domain_scores[d].append(sum(trait_by_domain[d]))
            else:
                domain_scores[d].append(np.nan)
    
    print(f"   Parsing errors: {parsing_errors}")
    print(f"   Missing trait values: {missing_traits}")
    
    return pd.DataFrame(domain_scores)

# --- 4. Process each model and trace participant loss ---
print("\n" + "="*80)
print("TRACING PARTICIPANT LOSS BY MODEL")
print("="*80)

for json_file in json_files:
    model_name = Path(json_file).stem.replace('bfi_to_minimarker_', '').replace('_temp1_0', '')
    
    print(f"\n--- Processing {model_name} ---")
    
    # Load JSON data
    with open(json_file, 'r') as f:
        sim_json = json.load(f)
    
    print(f"3a. Loaded simulation JSON: {len(sim_json)} participants")
    
    # Clean keys (same as convergent analysis)
    cleaned_sim_json = []
    cleaning_issues = 0
    
    for i, item in enumerate(sim_json):
        cleaned_item = {}
        for k, v in item.items():
            k_clean = k.lstrip()  # Remove leading whitespace/newlines
            if '. ' in k_clean:
                k_clean = k_clean.split('. ', 1)[1]
            cleaned_item[k_clean] = v
        
        # Check if all expected traits are present
        expected_traits = set(minimarker_domain_mapping.keys())
        found_traits = set(cleaned_item.keys())
        missing_traits = expected_traits - found_traits
        
        if missing_traits:
            print(f"   Participant {i}: Missing traits {missing_traits}")
            cleaning_issues += 1
        
        cleaned_sim_json.append(cleaned_item)
    
    print(f"3b. After key cleaning: {len(cleaned_sim_json)} participants")
    print(f"    Cleaning issues: {cleaning_issues}")
    
    # Convert to DataFrame and aggregate
    try:
        sim_df = pd.DataFrame(cleaned_sim_json)
        print(f"3c. Created DataFrame: {sim_df.shape[0]} participants")
        
        # Check for completely empty rows
        empty_rows = sim_df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            print(f"    Warning: {empty_rows} completely empty rows")
        
        # Aggregate to domain scores
        sim_domains = aggregate_minimarker(sim_df)
        print(f"3d. After aggregation: {sim_domains.shape[0]} participants")
        
    except Exception as e:
        print(f"   Error creating DataFrame or aggregating: {e}")
        continue
    
    # Data alignment (same as convergent analysis)
    original_n = min(len(data), len(sim_domains))
    print(f"4a. After alignment (min length): {original_n} participants")
    
    # Extract relevant columns
    bfi2_cols = ['bfi2_e', 'bfi2_a', 'bfi2_c', 'bfi2_n', 'bfi2_o']
    tda_cols = ['tda_e', 'tda_a', 'tda_c', 'tda_n', 'tda_o']
    
    emp_bfi2 = data.loc[:original_n-1, bfi2_cols].reset_index(drop=True)
    emp_tda = data.loc[:original_n-1, tda_cols].reset_index(drop=True)
    sim_tda = sim_domains.loc[:original_n-1, ['E','A','C','N','O']].reset_index(drop=True)
    
    # Check for invalid values (same as convergent analysis)
    print(f"4b. Data quality check:")
    print(f"    Simulated data NaN count: {sim_tda.isna().sum().sum()}")
    print(f"    Simulated data inf count: {np.isinf(sim_tda.values).sum()}")
    
    # Show which participants have invalid data
    nan_rows = sim_tda.isna().any(axis=1)
    inf_rows = np.isinf(sim_tda.values).any(axis=1)
    invalid_rows = nan_rows | inf_rows
    
    if invalid_rows.any():
        invalid_indices = sim_tda.index[invalid_rows].tolist()
        print(f"    Participants with invalid data: {invalid_indices}")
        print(f"    Invalid data breakdown:")
        for idx in invalid_indices[:5]:  # Show first 5
            row_data = sim_tda.iloc[idx]
            issues = []
            for col in row_data.index:
                if pd.isna(row_data[col]):
                    issues.append(f"{col}=NaN")
                elif np.isinf(row_data[col]):
                    issues.append(f"{col}=inf")
            print(f"      Participant {idx}: {', '.join(issues)}")
    
    # Apply the same filtering as convergent analysis
    valid_mask = ~(sim_tda.isna().any(axis=1) | np.isinf(sim_tda.values).any(axis=1))
    final_n = sum(valid_mask)
    
    print(f"4c. After removing invalid data: {final_n} participants")
    print(f"    Removed {original_n - final_n} participants due to invalid simulated data")
    
    # Summary for this model
    print(f"\nSUMMARY FOR {model_name.upper()}:")
    print(f"  Started with: {len(sim_json)} participants")
    print(f"  After alignment: {original_n} participants")
    print(f"  Final count: {final_n} participants")
    print(f"  Total lost: {len(sim_json) - final_n} participants")
    
    # Detailed loss breakdown
    loss_reasons = []
    if len(sim_json) > len(sim_domains):
        loss_reasons.append(f"Aggregation issues: {len(sim_json) - len(sim_domains)}")
    if len(sim_domains) > original_n:
        loss_reasons.append(f"Data alignment: {len(sim_domains) - original_n}")
    if original_n > final_n:
        loss_reasons.append(f"Invalid data filtering: {original_n - final_n}")
    
    if loss_reasons:
        print(f"  Loss breakdown: {'; '.join(loss_reasons)}")

print("\n" + "="*80)
print("DETECTION COMPLETE")
print("="*80)
print("\nTo fix participant count mismatches:")
print("1. Check simulation JSON files for parsing errors")
print("2. Investigate missing trait values in simulated data")
print("3. Consider handling NaN values differently in aggregation")
print("4. Verify that all expected Mini-Marker traits are present")
print("5. Check for any data type conversion issues") 