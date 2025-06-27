import pandas as pd
import numpy as np
import json
from scipy.stats import pearsonr
from pathlib import Path
import glob

# --- Mini-Marker to Big Five domain mapping (Saucier, 1994) ---
minimarker_domain_mapping = {
    # Extraversion (E)
    'Bashful': 'E', 'Bold': 'E', 'Energetic': 'E', 'Extraverted': 'E', 'Quiet': 'E', 
    'Shy': 'E', 'Talkative': 'E', 'Withdrawn': 'E',
    # Agreeableness (A) 
    'Cold': 'A', 'Cooperative': 'A', 'Envious': 'A', 'Harsh': 'A', 'Jealous': 'A',
    'Kind': 'A', 'Rude': 'A', 'Sympathetic': 'A', 'Unsympathetic': 'A', 'Warm': 'A',
    # Conscientiousness (C)
    'Careless': 'C', 'Disorganized': 'C', 'Efficient': 'C', 'Inefficient': 'C',
    'Organized': 'C', 'Practical': 'C', 'Sloppy': 'C', 'Systematic': 'C',
    # Neuroticism (N)
    'Fretful': 'N', 'Moody': 'N', 'Relaxed': 'N', 'Temperamental': 'N', 'Touchy': 'N',
    # Openness (O)
    'Complex': 'O', 'Creative': 'O', 'Deep': 'O', 'Imaginative': 'O', 'Intellectual': 'O',
    'Philosophical': 'O', 'Uncreative': 'O', 'Unenvious': 'O', 'Unintellectual': 'O'
}

# Items that need reverse coding (higher scores indicate LOWER levels of the trait)
reverse_coded_traits = {
    'Bashful', 'Quiet', 'Shy', 'Withdrawn',  # Extraversion (reverse)
    'Cold', 'Envious', 'Harsh', 'Jealous', 'Rude', 'Unsympathetic',  # Agreeableness (reverse)
    'Careless', 'Disorganized', 'Inefficient', 'Sloppy',  # Conscientiousness (reverse)
    'Fretful', 'Moody', 'Temperamental', 'Touchy',  # Neuroticism (reverse for emotional stability)
    'Uncreative', 'Unintellectual'  # Openness (reverse)
}

# --- 1. Load empirical data ---
data_path = Path(__file__).parent / 'study_2_results' / 'study2_preprocessed_data.csv'
data = pd.read_csv(data_path)

# Get BFI-2 and Mini-Marker columns
bfi2_cols = ['bfi2_e', 'bfi2_a', 'bfi2_c', 'bfi2_n', 'bfi2_o']
tda_cols = ['tda_e', 'tda_a', 'tda_c', 'tda_n', 'tda_o']

# --- 2. Load all simulated Mini-Marker data (JSON files) ---
results_dir = Path(__file__).parent / 'study_2_results'
json_files = glob.glob(str(results_dir / 'bfi_to_minimarker_*.json'))

print(f"Found {len(json_files)} model files:")
for f in json_files:
    print(f"  - {Path(f).name}")

# --- 3. Aggregate simulated Mini-Marker to Big Five domains ---
def aggregate_minimarker(df):
    domain_scores = {d: [] for d in ['E', 'A', 'C', 'N', 'O']}
    for idx, row in df.iterrows():
        trait_by_domain = {d: [] for d in ['E', 'A', 'C', 'N', 'O']}
        for trait, value in row.items():
            if trait not in minimarker_domain_mapping:
                continue
            domain = minimarker_domain_mapping[trait]
            # Reverse code if needed (1-9 scale)
            if trait in reverse_coded_traits:
                value = 10 - value
            trait_by_domain[domain].append(value)
        for d in trait_by_domain:
            if trait_by_domain[d]:
                domain_scores[d].append(np.mean(trait_by_domain[d]))
            else:
                domain_scores[d].append(np.nan)
    return pd.DataFrame(domain_scores)

# --- 4. Process each model ---
model_results = {}

for json_file in json_files:
    model_name = Path(json_file).stem.replace('bfi_to_minimarker_', '').replace('_temp1_0', '')
    
    print(f"\n--- Processing {model_name} ---")
    
    # Load JSON data
    with open(json_file, 'r') as f:
        sim_json = json.load(f)

    # Clean keys: remove leading whitespace/newlines and index-dot-space (e.g., '\n32. Talkative' -> 'Talkative', '\nUnintellectual' -> 'Unintellectual')
    cleaned_sim_json = []
    for item in sim_json:
        cleaned_item = {}
        for k, v in item.items():
            k_clean = k.lstrip()  # Remove leading whitespace/newlines
            if '. ' in k_clean:
                k_clean = k_clean.split('. ', 1)[1]
            cleaned_item[k_clean] = v
        cleaned_sim_json.append(cleaned_item)

    # Convert to DataFrame and aggregate
    sim_df = pd.DataFrame(cleaned_sim_json)
    sim_domains = aggregate_minimarker(sim_df)
    
    # Align data (truncate to shortest length)
    n = min(len(data), len(sim_domains))
    emp_bfi2 = data.loc[:n-1, bfi2_cols].reset_index(drop=True)
    emp_tda = data.loc[:n-1, tda_cols].reset_index(drop=True)
    sim_tda = sim_domains.loc[:n-1, ['E','A','C','N','O']].reset_index(drop=True)
    
    # Check for NaN/inf values and clean data
    print(f"  Data shape: BFI-2={emp_bfi2.shape}, TDA={emp_tda.shape}, Sim={sim_tda.shape}")
    print(f"  Simulated data NaN count: {sim_tda.isna().sum().sum()}")
    print(f"  Simulated data inf count: {np.isinf(sim_tda.values).sum()}")
    
    # Remove rows with NaN/inf values
    valid_mask = ~(sim_tda.isna().any(axis=1) | np.isinf(sim_tda.values).any(axis=1))
    if not valid_mask.all():
        print(f"  Removing {sum(~valid_mask)} rows with invalid data")
        emp_bfi2 = emp_bfi2[valid_mask].reset_index(drop=True)
        emp_tda = emp_tda[valid_mask].reset_index(drop=True)
        sim_tda = sim_tda[valid_mask].reset_index(drop=True)
        n = len(emp_bfi2)
    
    # Compute correlations with error handling
    def compute_corrs(arr1, arr2):
        corrs = []
        for i, trait in enumerate(['E','A','C','N','O']):
            try:
                # Check for valid data
                x = arr1.iloc[:,i].values
                y = arr2.iloc[:,i].values
                
                # Remove any remaining NaN/inf values
                valid_idx = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
                if sum(valid_idx) < 3:  # Need at least 3 points for correlation
                    print(f"    Warning: {trait} has insufficient valid data")
                    corrs.append(np.nan)
                    continue
                
                x_clean = x[valid_idx]
                y_clean = y[valid_idx]
                
                r, _ = pearsonr(x_clean, y_clean)
                corrs.append(r)
            except Exception as e:
                print(f"    Error computing correlation for {trait}: {e}")
                corrs.append(np.nan)
        return corrs
    
    # 1. BFI-2 vs. original Mini-Marker
    bfi_orig_corrs = compute_corrs(emp_bfi2, emp_tda)
    # 2. BFI-2 vs. Simulated Mini-Marker  
    bfi_sim_corrs = compute_corrs(emp_bfi2, sim_tda)
    # 3. Original vs. Simulated Mini-Marker
    orig_sim_corrs = compute_corrs(emp_tda, sim_tda)
    
    model_results[model_name] = {
        'bfi_orig': bfi_orig_corrs,
        'bfi_sim': bfi_sim_corrs, 
        'orig_sim': orig_sim_corrs,
        'n_participants': n
    }
    
    print(f"  Final participants: {n}")
    print(f"  BFI-2 vs Original avg: {np.nanmean(bfi_orig_corrs):.3f}")
    print(f"  BFI-2 vs Simulated avg: {np.nanmean(bfi_sim_corrs):.3f}")
    print(f"  Original vs Simulated avg: {np.nanmean(orig_sim_corrs):.3f}")

    # Check for NaN values in the simulated data
    nan_mask = sim_tda.isna().any(axis=1)
    if nan_mask.any():
        print(f"  Rows with NaN in {model_name} simulated data:")
        print(sim_tda.loc[nan_mask, :])

# --- 5. Print detailed results for each model ---
print("\n" + "="*80)
print("DETAILED CONVERGENT VALIDITY RESULTS")
print("="*80)

for model_name, results in model_results.items():
    print(f"\n{model_name.upper()} MODEL (n={results['n_participants']})")
    print("-" * 50)
    
    traits = ['E', 'A', 'C', 'N', 'O']
    
    print("BFI-2 vs Original Mini-Marker:")
    for i, trait in enumerate(traits):
        r_val = results['bfi_orig'][i]
        if np.isnan(r_val):
            print(f"  {trait}: r = NaN")
        else:
            print(f"  {trait}: r = {r_val:.3f}")
    print(f"  Average: r = {np.nanmean(results['bfi_orig']):.3f}")
    
    print("\nBFI-2 vs Simulated Mini-Marker:")
    for i, trait in enumerate(traits):
        r_val = results['bfi_sim'][i]
        if np.isnan(r_val):
            print(f"  {trait}: r = NaN")
        else:
            print(f"  {trait}: r = {r_val:.3f}")
    print(f"  Average: r = {np.nanmean(results['bfi_sim']):.3f}")
    
    print("\nOriginal vs Simulated Mini-Marker:")
    for i, trait in enumerate(traits):
        r_val = results['orig_sim'][i]
        if np.isnan(r_val):
            print(f"  {trait}: r = NaN")
        else:
            print(f"  {trait}: r = {r_val:.3f}")
    print(f"  Average: r = {np.nanmean(results['orig_sim']):.3f}")

# --- 6. Model comparison summary ---
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

print(f"{'Model':<15} {'BFI-Orig':<10} {'BFI-Sim':<10} {'Orig-Sim':<10}")
print("-" * 50)

for model_name, results in model_results.items():
    bfi_orig_avg = np.nanmean(results['bfi_orig'])
    bfi_sim_avg = np.nanmean(results['bfi_sim'])
    orig_sim_avg = np.nanmean(results['orig_sim'])
    
    print(f"{model_name:<15} {bfi_orig_avg:<10.3f} {bfi_sim_avg:<10.3f} {orig_sim_avg:<10.3f}")

print("\nNote: All correlations are Pearson r values")
print("BFI-Orig: BFI-2 vs Original Mini-Marker")
print("BFI-Sim: BFI-2 vs Simulated Mini-Marker") 
print("Orig-Sim: Original vs Simulated Mini-Marker") 