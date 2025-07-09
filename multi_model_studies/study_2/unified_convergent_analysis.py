#!/usr/bin/env python3
"""
Unified Convergent Analysis for Study 2

This script analyzes convergent validity across different formats:
- Binary Baseline
- Expanded Format 
- Likert Format

It compares simulated Mini-Marker results with empirical BFI-2 and TDA data,
providing detailed domain-level and aggregate results.
"""

import pandas as pd
import numpy as np
import json
import glob
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
    'Quiet': 'E',
    'Shy': 'E', 'Talkative': 'E', 'Withdrawn': 'E',
    # Agreeableness (A) 
    'Cold': 'A', 'Cooperative': 'A', 'Envious': 'A', 'Harsh': 'A',
    'Jealous': 'A',
    'Kind': 'A', 'Rude': 'A', 'Sympathetic': 'A', 'Unsympathetic': 'A',
    'Warm': 'A',
    # Conscientiousness (C)
    'Careless': 'C', 'Disorganized': 'C', 'Efficient': 'C', 'Inefficient': 'C',
    'Organized': 'C', 'Practical': 'C', 'Sloppy': 'C', 'Systematic': 'C',
    # Neuroticism (N)
    'Fretful': 'N', 'Moody': 'N', 'Relaxed': 'N', 'Temperamental': 'N',
    'Touchy': 'N',
    # Openness (O)
    'Complex': 'O', 'Creative': 'O', 'Deep': 'O', 'Imaginative': 'O',
    'Intellectual': 'O',
    'Philosophical': 'O', 'Uncreative': 'O', 'Unenvious': 'O',
    'Unintellectual': 'O'
}

# Items that need reverse coding (higher scores indicate LOWER levels of the trait)
reverse_coded_traits = {
    'Bashful', 'Quiet', 'Shy', 'Withdrawn',  # Extraversion (reverse)
    'Cold', 'Envious', 'Harsh', 'Jealous', 'Rude', 'Unsympathetic',
    # Agreeableness (reverse)
    'Careless', 'Disorganized', 'Inefficient', 'Sloppy',
    # Conscientiousness (reverse)
    'Relaxed', 'Unenvious', # Neuroticism (reverse for emotional stability)
    'Uncreative', 'Unintellectual'  # Openness (reverse)
}


def load_empirical_data(format_type):
    """Load empirical data based on format type."""
    if format_type == 'binary_simple':
        data_path = Path(__file__).parent / 'study_2_simple_binary_results' / 'study2_preprocessed_data.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
        return pd.read_csv(data_path)
    elif format_type == 'binary_elaborated':
        data_path = Path(__file__).parent / 'study_2_elaborated_binary_results' / 'study2_preprocessed_data.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
        return pd.read_csv(data_path)
    elif format_type == 'expanded_you_are':
        data_path = Path(__file__).parent / 'study_2_expanded_results_you_are' / 'study2_preprocessed_data.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
        return pd.read_csv(data_path)
    elif format_type == 'expanded_i_am':
        data_path = Path(__file__).parent / 'study_2_expanded_results_i_am' / 'study2_preprocessed_data.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
        return pd.read_csv(data_path)
    elif format_type == 'likert':
        data_path = Path(__file__).parent / 'study_2_likert_results' / 'study2_likert_preprocessed_data.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found at {data_path}")
        return pd.read_csv(data_path)
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def aggregate_minimarker(df, format_type='expanded'):
    """Aggregate Mini-Marker trait ratings to Big Five domain scores."""
    domain_scores = {d: [] for d in ['E', 'A', 'C', 'N', 'O']}

    for idx, row in df.iterrows():
        trait_by_domain = {d: [] for d in ['E', 'A', 'C', 'N', 'O']}

        for trait, value in row.items():
            if trait not in minimarker_domain_mapping:
                continue

            # Convert string values to integers
            if isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    print(
                        f"Warning: Non-integer value '{value}' for trait '{trait}' at index {idx}. Skipping.")
                    continue

            domain = minimarker_domain_mapping[trait]

            # Apply reverse coding (assuming 1-9 scale)
            if trait in reverse_coded_traits:
                value = 10 - value

            trait_by_domain[domain].append(value)

        # Aggregate domain scores
        for d in trait_by_domain:
            if trait_by_domain[d]:
                if format_type == 'likert':
                    # Use sum for Likert format
                    domain_scores[d].append(sum(trait_by_domain[d]))
                else:
                    # Use mean for binary and expanded formats
                    domain_scores[d].append(np.mean(trait_by_domain[d]))
            else:
                domain_scores[d].append(np.nan)

    return pd.DataFrame(domain_scores)


def compute_correlations(arr1, arr2, traits=['E', 'A', 'C', 'N', 'O']):
    """Compute correlations between two arrays with error handling."""
    corrs = []
    for i, trait in enumerate(traits):
        try:
            # Extract data
            x = arr1.iloc[:, i].values
            y = arr2.iloc[:, i].values

            # Remove NaN/inf values
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


def analyze_format(format_config, csv_results=None):
    """Analyze convergent validity for a specific format."""
    format_name = format_config['name']
    format_type = format_config['type']
    results_dir = Path(__file__).parent / format_config['results_dir']
    file_pattern = format_config['file_pattern']

    print(f"\n{'=' * 60}")
    print(f"ANALYZING {format_name.upper()}")
    print(f"{'=' * 60}")

    # Load empirical data
    try:
        data = load_empirical_data(format_type)
        print(f"Loaded {len(data)} participants")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None

    # Find simulation files
    if not results_dir.exists():
        print(f"Warning: Results directory not found at {results_dir}")
        return None

    json_files = glob.glob(str(results_dir / file_pattern))
    if len(json_files) == 0:
        print(f"Warning: No JSON files found matching pattern {file_pattern}")
        return None

    print(f"Found {len(json_files)} model files")

    # Define column names
    bfi2_cols = ['bfi2_e', 'bfi2_a', 'bfi2_c', 'bfi2_n', 'bfi2_o']
    tda_cols = ['tda_e', 'tda_a', 'tda_c', 'tda_n', 'tda_o']

    format_results = {}

    # Process each model
    for json_file in json_files:
        model_name = Path(json_file).stem.replace('bfi_to_minimarker_',
                                                  '').replace('_temp1_0',
                                                              '').replace(
            'binary_', '')

        print(f"\n--- Processing {model_name} ---")

        try:
            # Load JSON data
            with open(json_file, 'r') as f:
                sim_json = json.load(f)

            # Clean keys (remove leading whitespace and index numbers)
            cleaned_sim_json = []
            for item in sim_json:
                cleaned_item = {}
                for k, v in item.items():
                    k_clean = k.lstrip()
                    if '. ' in k_clean:
                        k_clean = k_clean.split('. ', 1)[1]
                    cleaned_item[k_clean] = v
                cleaned_sim_json.append(cleaned_item)

            # Convert to DataFrame and aggregate
            sim_df = pd.DataFrame(cleaned_sim_json)
            sim_domains = aggregate_minimarker(sim_df, format_type)

            # Align data (truncate to shortest length)
            n = min(len(data), len(sim_domains))
            emp_bfi2 = data.loc[:n - 1, bfi2_cols].reset_index(drop=True)
            emp_tda = data.loc[:n - 1, tda_cols].reset_index(drop=True)
            sim_tda = sim_domains.loc[:n - 1,
                      ['E', 'A', 'C', 'N', 'O']].reset_index(drop=True)

            # Clean invalid data
            valid_mask = ~(sim_tda.isna().any(axis=1) | np.isinf(
                sim_tda.values).any(axis=1))
            if not valid_mask.all():
                print(f"  Removing {sum(~valid_mask)} rows with invalid data")
                emp_bfi2 = emp_bfi2[valid_mask].reset_index(drop=True)
                emp_tda = emp_tda[valid_mask].reset_index(drop=True)
                sim_tda = sim_tda[valid_mask].reset_index(drop=True)
                n = len(emp_bfi2)

            # Compute correlations
            bfi_orig_corrs = compute_correlations(emp_bfi2, emp_tda)
            bfi_sim_corrs = compute_correlations(emp_bfi2, sim_tda)
            orig_sim_corrs = compute_correlations(emp_tda, sim_tda)

            # Store results
            format_results[model_name] = {
                'bfi_orig_by_domain': bfi_orig_corrs,
                'bfi_sim_by_domain': bfi_sim_corrs,
                'orig_sim_by_domain': orig_sim_corrs,
                'bfi_orig_avg': np.nanmean(bfi_orig_corrs),
                'bfi_sim_avg': np.nanmean(bfi_sim_corrs),
                'orig_sim_avg': np.nanmean(orig_sim_corrs),
                'n_participants': n
            }

            # Save to csv_results if provided
            if csv_results is not None:
                csv_results.append({
                    'condition': format_name,
                    'model': model_name,
                    'bfi_orig_E': bfi_orig_corrs[0],
                    'bfi_orig_A': bfi_orig_corrs[1],
                    'bfi_orig_C': bfi_orig_corrs[2],
                    'bfi_orig_N': bfi_orig_corrs[3],
                    'bfi_orig_O': bfi_orig_corrs[4],
                    'bfi_sim_E': bfi_sim_corrs[0],
                    'bfi_sim_A': bfi_sim_corrs[1],
                    'bfi_sim_C': bfi_sim_corrs[2],
                    'bfi_sim_N': bfi_sim_corrs[3],
                    'bfi_sim_O': bfi_sim_corrs[4],
                    'orig_sim_E': orig_sim_corrs[0],
                    'orig_sim_A': orig_sim_corrs[1],
                    'orig_sim_C': orig_sim_corrs[2],
                    'orig_sim_N': orig_sim_corrs[3],
                    'orig_sim_O': orig_sim_corrs[4],
                    'bfi_orig_avg': np.nanmean(bfi_orig_corrs),
                    'bfi_sim_avg': np.nanmean(bfi_sim_corrs),
                    'orig_sim_avg': np.nanmean(orig_sim_corrs),
                    'n_participants': n
                })

            print(f"  Final participants: {n}")
            print(f"  BFI-2 vs Original avg: {np.nanmean(bfi_orig_corrs):.3f}")
            print(f"  BFI-2 vs Simulated avg: {np.nanmean(bfi_sim_corrs):.3f}")
            print(
                f"  Original vs Simulated avg: {np.nanmean(orig_sim_corrs):.3f}")

        except Exception as e:
            print(f"  Error processing {model_name}: {str(e)}")
            continue

    return format_results


def print_detailed_results(format_results, format_name):
    """Print detailed results for a format with Format → Model → Domain structure."""
    print(f"\n{'=' * 80}")
    print(f"DETAILED CONVERGENT VALIDITY RESULTS - {format_name.upper()}")
    print(f"{'=' * 80}")

    traits = ['E', 'A', 'C', 'N', 'O']
    trait_names = ['Extraversion', 'Agreeableness', 'Conscientiousness',
                   'Neuroticism', 'Openness']

    for model_name, results in format_results.items():
        print(f"\n{'-' * 80}")
        print(f"MODEL: {model_name.upper()} (n={results['n_participants']})")
        print(f"{'-' * 80}")

        # Print domain-by-domain analysis
        print(
            f"\n{'Domain':<20} {'BFI-Orig':<12} {'BFI-Sim':<12} {'Orig-Sim':<12}")
        print("-" * 60)

        for i, (trait, trait_name) in enumerate(zip(traits, trait_names)):
            bfi_orig_r = results['bfi_orig_by_domain'][i]
            bfi_sim_r = results['bfi_sim_by_domain'][i]
            orig_sim_r = results['orig_sim_by_domain'][i]

            bfi_orig_str = f"{bfi_orig_r:.3f}" if not np.isnan(
                bfi_orig_r) else "NaN"
            bfi_sim_str = f"{bfi_sim_r:.3f}" if not np.isnan(
                bfi_sim_r) else "NaN"
            orig_sim_str = f"{orig_sim_r:.3f}" if not np.isnan(
                orig_sim_r) else "NaN"

            print(
                f"{trait_name:<20} {bfi_orig_str:<12} {bfi_sim_str:<12} {orig_sim_str:<12}")

        # Print averages
        print("-" * 60)
        print(
            f"{'AVERAGE':<20} {results['bfi_orig_avg']:<12.3f} {results['bfi_sim_avg']:<12.3f} {results['orig_sim_avg']:<12.3f}")

        # Print interpretation
        print(f"\nInterpretation:")
        print(
            f"  • BFI-Orig: Empirical baseline correlation (BFI-2 vs Original Mini-Marker)")
        print(
            f"  • BFI-Sim: Simulation validity (BFI-2 vs Simulated Mini-Marker)")
        print(
            f"  • Orig-Sim: Direct comparison (Original vs Simulated Mini-Marker)")

        # Highlight best and worst performing domains
        if not all(np.isnan(results['bfi_sim_by_domain'])):
            best_domain_idx = np.nanargmax(results['bfi_sim_by_domain'])
            worst_domain_idx = np.nanargmin(results['bfi_sim_by_domain'])

            print(
                f"\n  Best performing domain: {trait_names[best_domain_idx]} (r = {results['bfi_sim_by_domain'][best_domain_idx]:.3f})")
            print(
                f"  Worst performing domain: {trait_names[worst_domain_idx]} (r = {results['bfi_sim_by_domain'][worst_domain_idx]:.3f})")





def main():
    """Main analysis function."""
    # Set up logging to capture all output
    output_dir = Path(__file__).parent / 'unified_analysis_results'
    output_dir.mkdir(exist_ok=True)
    
    # Create timestamp for log file
    log_filename = output_dir / f'unified_analysis_log.txt'
    
    # Redirect stdout to both terminal and log file
    original_stdout = sys.stdout
    sys.stdout = Logger(str(log_filename))
    
    print("=== UNIFIED CONVERGENT VALIDITY ANALYSIS ===")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_filename}")
    print("=" * 60)

    # Define format configurations
    format_configs = [
        {
            'name': 'Simple Binary',
            'type': 'binary_simple',
            'results_dir': 'study_2_simple_binary_results',
            'file_pattern': 'bfi_to_minimarker_binary_*.json'
        },
        {
            'name': 'Elaborated Binary',
            'type': 'binary_elaborated',
            'results_dir': 'study_2_elaborated_binary_results',
            'file_pattern': 'bfi_to_minimarker_binary_*.json'
        },
        {
            'name': 'Expanded (You Are)',
            'type': 'expanded_you_are',
            'results_dir': 'study_2_expanded_results_you_are',
            'file_pattern': 'bfi_to_minimarker_*.json'
        },
        {
            'name': 'Expanded (I Am)',
            'type': 'expanded_i_am',
            'results_dir': 'study_2_expanded_results_i_am',
            'file_pattern': 'bfi_to_minimarker_*.json'
        },
        {
            'name': 'Likert Format',
            'type': 'likert',
            'results_dir': 'study_2_likert_results',
            'file_pattern': 'bfi_to_minimarker_*.json'
        }
    ]

    # Prepare to collect results for CSV
    csv_results = []

    # Analyze each format
    all_results = {}
    for config in format_configs:
        results = analyze_format(config, csv_results=csv_results)
        if results:
            all_results[config['name']] = results

    # Print detailed results for each format
    for format_name, format_results in all_results.items():
        print_detailed_results(format_results, format_name)

    if not all_results:
        print("No valid results found for any format.")
        return

    # Print format comparison summary
    print(f"\n{'=' * 80}")
    print("CROSS-FORMAT COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    traits = ['E', 'A', 'C', 'N', 'O']
    trait_names = ['Extraversion', 'Agreeableness', 'Conscientiousness',
                   'Neuroticism', 'Openness']

    # Collect all results for summary
    all_summary_data = []
    for format_name, format_results in all_results.items():
        for model_name, results in format_results.items():
            all_summary_data.append({
                'Format': format_name,
                'Model': model_name,
                'BFI2_vs_Original_Avg': results['bfi_orig_avg'],
                'BFI2_vs_Simulated_Avg': results['bfi_sim_avg'],
                'Original_vs_Simulated_Avg': results['orig_sim_avg']
            })

    # Organize by format first
    for format_name in set(item['Format'] for item in all_summary_data):
        format_data = [item for item in all_summary_data if item['Format'] == format_name]

        print(f"\n{'-' * 80}")
        print(f"FORMAT: {format_name.upper()}")
        print(f"{'-' * 80}")

        # Show models within format
        print(
            f"\n{'Model':<25} {'BFI-Orig':<12} {'BFI-Sim':<12} {'Orig-Sim':<12}")
        print("-" * 65)

        for item in format_data:
            print(
                f"{item['Model']:<25} {item['BFI2_vs_Original_Avg']:<12.3f} {item['BFI2_vs_Simulated_Avg']:<12.3f} {item['Original_vs_Simulated_Avg']:<12.3f}")

        # Format averages
        bfi_orig_avg = np.mean([item['BFI2_vs_Original_Avg'] for item in format_data])
        bfi_sim_avg = np.mean([item['BFI2_vs_Simulated_Avg'] for item in format_data])
        orig_sim_avg = np.mean([item['Original_vs_Simulated_Avg'] for item in format_data])
        
        bfi_orig_std = np.std([item['BFI2_vs_Original_Avg'] for item in format_data])
        bfi_sim_std = np.std([item['BFI2_vs_Simulated_Avg'] for item in format_data])
        orig_sim_std = np.std([item['Original_vs_Simulated_Avg'] for item in format_data])

        print("-" * 65)
        print(
            f"{'FORMAT AVERAGE':<25} {bfi_orig_avg:<12.3f} {bfi_sim_avg:<12.3f} {orig_sim_avg:<12.3f}")
        print(
            f"{'FORMAT STD':<25} {bfi_orig_std:<12.3f} {bfi_sim_std:<12.3f} {orig_sim_std:<12.3f}")

    # Overall format comparison
    print(f"\n{'=' * 80}")
    print("OVERALL FORMAT COMPARISON")
    print(f"{'=' * 80}")

    print(f"\n{'Format':<20} {'BFI-Orig':<12} {'BFI-Sim':<12} {'Orig-Sim':<12}")
    print("-" * 60)

    for format_name in set(item['Format'] for item in all_summary_data):
        format_data = [item for item in all_summary_data if item['Format'] == format_name]
        avg_bfi_orig = np.mean([item['BFI2_vs_Original_Avg'] for item in format_data])
        avg_bfi_sim = np.mean([item['BFI2_vs_Simulated_Avg'] for item in format_data])
        avg_orig_sim = np.mean([item['Original_vs_Simulated_Avg'] for item in format_data])

        print(
            f"{format_name:<20} {avg_bfi_orig:<12.3f} {avg_bfi_sim:<12.3f} {avg_orig_sim:<12.3f}")

    # Best performing format and models
    print(f"\n{'=' * 60}")
    print("PERFORMANCE HIGHLIGHTS")
    print(f"{'=' * 60}")

    best_item = max(all_summary_data, key=lambda x: x['BFI2_vs_Simulated_Avg'])
    worst_item = min(all_summary_data, key=lambda x: x['BFI2_vs_Simulated_Avg'])

    print(f"\nBest performing combination:")
    print(f"  Format: {best_item['Format']}")
    print(f"  Model: {best_item['Model']}")
    print(f"  BFI-Sim correlation: {best_item['BFI2_vs_Simulated_Avg']:.3f}")

    print(f"\nWorst performing combination:")
    print(f"  Format: {worst_item['Format']}")
    print(f"  Model: {worst_item['Model']}")
    print(f"  BFI-Sim correlation: {worst_item['BFI2_vs_Simulated_Avg']:.3f}")

    # Format rankings
    format_rankings = {}
    for format_name in set(item['Format'] for item in all_summary_data):
        format_data = [item for item in all_summary_data if item['Format'] == format_name]
        avg_score = np.mean([item['BFI2_vs_Simulated_Avg'] for item in format_data])
        format_rankings[format_name] = avg_score
    
    sorted_rankings = sorted(format_rankings.items(), key=lambda x: x[1], reverse=True)
    print(f"\nFormat rankings by BFI-Sim correlation:")
    for i, (format_name, score) in enumerate(sorted_rankings, 1):
        print(f"  {i}. {format_name}: {score:.3f}")

    print(f"\n{'=' * 60}")
    print("ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"Log file: {log_filename.name}")

    print("\nNote: All correlations are Pearson r values")
    print("BFI-Orig: BFI-2 vs Original Mini-Marker (Empirical Baseline)")
    print("BFI-Sim: BFI-2 vs Simulated Mini-Marker (Simulation Validity)")
    print("Orig-Sim: Original vs Simulated Mini-Marker (Direct Comparison)")
    
    # Restore original stdout
    sys.stdout = original_stdout
    print(f"\nAnalysis completed! Log saved to: {log_filename}")

    # Save results to CSV
    csv_df = pd.DataFrame(csv_results)
    csv_path = output_dir / 'unified_convergent_results.csv'
    csv_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to CSV: {csv_path}")

    # === Additional: Generate model/condition stats tables (from final_anlaysis.py) ===
    # Load the just-saved results
    df = pd.read_csv(csv_path)
    num_cols = df.select_dtypes(include='number').columns

    # Model-wise stats
    mean_df = df.groupby('model')[num_cols].mean().add_suffix('_mean')
    std_df = df.groupby('model')[num_cols].std().add_suffix('_std')
    count_df = df.groupby('model')[num_cols].count().add_suffix('_count')
    model_stats = pd.concat([mean_df, std_df, count_df], axis=1)
    model_stats_path = output_dir.parent / 'model_wise_stats.csv'
    model_stats.to_csv(model_stats_path)
    print(f"Model-wise table saved: {model_stats_path}")

    # Condition-wise stats
    mean_df = df.groupby('condition')[num_cols].mean().add_suffix('_mean')
    std_df = df.groupby('condition')[num_cols].std().add_suffix('_std')
    count_df = df.groupby('condition')[num_cols].count().add_suffix('_count')
    condition_stats = pd.concat([mean_df, std_df, count_df], axis=1)
    condition_stats_path = output_dir.parent / 'condition_wise_stats.csv'
    condition_stats.to_csv(condition_stats_path)
    print(f"Condition-wise table saved: {condition_stats_path}")

    # Model+Condition-wise stats
    mean_df = df.groupby(['model', 'condition'])[num_cols].mean().add_suffix('_mean')
    std_df = df.groupby(['model', 'condition'])[num_cols].std().add_suffix('_std')
    count_df = df.groupby(['model', 'condition'])[num_cols].count().add_suffix('_count')
    model_condition_stats = pd.concat([mean_df, std_df, count_df], axis=1)
    model_condition_stats_path = output_dir.parent / 'model_condition_stats.csv'
    model_condition_stats.to_csv(model_condition_stats_path)
    print(f"Model+Condition-wise table saved: {model_condition_stats_path}")


if __name__ == "__main__":
    main()
