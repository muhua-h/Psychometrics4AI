#!/usr/bin/env python3
"""
Unified Participant Recovery Script for Study 3

This script detects missing participants across all formats (Binary, Expanded, Likert)
and recovers them using the appropriate simulation models. It uses the same validation
rules as unified_convergent_analysis.py to identify problematic responses.

Usage:
    python recover_missing_participants.py [--dry-run] [--format FORMAT]
"""

import pandas as pd
import numpy as np
import json
import glob
from pathlib import Path
from typing import Dict, List, Any, Tuple
import argparse
import sys
from datetime import datetime

# Add shared modules to path
sys.path.append('../shared')
from simulation_utils import run_batch_simulation, SimulationConfig
from mini_marker_prompt import get_expanded_prompt, get_likert_prompt
from binary_baseline_prompt import get_binary_prompt, \
    generate_binary_personality_description

# Mini-Marker to Big Five domain mapping (same as unified_convergent_analysis.py)
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
    'Fretful': 'N', 'Moody': 'N', 'Relaxed': 'N', 'Temperamental': 'N',
    'Touchy': 'N',
    # Openness (O)
    'Complex': 'O', 'Creative': 'O', 'Deep': 'O', 'Imaginative': 'O',
    'Intellectual': 'O', 'Philosophical': 'O', 'Uncreative': 'O',
    'Unenvious': 'O', 'Unintellectual': 'O'
}

# Items that need reverse coding (same as unified_convergent_analysis.py)
reverse_coded_traits = {
    'Bashful', 'Quiet', 'Shy', 'Withdrawn',  # Extraversion (reverse)
    'Cold', 'Envious', 'Harsh', 'Jealous', 'Rude', 'Unsympathetic',
    # Agreeableness (reverse)
    'Careless', 'Disorganized', 'Inefficient', 'Sloppy',
    # Conscientiousness (reverse)
    'Relaxed', 'Unenvious', 'Uncreative', 'Unintellectual'
    # Neuroticism/Openness (reverse)
}


def aggregate_minimarker_for_validation(df, format_type='expanded'):
    """
    Aggregate Mini-Marker trait ratings to Big Five domain scores.
    Same logic as unified_convergent_analysis.py for validation.
    """
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
                    # Mark as invalid for this participant
                    continue

            if pd.isna(value):
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


def validate_response(response_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a single response using the same logic as unified_convergent_analysis.py
    """
    errors = []

    # Check for explicit error responses first (network failures, etc.)
    if isinstance(response_data, dict) and 'error' in response_data:
        errors.append(
            f"Error response: {response_data.get('error', 'Unknown error')}")
        return False, errors

    # Check if response has enough traits (should have ~40 traits)
    if isinstance(response_data, dict) and len(response_data) < 30:
        errors.append(f"Insufficient traits: only {len(response_data)} found")
        return False, errors

    # Clean the response keys first
    cleaned_response = {}
    for k, v in response_data.items():
        k_clean = k.lstrip()
        if '. ' in k_clean:
            k_clean = k_clean.split('. ', 1)[1]
        cleaned_response[k_clean] = v

    # Check for missing expected traits
    expected_traits = set(minimarker_domain_mapping.keys())
    found_traits = set(cleaned_response.keys())
    missing_traits = expected_traits - found_traits

    if missing_traits:
        errors.append(f"Missing traits: {sorted(list(missing_traits))}")

    # Check for invalid values
    for trait, value in cleaned_response.items():
        if trait in minimarker_domain_mapping:
            # Check if value can be converted to int
            if isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    errors.append(f"Invalid value for {trait}: {value}")
                    continue

            if pd.isna(value):
                errors.append(f"NaN value for {trait}")

    return len(errors) == 0, errors


def analyze_format_for_missing_participants(format_config: Dict[str, Any]) -> \
        Dict[str, Any]:
    """
    Analyze a format to identify missing participants.
    Returns analysis results including problematic indices.
    """
    format_name = format_config['name']
    format_type = format_config['type']
    results_dir = Path(__file__).parent / format_config['results_dir']
    file_pattern = format_config['file_pattern']

    print(f"\n{'=' * 60}")
    print(f"ANALYZING {format_name.upper()} FOR MISSING PARTICIPANTS")
    print(f"{'=' * 60}")

    # Load Study 3 simulated data
    data_path = Path(__file__).parent / 'facet_lvl_simulated_data.csv'

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return None

    data = pd.read_csv(data_path)
    expected_participants = len(data)
    print(f"Expected participants: {expected_participants}")

    # Find simulation files
    if not results_dir.exists():
        print(f"Warning: Results directory not found at {results_dir}")
        return None

    json_files = glob.glob(str(results_dir / file_pattern))
    if len(json_files) == 0:
        print(f"Warning: No JSON files found matching pattern {file_pattern}")
        return None

    print(f"Found {len(json_files)} model files")

    format_analysis = {}

    # Process each model
    for json_file in json_files:
        # Skip backup files
        if '.backup.' in json_file:
            continue

        file_stem = Path(json_file).stem
        model_name = file_stem.replace('bfi_to_minimarker_', '').replace(
            'binary_', '')

        print(f"\n--- Analyzing {model_name} ({Path(json_file).name}) ---")

        try:
            # Load JSON data
            with open(json_file, 'r') as f:
                sim_json = json.load(f)

            total_responses = len(sim_json)
            print(f"Total responses: {total_responses}")

            # Find error responses (like the simple_recover.py approach)
            error_indices = []
            valid_responses = 0

            for idx, response in enumerate(sim_json):
                # Check for explicit error responses first (network failures, etc.)
                if isinstance(response, dict) and 'error' in response:
                    error_indices.append(idx)
                else:
                    # Then check for validation issues
                    is_valid, errors = validate_response(response)
                    if is_valid:
                        valid_responses += 1
                    else:
                        error_indices.append(idx)

            print(f"Valid responses: {valid_responses}")
            print(f"Error/problematic responses: {len(error_indices)}")

            # Calculate final participants after aggregation
            final_n = valid_responses  # Simplified - use valid responses as final count
            missing_participants = expected_participants - final_n

            print(f"After aggregation and filtering: {final_n} participants")
            print(f"Missing participants: {missing_participants}")

            # Store analysis
            format_analysis[model_name] = {
                'file_path': json_file,
                'total_responses': total_responses,
                'valid_responses': valid_responses,
                'problematic_indices': error_indices,
                'final_participants': final_n,
                'missing_participants': missing_participants,
                'data_path': data_path
            }

        except Exception as e:
            print(f"Error analyzing {model_name}: {str(e)}")
            continue

    return format_analysis


def create_prompt_generator_for_format(format_type: str):
    """
    Create the appropriate prompt generator function for the given format type.
    """
    if format_type == 'binary':
        return get_binary_prompt
    elif format_type == 'expanded':
        return get_expanded_prompt
    elif format_type == 'likert':
        return get_likert_prompt
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def prepare_participant_data_for_format(participant_data: Dict[str, Any],
                                        format_type: str,
                                        data_row: pd.Series) -> Dict[str, Any]:
    """
    Prepare participant data for the specific format type.
    For Study 3, this involves creating the appropriate format descriptions.
    """
    if format_type == 'binary':
        # Create binary personality description
        prepared_data = participant_data.copy()
        binary_description = generate_binary_personality_description(
            participant_data)
        prepared_data['binary_personality'] = binary_description
        return prepared_data
    elif format_type == 'expanded':
        # For expanded format, create combined_bfi2 using expanded_scale mapping
        prepared_data = participant_data.copy()
        prepared_data['combined_bfi2'] = create_expanded_description(data_row)
        return prepared_data
    elif format_type == 'likert':
        # For likert format, create combined_bfi2 using likert_scale mapping
        prepared_data = participant_data.copy()
        prepared_data['combined_bfi2'] = create_likert_description(data_row)
        return prepared_data
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def create_expanded_description(data_row: pd.Series) -> str:
    """
    Create expanded format description from BFI-2 data row.
    This replicates the logic from Study 3 expanded notebook.
    """
    # Import the expanded_scale mapping
    try:
        from schema_bfi2 import expanded_scale
    except ImportError:
        sys.path.append('../shared')
        from schema_bfi2 import expanded_scale

    # BFI columns (all 60 items)
    bfi_columns = [f"bfi{i}" for i in range(1, 61)]

    # Apply reverse coding (same as in Study 3 notebook)
    reverse_coding_map = {
        'bfi1': 'bfi1', 'bfi2': 'bfi2', 'bfi3': 'bfi3R', 'bfi4': 'bfi4R',
        'bfi5': 'bfi5R',
        'bfi6': 'bfi6', 'bfi7': 'bfi7', 'bfi8': 'bfi8R', 'bfi9': 'bfi9R',
        'bfi10': 'bfi10',
        'bfi11': 'bfi11R', 'bfi12': 'bfi12R', 'bfi13': 'bfi13',
        'bfi14': 'bfi14', 'bfi15': 'bfi15',
        'bfi16': 'bfi16R', 'bfi17': 'bfi17R', 'bfi18': 'bfi18',
        'bfi19': 'bfi19', 'bfi20': 'bfi20',
        'bfi21': 'bfi21', 'bfi22': 'bfi22R', 'bfi23': 'bfi23R',
        'bfi24': 'bfi24R', 'bfi25': 'bfi25R',
        'bfi26': 'bfi26R', 'bfi27': 'bfi27', 'bfi28': 'bfi28R',
        'bfi29': 'bfi29R', 'bfi30': 'bfi30R',
        'bfi31': 'bfi31R', 'bfi32': 'bfi32', 'bfi33': 'bfi33', 'bfi34': 'bfi34',
        'bfi35': 'bfi35',
        'bfi36': 'bfi36R', 'bfi37': 'bfi37R', 'bfi38': 'bfi38',
        'bfi39': 'bfi39', 'bfi40': 'bfi40',
        'bfi41': 'bfi41', 'bfi42': 'bfi42R', 'bfi43': 'bfi43',
        'bfi44': 'bfi44R', 'bfi45': 'bfi45R',
        'bfi46': 'bfi46', 'bfi47': 'bfi47R', 'bfi48': 'bfi48R',
        'bfi49': 'bfi49R', 'bfi50': 'bfi50R',
        'bfi51': 'bfi51R', 'bfi52': 'bfi52', 'bfi53': 'bfi53', 'bfi54': 'bfi54',
        'bfi55': 'bfi55R',
        'bfi56': 'bfi56', 'bfi57': 'bfi57', 'bfi58': 'bfi58R', 'bfi59': 'bfi59',
        'bfi60': 'bfi60'
    }

    # Apply reverse coding
    row_copy = data_row.copy()
    for key, value in reverse_coding_map.items():
        if value.endswith('R'):  # Reverse coded
            row_copy[key] = 6 - row_copy[key]

    # Map to expanded descriptions
    descriptions = []
    for col in bfi_columns:
        if col in expanded_scale and col in row_copy:
            value = int(row_copy[col])
            index = value - 1  # Convert to 0-index (1-5 scale becomes 0-4 index)
            if 0 <= index < len(expanded_scale[col]):
                descriptions.append(expanded_scale[col][index])

    return ' '.join(descriptions)


def create_likert_description(data_row: pd.Series) -> str:
    """
    Create likert format description from BFI-2 data row.
    This replicates the logic from Study 3 likert notebook.
    """
    # Import the likert_scale mapping
    try:
        from schema_bfi2 import likert_scale
    except ImportError:
        sys.path.append('../shared')
        from schema_bfi2 import likert_scale

    # BFI columns (all 60 items)
    bfi_columns = [f"bfi{i}" for i in range(1, 61)]

    # For likert format, use the original values (no reverse coding in the description)
    descriptions = []
    for col in bfi_columns:
        if col in likert_scale and col in data_row:
            value = int(data_row[col])
            descriptions.append(f"{likert_scale[col]} {value};")

    return ' '.join(descriptions)


def cleanup_duplicate_files(recovered_file_path: str, model_name: str,
                            format_config: Dict[str, Any]):
    """
    Clean up duplicate files after successful recovery.
    Keeps the recovered version and removes older duplicates.
    """
    results_dir = Path(__file__).parent / format_config['results_dir']
    file_pattern = format_config['file_pattern']

    # Find all files that match this model
    all_files = glob.glob(str(results_dir / file_pattern))
    model_files = []

    for file_path in all_files:
        if '.backup.' in file_path:
            continue
        file_stem = Path(file_path).stem
        file_model_name = file_stem.replace('bfi_to_minimarker_', '').replace(
            'binary_', '')

        # Check if this file is for the same base model (e.g., both "llama_temp1" and "llama_temp1_0" are "llama")
        base_model = file_model_name.split('_temp')[0]
        recovered_base_model = model_name.split('_temp')[0]

        if base_model == recovered_base_model and file_path != recovered_file_path:
            model_files.append(file_path)

    # Remove duplicate files (keep only the recovered one)
    for duplicate_file in model_files:
        try:
            # Create a backup of the duplicate before removing
            duplicate_backup = Path(duplicate_file).with_suffix(
                '.duplicate_backup.json')
            print(f"Creating duplicate backup: {duplicate_backup}")

            with open(duplicate_file, 'r') as f:
                duplicate_data = json.load(f)
            with open(duplicate_backup, 'w') as f:
                json.dump(duplicate_data, f, indent=2)

            # Remove the duplicate
            Path(duplicate_file).unlink()
            print(f"Removed duplicate file: {duplicate_file}")

        except Exception as e:
            print(
                f"Warning: Could not remove duplicate file {duplicate_file}: {str(e)}")


def recover_participants_for_model(model_info: Dict[str, Any],
                                   format_config: Dict[str, Any],
                                   dry_run: bool = False) -> Dict[str, Any]:
    """
    Recover missing participants for a specific model.
    """
    model_name = \
        [k for k in model_info.keys() if k != 'file_path' and k != 'data_path'][
            0]
    info = model_info[model_name]

    print(f"\n{'=' * 60}")
    print(f"RECOVERING PARTICIPANTS FOR {model_name.upper()}")
    print(f"{'=' * 60}")

    if info['missing_participants'] == 0:
        print("No missing participants - skipping recovery")
        return {'status': 'no_recovery_needed'}

    if dry_run:
        print(
            f"DRY RUN: Would recover {info['missing_participants']} participants")
        print(f"Estimated API calls: {len(info['problematic_indices'])}")
        return {'status': 'dry_run'}

    # Load original data and results
    data = pd.read_csv(info['data_path'])
    with open(info['file_path'], 'r') as f:
        original_results = json.load(f)

    # Determine format type and create appropriate prompt generator
    format_type = format_config['type']
    prompt_generator = create_prompt_generator_for_format(format_type)

    # Extract model name for API call
    # For OpenAI models, convert to the format expected by portal.py
    if model_name.startswith('openai_'):
        api_model = model_name.replace('_', '-')
    else:
        # For other models, convert underscores to hyphens
        api_model = model_name.replace('_', '-')

    # Remove temperature suffix if present
    if api_model.endswith('-temp1'):
        api_model = api_model[:-6]  # Remove '-temp1'
    elif api_model.endswith('-temp1-0'):
        api_model = api_model[:-8]  # Remove '-temp1-0'

    print(
        f"Attempting to recover {len(info['problematic_indices'])} participants")
    print(f"Using model: {api_model}")
    print(f"Using format type: {format_type}")

    # Prepare participants data for recovery
    participants_to_recover = []
    for idx in info['problematic_indices']:
        if idx < len(data):
            data_row = data.iloc[idx]

            # Convert Study 3 data format to the format expected by prompts
            participant_data = {
                'bfi2_e': data_row['bfi_e'],
                'bfi2_a': data_row['bfi_a'],
                'bfi2_c': data_row['bfi_c'],
                'bfi2_n': data_row['bfi_n'],
                'bfi2_o': data_row['bfi_o']
            }

            # Prepare data for the specific format
            prepared_data = prepare_participant_data_for_format(
                participant_data, format_type, data_row)
            participants_to_recover.append({
                'original_index': idx,
                'participant_data': prepared_data
            })

    if not participants_to_recover:
        print("No valid participants to recover")
        return {'status': 'no_valid_participants'}

    # Use existing simulation framework to recover participants
    print(
        f"Running simulation for {len(participants_to_recover)} participants...")

    # Create simplified participant data for simulation
    recovery_data = []
    for p in participants_to_recover:
        recovery_data.append(p['participant_data'])

    try:
        # Create simulation config (smaller batch size for better reliability)
        config = SimulationConfig(
            model=api_model,
            temperature=1.0,
            batch_size=5,  # Smaller batch size like simple_recover.py
            max_workers=5
        )

        # Determine personality key based on format type
        if format_type == 'binary':
            personality_key = 'binary_personality'
        else:
            personality_key = 'combined_bfi2'

        # Run simulation using the framework
        recovery_results = run_batch_simulation(
            participants_data=recovery_data,
            prompt_generator=prompt_generator,
            config=config,
            personality_key=personality_key
        )

        print(f"Recovery simulation completed: {len(recovery_results)} results")

        # Update original results with recovered participants
        recovered_count = 0
        for i, result in enumerate(recovery_results):
            if 'error' not in result:
                original_index = participants_to_recover[i]['original_index']
                original_results[original_index] = result
                recovered_count += 1
                print(f"  ✓ Recovered participant {original_index}")
            else:
                original_index = participants_to_recover[i]['original_index']
                print(
                    f"  ✗ Still failed participant {original_index}: {result.get('error', 'Unknown error')}")

        print(
            f"\nSuccessfully recovered {recovered_count}/{len(participants_to_recover)} participants")

        # Save updated results
        backup_path = Path(info['file_path']).with_suffix('.backup.json')
        print(f"Creating backup: {backup_path}")

        with open(backup_path, 'w') as f:
            json.dump(original_results, f, indent=2)

        # Save recovered results
        with open(info['file_path'], 'w') as f:
            json.dump(original_results, f, indent=2)

        print(f"Updated results saved to: {info['file_path']}")

        # Clean up duplicate files if recovery was successful
        # Commented out for now to avoid conflicts during recovery
        # if recovered_count > 0:
        #     cleanup_duplicate_files(info['file_path'], model_name, format_config)

        return {
            'status': 'success',
            'recovered_count': recovered_count,
            'attempted_count': len(participants_to_recover),
            'backup_path': str(backup_path)
        }

    except Exception as e:
        print(f"Error during recovery: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'error': str(e)}


def main():
    """Main recovery function."""
    parser = argparse.ArgumentParser(
        description="Recover missing participants from Study 3 simulation results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze only, don't perform recovery")
    parser.add_argument("--format",
                        choices=['binary', 'expanded', 'likert', 'all'],
                        default='all', help="Format to recover (default: all)")

    args = parser.parse_args()

    # Define format configurations for Study 3
    format_configs = [
        {
            'name': 'Binary Format',
            'type': 'binary',
            'results_dir': 'study_3_binary_results',
            'file_pattern': 'bfi_to_minimarker_binary_*.json'
        },
        {
            'name': 'Expanded Format',
            'type': 'expanded',
            'results_dir': 'study_3_expanded_results',
            'file_pattern': 'bfi_to_minimarker_*.json'
        },
        {
            'name': 'Likert Format',
            'type': 'likert',
            'results_dir': 'study_3_likert_results',
            'file_pattern': 'bfi_to_minimarker_*.json'
        }
    ]

    # Filter formats based on argument
    if args.format != 'all':
        format_configs = [c for c in format_configs if c['type'] == args.format]

    print("=" * 80)
    print("UNIFIED PARTICIPANT RECOVERY FOR STUDY 3")
    print("=" * 80)
    print(
        f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'RECOVERY'}")
    print(f"Formats: {[c['name'] for c in format_configs]}")

    # Step 1: Analyze all formats to identify missing participants
    all_analyses = {}
    for config in format_configs:
        analysis = analyze_format_for_missing_participants(config)
        if analysis:
            all_analyses[config['name']] = {
                'config': config,
                'analysis': analysis
            }

    if not all_analyses:
        print("\nNo valid analyses found. Exiting.")
        return

    # Step 2: Summary of missing participants
    print(f"\n{'=' * 80}")
    print("SUMMARY OF MISSING PARTICIPANTS")
    print(f"{'=' * 80}")

    total_missing = 0
    for format_name, format_data in all_analyses.items():
        print(f"\n{format_name}:")
        for model_name, model_info in format_data['analysis'].items():
            missing = model_info['missing_participants']
            total_missing += missing
            print(f"  {model_name}: {missing} missing participants")

    print(f"\nTotal missing participants across all formats: {total_missing}")

    if total_missing == 0:
        print("✓ All formats have complete participant data!")
        return

    # Step 3: Recover missing participants
    if not args.dry_run:
        print(f"\n{'=' * 80}")
        print("BEGINNING RECOVERY PROCESS")
        print(f"{'=' * 80}")

        recovery_summary = {}
        for format_name, format_data in all_analyses.items():
            format_config = format_data['config']
            format_analysis = format_data['analysis']

            recovery_summary[format_name] = {}

            for model_name, model_info in format_analysis.items():
                if model_info['missing_participants'] > 0:
                    result = recover_participants_for_model(
                        {model_name: model_info},
                        format_config,
                        dry_run=args.dry_run
                    )
                    recovery_summary[format_name][model_name] = result

        # Final summary
        print(f"\n{'=' * 80}")
        print("RECOVERY COMPLETE")
        print(f"{'=' * 80}")

        total_recovered = 0
        files_cleaned = 0
        for format_name, format_results in recovery_summary.items():
            print(f"\n{format_name}:")
            for model_name, result in format_results.items():
                if result['status'] == 'success':
                    recovered = result['recovered_count']
                    attempted = result['attempted_count']
                    total_recovered += recovered
                    print(f"  {model_name}: {recovered}/{attempted} recovered")
                    if recovered > 0:
                        files_cleaned += 1
                        print(
                            f"    ✓ Cleaned up duplicates, backup at: {result.get('backup_path', 'N/A')}")
                else:
                    print(f"  {model_name}: {result['status']}")

        print(f"\nTotal participants recovered: {total_recovered}")
        print(f"Success rate: {100 * total_recovered / total_missing:.1f}%")
        print(f"Files cleaned up: {files_cleaned}")

        print(
            f"\nRecommendation: Re-run unified_convergent_analysis.py to verify recovery")

    else:
        print(f"\nDRY RUN COMPLETE - no recovery performed")


if __name__ == "__main__":
    main() 