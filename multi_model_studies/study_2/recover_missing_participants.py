#!/usr/bin/env python3
"""
Unified Participant Recovery Script for Study 2

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
from mini_marker_prompt import get_expanded_prompt, get_likert_prompt, generate_binary_personality_description
from binary_baseline_prompt import get_binary_prompt

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
    'Cold', 'Envious', 'Harsh', 'Jealous', 'Rude', 'Unsympathetic',  # Agreeableness (reverse)
    'Careless', 'Disorganized', 'Inefficient', 'Sloppy',  # Conscientiousness (reverse)
    'Relaxed', 'Unenvious', 'Uncreative', 'Unintellectual'  # Neuroticism/Openness (reverse)
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

def analyze_format_for_missing_participants(format_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a format to identify missing participants.
    Returns analysis results including problematic indices.
    """
    format_name = format_config['name']
    format_type = format_config['type']
    results_dir = Path(__file__).parent / format_config['results_dir']
    file_pattern = format_config['file_pattern']
    
    print(f"\n{'='*60}")
    print(f"ANALYZING {format_name.upper()} FOR MISSING PARTICIPANTS")
    print(f"{'='*60}")
    
    # Load empirical data
    if format_type == 'binary':
        # For binary, use the binary results directory data
        data_path = Path(__file__).parent / 'study_2_binary_results' / 'study2_preprocessed_data.csv'
    elif format_type == 'expanded':
        # For expanded, use the expanded results directory data
        # Check which expanded directory we're using
        if 'i_am' in results_dir.name:
            data_path = Path(__file__).parent / 'study_2_expanded_results_i_am' / 'study2_preprocessed_data.csv'
        else:
            data_path = Path(__file__).parent / 'study_2_expanded_results_you_are' / 'study2_preprocessed_data.csv'
    elif format_type == 'likert':
        data_path = Path(__file__).parent / 'study_2_likert_results' / 'study2_likert_preprocessed_data.csv'
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
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
        model_name = Path(json_file).stem.replace('bfi_to_minimarker_', '').replace('_temp1_0', '').replace('binary_', '')
        
        print(f"\n--- Analyzing {model_name} ---")
        
        try:
            # Load JSON data
            with open(json_file, 'r') as f:
                sim_json = json.load(f)
            
            total_responses = len(sim_json)
            print(f"Total responses: {total_responses}")
            
            # Clean keys and validate each response
            problematic_indices = []
            valid_responses = 0
            
            for idx, response in enumerate(sim_json):
                is_valid, errors = validate_response(response)
                if is_valid:
                    valid_responses += 1
                else:
                    problematic_indices.append(idx)
            
            print(f"Valid responses: {valid_responses}")
            print(f"Problematic responses: {len(problematic_indices)}")
            
            # Test aggregation to see how many participants survive
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
            sim_domains = aggregate_minimarker_for_validation(sim_df, format_type)
            
            # Apply same filtering as unified_convergent_analysis.py
            n = min(len(data), len(sim_domains))
            sim_tda = sim_domains.loc[:n-1, ['E','A','C','N','O']].reset_index(drop=True)
            
            # Clean invalid data (same as unified_convergent_analysis.py)
            valid_mask = ~(sim_tda.isna().any(axis=1) | np.isinf(sim_tda.values).any(axis=1))
            final_n = sum(valid_mask)
            
            print(f"After aggregation and filtering: {final_n} participants")
            print(f"Missing participants: {expected_participants - final_n}")
            
            # Store analysis
            format_analysis[model_name] = {
                'file_path': json_file,
                'total_responses': total_responses,
                'valid_responses': valid_responses,
                'problematic_indices': problematic_indices,
                'final_participants': final_n,
                'missing_participants': expected_participants - final_n,
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
        def binary_prompt_generator(personality_description):
            # For binary format, the personality_description is actually the participant data
            # We need to generate the binary personality description from the participant data
            if isinstance(personality_description, dict):
                # This is participant data, generate binary description
                binary_desc = generate_binary_personality_description(personality_description)
                return get_binary_prompt(binary_desc)
            else:
                # This is already a personality description string
                return get_binary_prompt(personality_description)
        return binary_prompt_generator
    elif format_type == 'expanded':
        return get_expanded_prompt
    elif format_type == 'likert':
        return get_likert_prompt
    else:
        raise ValueError(f"Unknown format type: {format_type}")

def prepare_participant_data_for_format(participant_data: Dict[str, Any], format_type: str) -> Dict[str, Any]:
    """
    Prepare participant data for the specific format type.
    """
    if format_type == 'binary':
        # For binary format, we need to create a special structure
        # The simulation framework expects a 'personality_key' field
        prepared_data = participant_data.copy()
        # Generate binary personality description and store it
        binary_description = generate_binary_personality_description(participant_data)
        prepared_data['bfi2_binary'] = binary_description
        return prepared_data
    elif format_type == 'expanded':
        # For expanded format, the prompt generator expects the combined_bfi2 field
        return participant_data
    elif format_type == 'likert':
        # For likert format, the prompt generator expects the combined_bfi2 field
        return participant_data
    else:
        raise ValueError(f"Unknown format type: {format_type}")

def recover_participants_for_model(model_info: Dict[str, Any], 
                                 format_config: Dict[str, Any],
                                 dry_run: bool = False) -> Dict[str, Any]:
    """
    Recover missing participants for a specific model.
    """
    model_name = [k for k in model_info.keys() if k != 'file_path' and k != 'data_path'][0]
    info = model_info[model_name]
    
    print(f"\n{'='*60}")
    print(f"RECOVERING PARTICIPANTS FOR {model_name.upper()}")
    print(f"{'='*60}")
    
    if info['missing_participants'] == 0:
        print("No missing participants - skipping recovery")
        return {'status': 'no_recovery_needed'}
    
    if dry_run:
        print(f"DRY RUN: Would recover {info['missing_participants']} participants")
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
    api_model = model_name.replace('openai_', '').replace('_', '-')
    
    print(f"Attempting to recover {len(info['problematic_indices'])} participants")
    print(f"Using model: {api_model}")
    print(f"Using format type: {format_type}")
    
    # Prepare participants data for recovery
    participants_to_recover = []
    for idx in info['problematic_indices']:
        if idx < len(data):
            participant_data = data.iloc[idx].to_dict()
            # Prepare data for the specific format
            prepared_data = prepare_participant_data_for_format(participant_data, format_type)
            participants_to_recover.append({
                'original_index': idx,
                'participant_data': prepared_data
            })
    
    if not participants_to_recover:
        print("No valid participants to recover")
        return {'status': 'no_valid_participants'}
    
    # Use existing simulation framework to recover participants
    print(f"Running simulation for {len(participants_to_recover)} participants...")
    
    # Create simplified participant data for simulation
    recovery_data = []
    for p in participants_to_recover:
        recovery_data.append(p['participant_data'])
    
    try:
        # Create simulation config
        config = SimulationConfig(
            model=api_model,
            temperature=1.0,
            batch_size=10,
            max_workers=10
        )
        
        # Determine personality key based on format type
        if format_type == 'binary':
            personality_key = 'bfi2_binary'  # This will be handled by the binary prompt generator
        else:
            personality_key = 'combined_bfi2'
        
        # Run simulation using the framework
        recovery_results = run_batch_simulation(
            participants_data=recovery_data,
            prompt_generator=prompt_generator,
            config=config,
            personality_key=personality_key
        )
        
        # Update original results with recovered participants
        recovered_count = 0
        for i, result in enumerate(recovery_results):
            if 'error' not in result:
                original_index = participants_to_recover[i]['original_index']
                original_results[original_index] = result
                recovered_count += 1
        
        print(f"Successfully recovered {recovered_count} participants")
        
        # Save updated results
        backup_path = Path(info['file_path']).with_suffix('.backup.json')
        print(f"Creating backup: {backup_path}")
        
        with open(backup_path, 'w') as f:
            json.dump(original_results, f, indent=2)
        
        # Save recovered results
        with open(info['file_path'], 'w') as f:
            json.dump(original_results, f, indent=2)
        
        print(f"Updated results saved to: {info['file_path']}")
        
        return {
            'status': 'success',
            'recovered_count': recovered_count,
            'attempted_count': len(participants_to_recover)
        }
        
    except Exception as e:
        print(f"Error during recovery: {str(e)}")
        return {'status': 'error', 'error': str(e)}

def main():
    """Main recovery function."""
    parser = argparse.ArgumentParser(description="Recover missing participants from Study 2 simulation results")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, don't perform recovery")
    parser.add_argument("--format", choices=['binary', 'expanded', 'likert', 'all'], 
                       default='all', help="Format to recover (default: all)")
    
    args = parser.parse_args()
    
    # Define format configurations (same as unified_convergent_analysis.py)
    format_configs = [
        {
            'name': 'Binary Baseline',
            'type': 'binary',
            'results_dir': 'study_2_binary_results',
            'file_pattern': 'bfi_to_minimarker_binary_*.json'
        },
        {
            'name': 'Expanded Format (I am)',
            'type': 'expanded',
            'results_dir': 'study_2_expanded_results_i_am',
            'file_pattern': 'bfi_to_minimarker_*.json'
        },
        {
            'name': 'Expanded Format (You are)',
            'type': 'expanded',
            'results_dir': 'study_2_expanded_results_you_are',
            'file_pattern': 'bfi_to_minimarker_*.json'
        },
        {
            'name': 'Likert Format',
            'type': 'likert',
            'results_dir': 'study_2_likert_results',
            'file_pattern': 'bfi_to_minimarker_*.json'
        }
    ]
    
    # Filter formats based on argument
    if args.format != 'all':
        format_configs = [c for c in format_configs if c['type'] == args.format]
    
    print("="*80)
    print("UNIFIED PARTICIPANT RECOVERY FOR STUDY 2")
    print("="*80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    print(f"\n{'='*80}")
    print("SUMMARY OF MISSING PARTICIPANTS")
    print(f"{'='*80}")
    
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
        print(f"\n{'='*80}")
        print("BEGINNING RECOVERY PROCESS")
        print(f"{'='*80}")
        
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
        print(f"\n{'='*80}")
        print("RECOVERY COMPLETE")
        print(f"{'='*80}")
        
        total_recovered = 0
        for format_name, format_results in recovery_summary.items():
            print(f"\n{format_name}:")
            for model_name, result in format_results.items():
                if result['status'] == 'success':
                    recovered = result['recovered_count']
                    attempted = result['attempted_count']
                    total_recovered += recovered
                    print(f"  {model_name}: {recovered}/{attempted} recovered")
                else:
                    print(f"  {model_name}: {result['status']}")
        
        print(f"\nTotal participants recovered: {total_recovered}")
        print(f"Success rate: {100 * total_recovered / total_missing:.1f}%")
        
        print(f"\nRecommendation: Re-run unified_convergent_analysis.py to verify recovery")
    
    else:
        print(f"\nDRY RUN COMPLETE - no recovery performed")

if __name__ == "__main__":
    main()