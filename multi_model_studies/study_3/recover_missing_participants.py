#!/usr/bin/env python3
"""
Study 2b Participant Recovery Script

This script detects missing participants in Study 2b moral and risk scenario simulations
and recovers them using the appropriate simulation models.

Usage:
    python recover_missing_participants.py [--dry-run] [--scenario SCENARIO]
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
sys.path.append('../../')
from portal import get_model_response
from moral_stories import get_prompt as get_moral_prompt
from risk_taking import get_prompt as get_risk_prompt

def validate_moral_response(response_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate moral scenario response"""
    errors = []
    
    if not isinstance(response_dict, dict):
        errors.append("Response is not a dictionary")
        return False, errors
    
    required_fields = ['Confidential_Info', 'Underage_Drinking', 'Exam_Cheating', 'Honest_Feedback', 'Workplace_Theft']
    
    for field in required_fields:
        if field not in response_dict:
            errors.append(f"Missing field: {field}")
            continue
            
        value = response_dict[field]
        if not isinstance(value, (int, float)) or not (1 <= value <= 10):
            errors.append(f"Invalid value for {field}: {value} (expected 1-10)")
    
    return len(errors) == 0, errors


def validate_risk_response(response_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate risk scenario response"""
    errors = []
    
    if not isinstance(response_dict, dict):
        errors.append("Response is not a dictionary")
        return False, errors
    
    required_fields = ['Investment', 'Extreme_Sports', 'Entrepreneurial_Venture', 'Confessing_Feelings', 'Study_Overseas']
    
    for field in required_fields:
        if field not in response_dict:
            errors.append(f"Missing field: {field}")
            continue
            
        value = response_dict[field]
        if not isinstance(value, (int, float)) or not (1 <= value <= 10):
            errors.append(f"Invalid value for {field}: {value} (expected 1-10)")
    
    return len(errors) == 0, errors


def validate_response(response_data: Dict[str, Any], scenario_type: str) -> Tuple[bool, List[str]]:
    """Validate a single response using appropriate validation logic"""
    errors = []
    
    # Check for explicit error responses first
    if isinstance(response_data, dict) and 'error' in response_data:
        errors.append(f"Error response: {response_data.get('error', 'Unknown error')}")
        return False, errors
    
    # Validate based on scenario type
    if scenario_type == 'moral':
        return validate_moral_response(response_data)
    elif scenario_type == 'risk':
        return validate_risk_response(response_data)
    else:
        errors.append(f"Unknown scenario type: {scenario_type}")
        return False, errors


def analyze_scenario_for_missing_participants(scenario_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a scenario to identify missing participants.
    Returns analysis results including problematic indices.
    """
    scenario_name = scenario_config['name']
    scenario_type = scenario_config['type']
    results_dir = Path(__file__).parent / scenario_config['results_dir']
    file_pattern = scenario_config['file_pattern']
    
    print(f"\n{'=' * 60}")
    print(f"ANALYZING {scenario_name.upper()} FOR MISSING PARTICIPANTS")
    print(f"{'=' * 60}")
    
    # Load Study 2b data
    data_path = Path(__file__).parent / '../../raw_data/york_data_clean.csv'
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return None
    
    # Load and filter data to match simulation criteria
    data = pd.read_csv(data_path)
    
    # Apply same filtering as in simulation scripts (English == 5, no null in bfi6)
    data = data[data['8) English language reading/comprehension ability:'] == 5]
    data = data.dropna(subset=[data.columns[17]])  # bfi6 column
    
    expected_participants = len(data)
    print(f"Expected participants: {expected_participants}")
    
    # Find simulation files
    if not results_dir.exists():
        print(f"Warning: Results directory not found at {results_dir}")
        return None
    
    # Handle both direct and subdirectory patterns
    if '/' in file_pattern:
        json_files = glob.glob(str(results_dir / file_pattern))
    else:
        json_files = glob.glob(str(results_dir / file_pattern))
    
    if len(json_files) == 0:
        print(f"Warning: No JSON files found matching pattern {results_dir / file_pattern}")
        return None
    
    print(f"Found {len(json_files)} model files")
    
    scenario_analysis = {}
    
    # Process each model
    for json_file in json_files:
        if '.backup.' in json_file:
            continue
        
        file_stem = Path(json_file).stem
        model_name = file_stem.replace('moral_', '').replace('risk_', '').replace('_temp1.0', '').replace('_temp0.0', '')
        
        print(f"\n--- Analyzing {model_name} ({Path(json_file).name}) ---")
        
        try:
            # Load JSON data
            with open(json_file, 'r') as f:
                sim_json = json.load(f)
            
            total_responses = len(sim_json)
            print(f"Total responses: {total_responses}")
            
            # Find error responses
            error_indices = []
            valid_responses = 0
            
            for idx, response in enumerate(sim_json):
                # Check for explicit error responses first
                if isinstance(response, dict) and 'error' in response:
                    error_indices.append(idx)
                else:
                    # Then check for validation issues
                    is_valid, errors = validate_response(response, scenario_type)
                    if is_valid:
                        valid_responses += 1
                    else:
                        error_indices.append(idx)
            
            print(f"Valid responses: {valid_responses}")
            print(f"Error/problematic responses: {len(error_indices)}")
            
            missing_participants = expected_participants - valid_responses
            
            print(f"Missing participants: {missing_participants}")
            
            # Store analysis
            scenario_analysis[model_name] = {
                'file_path': json_file,
                'total_responses': total_responses,
                'valid_responses': valid_responses,
                'problematic_indices': error_indices,
                'final_participants': valid_responses,
                'missing_participants': missing_participants,
                'data_path': str(data_path),
                'scenario_type': scenario_type
            }
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {str(e)}")
            continue
    
    return scenario_analysis


def load_york_data():
    """Load and preprocess York behavioral data"""
    data_path = Path('../../raw_data/york_data_clean.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = pd.read_csv(data_path)
    
    # Apply same filtering as in simulation scripts
    data = data[data['8) English language reading/comprehension ability:'] == 5]
    data = data.dropna(subset=[data.columns[17]])  # bfi6 column
    
    return data


def prepare_personality_descriptions(data_row):
    """Prepare personality descriptions for all formats"""
    # Extract BFI-2 scores for personality description
    personality_data = {}
    
    # Map columns to personality traits (adjust based on your data structure)
    bfi_columns = [col for col in data_row.index if 'bfi' in col.lower()]
    
    # Simple approach: use raw BFI values for personality description
    personality_str = f"Big Five personality scores: "
    personality_str += f"Extraversion={data_row.get('bfi_e', 'N/A')}, "
    personality_str += f"Agreeableness={data_row.get('bfi_a', 'N/A')}, "
    personality_str += f"Conscientiousness={data_row.get('bfi_c', 'N/A')}, "
    personality_str += f"Neuroticism={data_row.get('bfi_n', 'N/A')}, "
    personality_str += f"Openness={data_row.get('bfi_o', 'N/A')}"
    
    return personality_str


def recover_participants_for_model(model_info: Dict[str, Any],
                                   scenario_config: Dict[str, Any],
                                   dry_run: bool = False) -> Dict[str, Any]:
    """
    Recover missing participants for a specific model.
    """
    model_name = list(model_info.keys())[0]
    info = model_info[model_name]
    
    print(f"\n{'=' * 60}")
    print(f"RECOVERING PARTICIPANTS FOR {model_name.upper()}")
    print(f"{'=' * 60}")
    
    if info['missing_participants'] == 0:
        print("No missing participants - skipping recovery")
        return {'status': 'no_recovery_needed'}
    
    if dry_run:
        print(f"DRY RUN: Would recover {info['missing_participants']} participants")
        print(f"Estimated API calls: {len(info['problematic_indices'])}")
        return {'status': 'dry_run'}
    
    # Load original data and results
    data = load_york_data()
    with open(info['file_path'], 'r') as f:
        original_results = json.load(f)
    
    # Determine scenario type and create appropriate prompt function
    scenario_type = info['scenario_type']
    
    if scenario_type == 'moral':
        prompt_func = get_moral_prompt
    elif scenario_type == 'risk':
        prompt_func = get_risk_prompt
    else:
        return {'status': 'error', 'error': f"Unknown scenario type: {scenario_type}"}
    
    print(f"Attempting to recover {len(info['problematic_indices'])} participants")
    print(f"Using scenario type: {scenario_type}")
    
    # Prepare participants data for recovery
    participants_to_recover = []
    for idx in info['problematic_indices']:
        if idx < len(data):
            data_row = data.iloc[idx]
            
            # Prepare personality description
            personality = prepare_personality_descriptions(data_row)
            
            participants_to_recover.append({
                'original_index': idx,
                'personality': personality
            })
    
    if not participants_to_recover:
        print("No valid participants to recover")
        return {'status': 'no_valid_participants'}
    
    # Process participants individually to avoid overwhelming API
    print(f"Processing {len(participants_to_recover)} participants...")
    
    recovered_count = 0
    for i, participant in enumerate(participants_to_recover):
        original_index = participant['original_index']
        
        try:
            # Generate prompt
            prompt = prompt_func(participant['personality'])
            
            # Get response from model
            response = get_model_response(model_name, prompt, temperature=0.0)
            
            # Parse and validate response
            if isinstance(response, str):
                try:
                    response_dict = json.loads(response.strip())
                except json.JSONDecodeError:
                    # Try to extract JSON from text
                    import re
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    matches = re.findall(json_pattern, response, re.DOTALL)
                    if matches:
                        try:
                            response_dict = json.loads(matches[0])
                        except:
                            response_dict = {"error": "Could not parse JSON"}
                    else:
                        response_dict = {"error": "No valid JSON found"}
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {"error": f"Unexpected response type: {type(response)}"}
            
            # Validate response
            is_valid, errors = validate_response(response_dict, scenario_type)
            
            if is_valid:
                original_results[original_index] = response_dict
                recovered_count += 1
                print(f"  ✓ Recovered participant {original_index}")
            else:
                original_results[original_index] = {"error": f"Invalid response: {errors}", "raw_response": response}
                print(f"  ✗ Failed participant {original_index}: {errors}")
            
            # Add small delay to avoid rate limits
            import time
            time.sleep(0.2)
            
        except Exception as e:
            original_results[original_index] = {"error": f"Processing error: {str(e)}"}
            print(f"  ✗ Error processing participant {original_index}: {str(e)}")
    
    print(f"\nSuccessfully recovered {recovered_count}/{len(participants_to_recover)} participants")
    
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
        'attempted_count': len(participants_to_recover),
        'backup_path': str(backup_path)
    }


def main():
    """Main recovery function."""
    parser = argparse.ArgumentParser(
        description="Recover missing participants from Study 2b simulation results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Analyze only, don't perform recovery")
    parser.add_argument("--scenario",
                        choices=['moral', 'risk', 'all'],
                        default='all', help="Scenario to recover (default: all)")
    
    args = parser.parse_args()
    
    # Define scenario configurations for Study 2b
    scenario_configs = [
        {
            'name': 'Moral Scenarios',
            'type': 'moral',
            'results_dir': 'study_4_moral_results',
            'file_pattern': 'moral_*.json'
        },
        {
            'name': 'Risk Scenarios',
            'type': 'risk',
            'results_dir': 'study_4_risk_results',
            'file_pattern': 'risk_*.json'
        },
        {
            'name': 'Generalized Expanded Moral',
            'type': 'moral',
            'results_dir': 'study_4_generalized_results/bfi_expanded_format',
            'file_pattern': 'moral/*.json'
        },
        {
            'name': 'Generalized Expanded Risk',
            'type': 'risk',
            'results_dir': 'study_4_generalized_results/bfi_expanded_format',
            'file_pattern': 'risk/*.json'
        },
        {
            'name': 'Generalized Likert Moral',
            'type': 'moral',
            'results_dir': 'study_4_generalized_results/bfi_likert_format',
            'file_pattern': 'moral/*.json'
        },
        {
            'name': 'Generalized Likert Risk',
            'type': 'risk',
            'results_dir': 'study_4_generalized_results/bfi_likert_format',
            'file_pattern': 'risk/*.json'
        },
        {
            'name': 'Generalized Binary Elaborated Moral',
            'type': 'moral',
            'results_dir': 'study_4_generalized_results/bfi_binary_elaborated_format',
            'file_pattern': 'moral/*.json'
        },
        {
            'name': 'Generalized Binary Elaborated Risk',
            'type': 'risk',
            'results_dir': 'study_4_generalized_results/bfi_binary_elaborated_format',
            'file_pattern': 'risk/*.json'
        },
        {
            'name': 'Generalized Binary Simple Moral',
            'type': 'moral',
            'results_dir': 'study_4_generalized_results/bfi_binary_simple_format',
            'file_pattern': 'moral/*.json'
        },
        {
            'name': 'Generalized Binary Simple Risk',
            'type': 'risk',
            'results_dir': 'study_4_generalized_results/bfi_binary_simple_format',
            'file_pattern': 'risk/*.json'
        }
    ]
    
    # Filter scenarios based on argument
    if args.scenario != 'all':
        scenario_configs = [c for c in scenario_configs if c['type'] == args.scenario]
    
    print("=" * 80)
    print("STUDY 2b PARTICIPANT RECOVERY")
    print("=" * 80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'RECOVERY'}")
    print(f"Scenarios: {[c['name'] for c in scenario_configs]}")
    
    # Step 1: Analyze all scenarios to identify missing participants
    all_analyses = {}
    for config in scenario_configs:
        analysis = analyze_scenario_for_missing_participants(config)
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
    for scenario_name, scenario_data in all_analyses.items():
        print(f"\n{scenario_name}:")
        for model_name, model_info in scenario_data['analysis'].items():
            missing = model_info['missing_participants']
            total_missing += missing
            print(f"  {model_name}: {missing} missing participants")
    
    print(f"\nTotal missing participants across all scenarios: {total_missing}")
    
    if total_missing == 0:
        print("✓ All scenarios have complete participant data!")
        return
    
    # Step 3: Recover missing participants
    if not args.dry_run:
        print(f"\n{'=' * 80}")
        print("BEGINNING RECOVERY PROCESS")
        print(f"{'=' * 80}")
        
        recovery_summary = {}
        for scenario_name, scenario_data in all_analyses.items():
            scenario_config = scenario_data['config']
            scenario_analysis = scenario_data['analysis']
            
            recovery_summary[scenario_name] = {}
            
            for model_name, model_info in scenario_analysis.items():
                if model_info['missing_participants'] > 0:
                    result = recover_participants_for_model(
                        {model_name: model_info},
                        scenario_config,
                        dry_run=args.dry_run
                    )
                    recovery_summary[scenario_name][model_name] = result
        
        # Final summary
        print(f"\n{'=' * 80}")
        print("RECOVERY COMPLETE")
        print(f"{'=' * 80}")
        
        total_recovered = 0
        for scenario_name, scenario_results in recovery_summary.items():
            print(f"\n{scenario_name}:")
            for model_name, result in scenario_results.items():
                if result['status'] == 'success':
                    recovered = result['recovered_count']
                    attempted = result['attempted_count']
                    total_recovered += recovered
                    print(f"  {model_name}: {recovered}/{attempted} recovered")
                else:
                    print(f"  {model_name}: {result['status']}")
        
        print(f"\nTotal participants recovered: {total_recovered}")
        print(f"Success rate: {100 * total_recovered / total_missing:.1f}%")
        
        print("\nRecommendation: Re-run behavioral analysis scripts to verify recovery")
    else:
        print(f"\nDRY RUN COMPLETE - no recovery performed")


if __name__ == "__main__":
    main()