#!/usr/bin/env python3
"""
Recover Failed Responses for Study 4

This script identifies and re-simulates failed/invalid responses in Study 4 results
by re-running the simulation for specific participant indices and formats.

Usage:
    python recover_failed_responses.py [--dry-run] [--format FORMAT] [--scenario SCENARIO]
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
import time
import re

# Add shared modules to path
sys.path.append('../shared')
from portal import get_model_response
from moral_stories import get_prompt as get_moral_prompt
from risk_taking import get_prompt as get_risk_prompt

def extract_jsons(text):
    """Extract JSON objects from text"""
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text)
    
    jsons = []
    for match in matches:
        try:
            json_obj = json.loads(match)
            jsons.append(json_obj)
        except json.JSONDecodeError:
            continue
    
    return jsons

def validate_moral_response(response_dict):
    """Validate that the moral response contains expected fields"""
    # The actual scenario names from the prompt
    required_fields = ['Confidential_Info', 'Underage_Drinking', 'Exam_Cheating', 'Honest_Feedback', 'Workplace_Theft']

    if not isinstance(response_dict, dict):
        return False, "Response is not a dictionary"

    missing_fields = [field for field in required_fields if field not in response_dict]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"

    # Check if values are numeric (ratings)
    for field in required_fields:
        value = response_dict.get(field)
        if not isinstance(value, (int, float)) or not (1 <= value <= 10):
            return False, f"Invalid value for {field}: {value} (expected 1-10)"

    return True, "Valid response"

def validate_risk_response(response_dict):
    """Validate that the risk response contains expected fields"""
    required_fields = ['Investment', 'Extreme_Sports', 'Entrepreneurial_Venture', 'Confessing_Feelings', 'Study_Overseas']

    if not isinstance(response_dict, dict):
        return False, "Response is not a dictionary"

    missing_fields = [field for field in required_fields if field not in response_dict]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"

    # Check if values are numeric (ratings)
    for field in required_fields:
        value = response_dict.get(field)
        if not isinstance(value, (int, float)) or not (1 <= value <= 10):
            return False, f"Invalid value for {field}: {value} (expected 1-10)"

    return True, "Valid response"

def load_york_data():
    """Load and preprocess York behavioral data"""
    data_path = Path('../../raw_data/york_data_clean.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path)
    print(f"Loaded York data shape: {data.shape}")

    # Filter for good English comprehension (value 5 = excellent)
    data = data[data['8) English language reading/comprehension ability:'] == 5]
    # Remove rows with null values in bfi6 column (index 17)
    data = data.dropna(subset=[data.columns[17]])

    print(f"Filtered data shape: {data.shape}")
    return data

def analyze_results_for_failed_responses(results_dir: Path, scenario_type: str) -> Dict[str, Any]:
    """
    Analyze results to identify failed/invalid responses.
    Returns analysis results including problematic indices.
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING {scenario_type.upper()} RESULTS FOR FAILED RESPONSES")
    print(f"{'='*60}")
    
    # Load York data for reference
    try:
        york_data = load_york_data()
        expected_participants = len(york_data)
        print(f"Expected participants: {expected_participants}")
    except Exception as e:
        print(f"Error loading York data: {str(e)}")
        return None
    
    if not results_dir.exists():
        print(f"Warning: Results directory not found at {results_dir}")
        return None
    
    # Find all JSON files in the results directory
    json_files = list(results_dir.rglob("*.json"))
    if len(json_files) == 0:
        print(f"Warning: No JSON files found in {results_dir}")
        return None
    
    print(f"Found {len(json_files)} JSON files")
    
    analysis_results = {}
    
    # Process each JSON file
    for json_file in json_files:
        # Skip metadata files
        if "metadata" in str(json_file):
            continue
            
        # Determine if this is a scenario file
        if scenario_type not in str(json_file):
            continue
            
        # Extract model name from filename
        filename = json_file.stem
        if scenario_type == 'moral':
            model_name = filename.replace('moral_', '').replace('_temp1.0', '').replace('_temp1', '')
        elif scenario_type == 'risk':
            model_name = filename.replace('risk_', '').replace('_temp1.0', '').replace('_temp1', '')
        else:
            continue
        
        print(f"\n--- Analyzing {model_name} ---")
        
        try:
            # Load JSON data
            with open(json_file, 'r') as f:
                results = json.load(f)
            
            total_responses = len(results)
            print(f"Total responses: {total_responses}")
            
            # Validate each response
            failed_indices = []
            valid_responses = 0
            
            for idx, response in enumerate(results):
                if isinstance(response, dict) and 'error' in response:
                    # This is an error response
                    failed_indices.append(idx)
                else:
                    # This is a regular response, validate it
                    if scenario_type == 'moral':
                        is_valid, errors = validate_moral_response(response)
                    elif scenario_type == 'risk':
                        is_valid, errors = validate_risk_response(response)
                    else:
                        continue
                    
                    if is_valid:
                        valid_responses += 1
                    else:
                        failed_indices.append(idx)
            
            print(f"Valid responses: {valid_responses}")
            print(f"Failed responses: {len(failed_indices)}")
            
            # Store analysis
            analysis_results[model_name] = {
                'file_path': json_file,
                'total_responses': total_responses,
                'valid_responses': valid_responses,
                'failed_indices': failed_indices,
                'success_rate': (valid_responses / total_responses * 100) if total_responses > 0 else 0
            }
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {str(e)}")
            continue
    
    return analysis_results

def recover_failed_responses_for_model(model_info: Dict[str, Any], 
                                     scenario_type: str,
                                     personality_format: str,
                                     dry_run: bool = False) -> Dict[str, Any]:
    """
    Recover failed responses for a specific model.
    """
    model_name = [k for k in model_info.keys() if k != 'file_path'][0]
    info = model_info[model_name]
    
    print(f"\n{'='*60}")
    print(f"RECOVERING FAILED RESPONSES FOR {model_name.upper()}")
    print(f"{'='*60}")
    
    if len(info['failed_indices']) == 0:
        print("No failed responses - skipping recovery")
        return {'status': 'no_recovery_needed'}
    
    if dry_run:
        print(f"DRY RUN: Would recover {len(info['failed_indices'])} responses")
        print(f"Estimated API calls: {len(info['failed_indices'])}")
        return {'status': 'dry_run'}
    
    # Load original data and results
    york_data = load_york_data()
    with open(info['file_path'], 'r') as f:
        original_results = json.load(f)
    
    # Extract model name for API call
    api_model = model_name.replace('gpt-4', 'gpt-4').replace('gpt-4o', 'gpt-4o').replace('llama', 'llama').replace('deepseek', 'deepseek')
    
    print(f"Attempting to recover {len(info['failed_indices'])} responses")
    print(f"Using model: {api_model}")
    print(f"Using scenario: {scenario_type}")
    print(f"Using personality format: {personality_format}")
    
    # Prepare participants data for recovery
    participants_to_recover = []
    for idx in info['failed_indices']:
        if idx < len(york_data):
            participant_data = york_data.iloc[idx].to_dict()
            participants_to_recover.append({
                'original_index': idx,
                'participant_data': participant_data
            })
    
    if not participants_to_recover:
        print("No valid participants to recover")
        return {'status': 'no_valid_participants'}
    
    # Re-simulate failed responses
    print(f"Running re-simulation for {len(participants_to_recover)} participants...")
    
    recovered_count = 0
    recovery_results = []
    
    for i, participant_info in enumerate(participants_to_recover):
        idx = participant_info['original_index']
        participant_data = participant_info['participant_data']
        
        print(f"  Recovering participant {idx + 1}/{len(participants_to_recover)} (original index: {idx})")
        
        try:
            # Get personality description based on format
            personality = participant_data[personality_format]
            
            # Generate scenario prompt based on type
            if scenario_type == 'moral':
                prompt = get_moral_prompt(personality)
                validate_func = validate_moral_response
            elif scenario_type == 'risk':
                prompt = get_risk_prompt(personality)
                validate_func = validate_risk_response
            else:
                print(f"    Unknown scenario type: {scenario_type}")
                recovery_results.append(original_results[idx])
                continue
            
            # Get response from model
            response = get_model_response(api_model, prompt, temperature=1.0)
            
            # Parse and validate JSON response
            if isinstance(response, str):
                extracted_jsons = extract_jsons(response)

                if extracted_jsons:
                    # Use the first valid JSON found
                    valid_found = False
                    for json_obj in extracted_jsons:
                        is_valid, validation_msg = validate_func(json_obj)
                        if is_valid:
                            original_results[idx] = json_obj
                            recovered_count += 1
                            valid_found = True
                            print(f"    ✓ Successfully recovered")
                            break

                    if not valid_found:
                        # If no valid JSON, keep the original error
                        print(f"    ✗ No valid JSON found in response")
                        recovery_results.append(original_results[idx])
                else:
                    print(f"    ✗ No JSON found in response")
                    recovery_results.append(original_results[idx])
            else:
                print(f"    ✗ Unexpected response type: {type(response)}")
                recovery_results.append(original_results[idx])
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"    ✗ Error during recovery: {str(e)}")
            recovery_results.append(original_results[idx])
    
    print(f"Successfully recovered {recovered_count} responses")
    
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

def main():
    """Main recovery function."""
    parser = argparse.ArgumentParser(description="Recover failed responses from Study 4 simulation results")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, don't perform recovery")
    parser.add_argument("--format", choices=['bfi_expanded', 'bfi_likert', 'bfi_binary_elaborated', 'bfi_binary_simple', 'all'], 
                       default='all', help="Personality format to recover (default: all)")
    parser.add_argument("--scenario", choices=['moral', 'risk', 'all'], 
                       default='all', help="Scenario type to recover (default: all)")
    
    args = parser.parse_args()
    
    # Define base directory
    base_dir = Path("study_4_generalized_results")
    
    if not base_dir.exists():
        print(f"Results directory not found: {base_dir}")
        return
    
    print("="*80)
    print("RECOVER FAILED RESPONSES FOR STUDY 4")
    print("="*80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'RECOVERY'}")
    print(f"Format: {args.format}")
    print(f"Scenario: {args.scenario}")
    
    # Find all format directories
    format_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.endswith('_format')]
    
    if not format_dirs:
        print("No format directories found")
        return
    
    # Filter formats based on argument
    if args.format != 'all':
        format_dirs = [d for d in format_dirs if args.format in d.name]
    
    print(f"Found format directories: {[d.name for d in format_dirs]}")
    
    # Step 1: Analyze all formats to identify failed responses
    all_analyses = {}
    for format_dir in format_dirs:
        format_name = format_dir.name
        print(f"\nProcessing format: {format_name}")
        
        # Determine personality format from directory name
        if 'likert' in format_name:
            personality_format = 'bfi_likert'
        elif 'expanded' in format_name:
            personality_format = 'bfi_expanded'
        elif 'binary_elaborated' in format_name:
            personality_format = 'bfi_binary_elaborated'
        elif 'binary_simple' in format_name:
            personality_format = 'bfi_binary_simple'
        else:
            print(f"Unknown format type: {format_name}")
            continue
        
        format_analyses = {}
        
        # Analyze each scenario type
        scenario_types = ['moral', 'risk'] if args.scenario == 'all' else [args.scenario]
        
        for scenario_type in scenario_types:
            scenario_dir = format_dir / scenario_type
            if scenario_dir.exists():
                analysis = analyze_results_for_failed_responses(scenario_dir, scenario_type)
                if analysis:
                    format_analyses[scenario_type] = {
                        'personality_format': personality_format,
                        'analysis': analysis
                    }
        
        if format_analyses:
            all_analyses[format_name] = format_analyses
    
    if not all_analyses:
        print("\nNo valid analyses found. Exiting.")
        return
    
    # Step 2: Summary of failed responses
    print(f"\n{'='*80}")
    print("SUMMARY OF FAILED RESPONSES")
    print(f"{'='*80}")
    
    total_failed = 0
    for format_name, format_data in all_analyses.items():
        print(f"\n{format_name}:")
        for scenario_type, scenario_data in format_data.items():
            print(f"  {scenario_type}:")
            for model_name, model_info in scenario_data['analysis'].items():
                failed = len(model_info['failed_indices'])
                total_failed += failed
                success_rate = model_info['success_rate']
                print(f"    {model_name}: {failed} failed responses ({success_rate:.1f}% success rate)")
    
    print(f"\nTotal failed responses across all formats: {total_failed}")
    
    if total_failed == 0:
        print("✓ All formats have 100% success rate!")
        return
    
    # Step 3: Recover failed responses
    if not args.dry_run:
        print(f"\n{'='*80}")
        print("BEGINNING RECOVERY PROCESS")
        print(f"{'='*80}")
        
        recovery_summary = {}
        for format_name, format_data in all_analyses.items():
            recovery_summary[format_name] = {}
            
            for scenario_type, scenario_data in format_data.items():
                recovery_summary[format_name][scenario_type] = {}
                personality_format = scenario_data['personality_format']
                
                for model_name, model_info in scenario_data['analysis'].items():
                    if len(model_info['failed_indices']) > 0:
                        result = recover_failed_responses_for_model(
                            {model_name: model_info},
                            scenario_type,
                            personality_format,
                            dry_run=args.dry_run
                        )
                        recovery_summary[format_name][scenario_type][model_name] = result
        
        # Final summary
        print(f"\n{'='*80}")
        print("RECOVERY COMPLETE")
        print(f"{'='*80}")
        
        total_recovered = 0
        for format_name, format_results in recovery_summary.items():
            print(f"\n{format_name}:")
            for scenario_type, scenario_results in format_results.items():
                print(f"  {scenario_type}:")
                for model_name, result in scenario_results.items():
                    if result['status'] == 'success':
                        recovered = result['recovered_count']
                        attempted = result['attempted_count']
                        total_recovered += recovered
                        print(f"    {model_name}: {recovered}/{attempted} recovered")
                    else:
                        print(f"    {model_name}: {result['status']}")
        
        print(f"\nTotal responses recovered: {total_recovered}")
        print(f"Success rate: {100 * total_recovered / total_failed:.1f}%")
        
        print(f"\nRecommendation: Re-run the analysis to verify recovery")
    
    else:
        print(f"\nDRY RUN COMPLETE - no recovery performed")

if __name__ == "__main__":
    main() 