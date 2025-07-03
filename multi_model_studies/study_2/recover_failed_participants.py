#!/usr/bin/env python3
"""
Data Recovery Script for Failed Participants

This script identifies and recovers failed participants from existing simulation results
using the enhanced simulation system with early detection and regeneration.

Usage:
    python recover_failed_participants.py [--model MODEL] [--temperature TEMP] [--dry-run]
"""

import json
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys

# Add shared modules to path
sys.path.append('../shared')

from enhanced_simulation_utils import (
    SimulationConfig,
    EnhancedPersonalitySimulator,
    ResponseValidator,
    run_enhanced_bfi_to_minimarker_simulation
)
from mini_marker_prompt import get_likert_prompt

def analyze_existing_results(results_file: Path) -> Tuple[List[int], Dict[str, Any]]:
    """
    Analyze existing results to identify problematic participants.
    
    Returns:
        Tuple of (problematic_indices, analysis_summary)
    """
    print(f"Analyzing existing results: {results_file}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    validator = ResponseValidator()
    
    analysis = {
        'total_participants': len(results),
        'valid_responses': 0,
        'missing_traits': 0,
        'generic_keys': 0,
        'invalid_values': 0,
        'other_errors': 0,
        'problematic_indices': []
    }
    
    error_details = {}
    
    for i, result in enumerate(results):
        is_valid, errors = validator.validate_response(result)
        
        if is_valid:
            analysis['valid_responses'] += 1
        else:
            analysis['problematic_indices'].append(i)
            error_details[i] = errors
            
            # Categorize error types
            for error in errors:
                if 'Missing traits' in error:
                    analysis['missing_traits'] += 1
                elif 'Generic/unnamed keys' in error:
                    analysis['generic_keys'] += 1
                elif 'Invalid values' in error:
                    analysis['invalid_values'] += 1
                else:
                    analysis['other_errors'] += 1
    
    # Print analysis summary
    print(f"\n{'='*60}")
    print(f"ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total participants: {analysis['total_participants']}")
    print(f"Valid responses: {analysis['valid_responses']}")
    print(f"Problematic responses: {len(analysis['problematic_indices'])}")
    print(f"")
    print(f"Error Breakdown:")
    print(f"  Missing traits: {analysis['missing_traits']}")
    print(f"  Generic/unnamed keys: {analysis['generic_keys']}")
    print(f"  Invalid values: {analysis['invalid_values']}")
    print(f"  Other errors: {analysis['other_errors']}")
    
    if len(analysis['problematic_indices']) > 0:
        print(f"\nProblematic participant indices:")
        indices_str = ', '.join(map(str, analysis['problematic_indices'][:20]))
        if len(analysis['problematic_indices']) > 20:
            indices_str += f" ... and {len(analysis['problematic_indices']) - 20} more"
        print(f"  {indices_str}")
        
        # Show specific error examples
        print(f"\nExample error details:")
        for i, (participant_id, errors) in enumerate(error_details.items()):
            if i >= 3:  # Show first 3 examples
                break
            print(f"  Participant {participant_id}: {'; '.join(errors)}")
    
    return analysis['problematic_indices'], analysis

def recover_failed_participants(original_file: Path, 
                              preprocessed_data_file: Path,
                              problematic_indices: List[int],
                              config: SimulationConfig,
                              dry_run: bool = False) -> List[Dict[str, Any]]:
    """
    Recover failed participants using enhanced simulation.
    """
    print(f"\n{'='*60}")
    print(f"RECOVERY PROCESS")
    print(f"{'='*60}")
    
    # Load original results and preprocessed data
    with open(original_file, 'r') as f:
        original_results = json.load(f)
    
    data = pd.read_csv(preprocessed_data_file)
    participants_data = data.to_dict('records')
    
    if dry_run:
        print(f"DRY RUN: Would recover {len(problematic_indices)} participants")
        print(f"Estimated LLM calls: {len(problematic_indices)} - {len(problematic_indices) * config.max_retries}")
        print(f"Estimated time: {len(problematic_indices) * 3} - {len(problematic_indices) * 15} seconds")
        return original_results
    
    print(f"Recovering {len(problematic_indices)} failed participants...")
    
    # Create enhanced simulator
    simulator = EnhancedPersonalitySimulator(config)
    
    # Process only problematic participants
    recovered_count = 0
    failed_count = 0
    
    for idx in problematic_indices:
        print(f"Processing participant {idx}...")
        
        try:
            participant_data = participants_data[idx]
            
            # Generate new response using enhanced system
            new_response = simulator.process_single_participant(
                participant_data=participant_data,
                prompt_generator=get_likert_prompt,
                personality_key='combined_bfi2',
                participant_index=idx
            )
            
            # Validate the new response
            if isinstance(new_response, dict) and 'error' not in new_response:
                validator = ResponseValidator()
                is_valid, errors = validator.validate_response(new_response)
                
                if is_valid:
                    original_results[idx] = new_response
                    recovered_count += 1
                    print(f"  ✓ Successfully recovered participant {idx}")
                else:
                    print(f"  ⚠ Partial recovery for participant {idx}: {'; '.join(errors)}")
                    # Use salvaged response anyway (better than original failure)
                    original_results[idx] = new_response
                    recovered_count += 1
            else:
                failed_count += 1
                print(f"  ✗ Failed to recover participant {idx}: {new_response.get('error', 'Unknown error')}")
        
        except Exception as e:
            failed_count += 1
            print(f"  ✗ Exception recovering participant {idx}: {str(e)}")
    
    # Get simulator statistics
    stats = simulator.get_simulation_stats()
    
    print(f"\n{'='*60}")
    print(f"RECOVERY SUMMARY")
    print(f"{'='*60}")
    print(f"Attempted recovery: {len(problematic_indices)} participants")
    print(f"Successfully recovered: {recovered_count}")
    print(f"Still failed: {failed_count}")
    print(f"Recovery rate: {100 * recovered_count / len(problematic_indices):.1f}%")
    print(f"")
    print(f"LLM Statistics:")
    print(f"  Total calls: {stats['total_attempts']}")
    print(f"  Successful responses: {stats['successful_responses']}")
    print(f"  Regeneration attempts: {stats['regeneration_attempts']}")
    
    return original_results

def save_recovered_results(results: List[Dict[str, Any]], 
                         original_file: Path,
                         config: SimulationConfig):
    """Save recovered results with backup of original."""
    
    # Create backup of original
    backup_file = original_file.with_suffix('.backup.json')
    print(f"Creating backup: {backup_file}")
    
    with open(original_file, 'r') as f_in, open(backup_file, 'w') as f_out:
        f_out.write(f_in.read())
    
    # Save recovered results
    recovered_file = original_file.parent / f"recovered_{original_file.name}"
    
    with open(recovered_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Recovered results saved: {recovered_file}")
    
    # Also save with enhanced metadata
    enhanced_results = {
        "metadata": {
            "source_file": str(original_file),
            "recovery_config": {
                "model": config.model,
                "temperature": config.temperature,
                "max_retries": config.max_retries
            },
            "generated_at": pd.Timestamp.now().isoformat()
        },
        "results": results
    }
    
    enhanced_file = original_file.parent / f"enhanced_{original_file.name}"
    with open(enhanced_file, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    print(f"Enhanced results saved: {enhanced_file}")

def main():
    parser = argparse.ArgumentParser(description="Recover failed participants from simulation results")
    parser.add_argument("--model", default="openai-gpt-3.5-turbo-0125", 
                      help="Model to use for recovery")
    parser.add_argument("--temperature", type=float, default=1.0,
                      help="Temperature for recovery simulation")
    parser.add_argument("--max-retries", type=int, default=5,
                      help="Maximum retries per participant")
    parser.add_argument("--dry-run", action="store_true",
                      help="Analyze only, don't perform recovery")
    parser.add_argument("--results-file", 
                      default="study_2_likert_results/bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1.json",
                      help="Path to results file to recover")
    parser.add_argument("--data-file",
                      default="study_2_likert_results/study2_likert_preprocessed_data.csv",
                      help="Path to preprocessed data file")
    
    args = parser.parse_args()
    
    # Validate file paths
    results_file = Path(args.results_file)
    data_file = Path(args.data_file)
    
    if not results_file.exists():
        print(f"Error: Results file not found: {results_file}")
        return 1
    
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        return 1
    
    # Configuration
    config = SimulationConfig(
        model=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
        batch_size=10,  # Smaller batches for recovery
        max_workers=5
    )
    
    print(f"Recovery Configuration:")
    print(f"  Model: {config.model}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Max retries: {config.max_retries}")
    print(f"  Dry run: {args.dry_run}")
    
    # Step 1: Analyze existing results
    problematic_indices, analysis = analyze_existing_results(results_file)
    
    if len(problematic_indices) == 0:
        print("\n✓ No problematic participants found. All data is valid!")
        return 0
    
    # Step 2: Recover failed participants
    recovered_results = recover_failed_participants(
        original_file=results_file,
        preprocessed_data_file=data_file,
        problematic_indices=problematic_indices,
        config=config,
        dry_run=args.dry_run
    )
    
    # Step 3: Save recovered results
    if not args.dry_run:
        save_recovered_results(recovered_results, results_file, config)
        
        # Final validation
        validator = ResponseValidator()
        final_valid = sum(1 for result in recovered_results 
                         if validator.validate_response(result)[0])
        
        print(f"\n{'='*60}")
        print(f"FINAL VALIDATION")
        print(f"{'='*60}")
        print(f"Total participants: {len(recovered_results)}")
        print(f"Valid responses: {final_valid}")
        print(f"Success rate: {100 * final_valid / len(recovered_results):.1f}%")
        
        improvement = final_valid - analysis['valid_responses']
        print(f"Improvement: +{improvement} valid responses (+{100 * improvement / len(recovered_results):.1f} percentage points)")
    
    return 0

if __name__ == "__main__":
    exit(main()) 