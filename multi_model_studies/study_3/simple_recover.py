#!/usr/bin/env python3
"""
Simplified Recovery Script for Study 3

This focuses just on recovering the network error responses in the llama file.
"""

import pandas as pd
import json
import sys
from pathlib import Path

# Add shared modules to path
sys.path.append('../shared')
from simulation_utils import run_batch_simulation, SimulationConfig
from mini_marker_prompt import get_expanded_prompt
from schema_bfi2 import expanded_scale

def create_expanded_description(data_row: pd.Series) -> str:
    """Create expanded format description from BFI-2 data row."""
    # BFI columns (all 60 items)
    bfi_columns = [f"bfi{i}" for i in range(1, 61)]
    
    # Apply reverse coding (same as in Study 3 notebook)
    reverse_coding_map = {
        'bfi1': 'bfi1', 'bfi2': 'bfi2', 'bfi3': 'bfi3R', 'bfi4': 'bfi4R', 'bfi5': 'bfi5R',
        'bfi6': 'bfi6', 'bfi7': 'bfi7', 'bfi8': 'bfi8R', 'bfi9': 'bfi9R', 'bfi10': 'bfi10',
        'bfi11': 'bfi11R', 'bfi12': 'bfi12R', 'bfi13': 'bfi13', 'bfi14': 'bfi14', 'bfi15': 'bfi15',
        'bfi16': 'bfi16R', 'bfi17': 'bfi17R', 'bfi18': 'bfi18', 'bfi19': 'bfi19', 'bfi20': 'bfi20',
        'bfi21': 'bfi21', 'bfi22': 'bfi22R', 'bfi23': 'bfi23R', 'bfi24': 'bfi24R', 'bfi25': 'bfi25R',
        'bfi26': 'bfi26R', 'bfi27': 'bfi27', 'bfi28': 'bfi28R', 'bfi29': 'bfi29R', 'bfi30': 'bfi30R',
        'bfi31': 'bfi31R', 'bfi32': 'bfi32', 'bfi33': 'bfi33', 'bfi34': 'bfi34', 'bfi35': 'bfi35',
        'bfi36': 'bfi36R', 'bfi37': 'bfi37R', 'bfi38': 'bfi38', 'bfi39': 'bfi39', 'bfi40': 'bfi40',
        'bfi41': 'bfi41', 'bfi42': 'bfi42R', 'bfi43': 'bfi43', 'bfi44': 'bfi44R', 'bfi45': 'bfi45R',
        'bfi46': 'bfi46', 'bfi47': 'bfi47R', 'bfi48': 'bfi48R', 'bfi49': 'bfi49R', 'bfi50': 'bfi50R',
        'bfi51': 'bfi51R', 'bfi52': 'bfi52', 'bfi53': 'bfi53', 'bfi54': 'bfi54', 'bfi55': 'bfi55R',
        'bfi56': 'bfi56', 'bfi57': 'bfi57', 'bfi58': 'bfi58R', 'bfi59': 'bfi59', 'bfi60': 'bfi60'
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

def recover_llama_errors():
    """Recover the network error responses in the llama file."""
    print("=== RECOVERING LLAMA ERROR RESPONSES ===")
    
    # Load data
    data_path = Path('facet_lvl_simulated_data.csv')
    data = pd.read_csv(data_path)
    print(f"Loaded data: {data.shape}")
    
    # Load problematic file
    llama_file = "study_3_expanded_results/bfi_to_minimarker_llama_temp1.json"
    with open(llama_file, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded results: {len(results)} responses")
    
    # Find error responses
    error_indices = []
    for i, response in enumerate(results):
        if isinstance(response, dict) and 'error' in response:
            error_indices.append(i)
    
    print(f"Found {len(error_indices)} error responses: {error_indices}")
    
    if len(error_indices) == 0:
        print("No error responses to recover!")
        return
    
    # Prepare recovery data
    recovery_data = []
    for idx in error_indices:
        if idx < len(data):
            data_row = data.iloc[idx]
            
            # Create expanded description
            combined_bfi2 = create_expanded_description(data_row)
            
            participant = {
                'participant_id': idx,
                'combined_bfi2': combined_bfi2
            }
            recovery_data.append(participant)
    
    print(f"Prepared {len(recovery_data)} participants for recovery")
    
    # Run recovery simulation
    config = SimulationConfig(
        model='llama',
        temperature=1.0,
        batch_size=5,  # Small batch size
        max_workers=5
    )
    
    print("Running recovery simulation...")
    try:
        recovery_results = run_batch_simulation(
            participants_data=recovery_data,
            prompt_generator=get_expanded_prompt,
            config=config,
            personality_key='combined_bfi2'
        )
        
        print(f"Recovery simulation completed: {len(recovery_results)} results")
        
        # Update original results
        recovered_count = 0
        for i, result in enumerate(recovery_results):
            if 'error' not in result:
                original_index = error_indices[i]
                results[original_index] = result
                recovered_count += 1
                print(f"  ✓ Recovered participant {original_index}")
            else:
                print(f"  ✗ Still failed participant {error_indices[i]}: {result.get('error', 'Unknown error')}")
        
        print(f"\nSuccessfully recovered {recovered_count}/{len(error_indices)} participants")
        
        # Save backup and updated results
        backup_file = llama_file.replace('.json', '.backup.json')
        print(f"Creating backup: {backup_file}")
        
        with open(backup_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save updated results
        with open(llama_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Updated results saved to: {llama_file}")
        print("✓ Recovery complete!")
        
    except Exception as e:
        print(f"Error during recovery: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    recover_llama_errors() 