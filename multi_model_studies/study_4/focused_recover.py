#!/usr/bin/env python3
"""
Focused Recovery Script for Study 4 Generalized Framework

This script identifies and recovers exactly the 47 missing participants 
in the generalized framework simulations.
"""

import pandas as pd
import json
import sys
import time
from pathlib import Path
import glob

# Add shared modules to path
sys.path.append('../shared')
sys.path.append('../../')
from portal import get_model_response
from moral_stories import get_prompt as get_moral_prompt
from risk_taking import get_prompt as get_risk_prompt

def load_york_data():
    """Load and preprocess York behavioral data"""
    data_path = Path('../../raw_data/york_data_clean.csv')
    data = pd.read_csv(data_path)
    
    # Apply same filtering as in simulation scripts
    data = data[data['8) English language reading/comprehension ability:'] >= 4]
    data = data.dropna(subset=[data.columns[17]])  # bfi6 column
    
    return data

def prepare_personality_description(data_row, personality_format):
    """Prepare personality description based on format"""
    if personality_format == 'bfi_expanded':
        from schema_bfi2 import expanded_scale
        # Use expanded format description
        bfi_columns = [f"bfi{i}" for i in range(1, 61)]
        reverse_coding_map = {
            # Add reverse coding mapping here
            'bfi1': 'bfi1', 'bfi2': 'bfi2', 'bfi3': 'bfi3R', 'bfi4': 'bfi4R', 'bfi5': 'bfi5R',
            'bfi6': 'bfi6', 'bfi7': 'bfi7', 'bfi8': 'bfi8R', 'bfi9': 'bfi9R', 'bfi10': 'bfi10',
            'bfi11': 'bfi11R', 'bfi12': 'bfi12R', 'bfi13': 'bfi13', 'bfi14': 'bfi14', 'bfi15': 'bfi15',
            'bfi16': 'bfi16R', 'bfi17': 'bfi17R', 'bfi18': 'bfi18', 'bfi19': 'bfi19', 'bfi20': 'bfi20',
            'bfi21': 'bfi21', 'bfi22': 'bfi22R', 'bfi23': 'bfi23R', 'bfi24': 'bfi24R', 'bfi25': 'bfi25R',
            'bfi26': 'bfi26R', 'bfi27': 'bfi27', 'bfi28': 'bfi28R', 'bfi29': 'bfi29R', 'bfi30': 'bfi30R',
            'bfi31': 'bfi31R', 'bfi32': 'bfi32', 'bfi33': 'bfi33', 'bfi34': 'bfi34', 'bfi35': 'bfi35R',
            'bfi36': 'bfi36', 'bfi37': 'bfi37R', 'bfi38': 'bfi38R', 'bfi39': 'bfi39', 'bfi40': 'bfi40',
            'bfi41': 'bfi41', 'bfi42': 'bfi42R', 'bfi43': 'bfi43', 'bfi44': 'bfi44R', 'bfi45': 'bfi45R',
            'bfi46': 'bfi46', 'bfi47': 'bfi47R', 'bfi48': 'bfi48R', 'bfi49': 'bfi49R', 'bfi50': 'bfi50R',
            'bfi51': 'bfi51R', 'bfi52': 'bfi52', 'bfi53': 'bfi53', 'bfi54': 'bfi54', 'bfi55': 'bfi55',
            'bfi56': 'bfi56', 'bfi57': 'bfi57', 'bfi58': 'bfi58R', 'bfi59': 'bfi59', 'bfi60': 'bfi60'
        }
        
        descriptions = []
        for col in bfi_columns:
            if col in expanded_scale and col in data_row:
                value = int(data_row[col])
                if col in reverse_coding_map and reverse_coding_map[col].endswith('R'):
                    value = 6 - value  # Reverse coding for 1-5 scale
                index = value - 1
                if 0 <= index < len(expanded_scale[col]):
                    descriptions.append(expanded_scale[col][index])
        
        return ' '.join(descriptions)
    
    elif personality_format == 'bfi_likert':
        from schema_bfi2 import likert_scale
        descriptions = []
        for col in [f"bfi{i}" for i in range(1, 61)]:
            if col in likert_scale and col in data_row:
                value = int(data_row[col])
                descriptions.append(f"{likert_scale[col]} {value};")
        return ' '.join(descriptions)
    
    elif personality_format == 'bfi_binary_elaborated':
        # Use simple binary descriptions
        from mini_marker_prompt import generate_binary_personality_description
        
        # Create participant data structure
        participant_data = {
            'bfi2_e': data_row.get('bfi_e', 3),
            'bfi2_a': data_row.get('bfi_a', 3),
            'bfi2_c': data_row.get('bfi_c', 3),
            'bfi2_n': data_row.get('bfi_n', 3),
            'bfi2_o': data_row.get('bfi_o', 3)
        }
        
        return generate_binary_personality_description(participant_data)
    
    else:  # bfi_binary_simple
        from mini_marker_prompt import generate_binary_personality_description
        participant_data = {
            'bfi2_e': data_row.get('bfi_e', 3),
            'bfi2_a': data_row.get('bfi_a', 3),
            'bfi2_c': data_row.get('bfi_c', 3),
            'bfi2_n': data_row.get('bfi_n', 3),
            'bfi2_o': data_row.get('bfi_o', 3)
        }
        return generate_binary_personality_description(participant_data)

def recover_single_file(model_name, scenario_type, personality_format):
    """Recover missing participants for a single file"""
    data = load_york_data()
    
    # File path
    results_dir = Path(f'study_4_generalized_results/{personality_format}_format')
    scenario_dir = results_dir / scenario_type
    filename = f"{scenario_type}_{model_name}_temp1.0.json"
    file_path = scenario_dir / filename
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return 0
    
    # Load existing results
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    current_count = len(results)
    expected_count = 337
    
    if current_count == expected_count:
        print(f"✅ {model_name} {scenario_type} {personality_format}: Already complete ({current_count})")
        return 0
    
    missing_count = expected_count - current_count
    print(f"🔄 {model_name} {scenario_type} {personality_format}: Missing {missing_count} participants")
    
    # Get missing participant indices (290-336)
    missing_indices = list(range(current_count, expected_count))
    
    # Determine prompt function
    prompt_func = get_moral_prompt if scenario_type == 'moral' else get_risk_prompt
    
    # Recover participants
    recovered_count = 0
    
    for i, idx in enumerate(missing_indices):
        if idx < len(data):
            data_row = data.iloc[idx]
            personality = prepare_personality_description(data_row, personality_format)
            
            try:
                prompt = prompt_func(personality)
                response = get_model_response(model_name, prompt, temperature=0.0)
                
                # Parse and validate response
                import re
                if isinstance(response, str):
                    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                    matches = re.findall(json_pattern, response, re.DOTALL)
                    if matches:
                        try:
                            response_dict = json.loads(matches[0])
                        except:
                            response_dict = {"error": "Invalid JSON"}
                    else:
                        response_dict = {"error": "No JSON found"}
                elif isinstance(response, dict):
                    response_dict = response
                else:
                    response_dict = {"error": f"Unexpected type: {type(response)}"}
                
                results.append(response_dict)
                recovered_count += 1
                
                if recovered_count % 5 == 0:
                    print(f"  Progress: {recovered_count}/{missing_count}")
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"  Error with participant {idx}: {e}")
                results.append({"error": str(e)})
    
    # Save updated results
    backup_path = file_path.with_suffix('.backup.json')
    with open(backup_path, 'w') as f:
        json.dump(results[:current_count], f, indent=2)
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Recovered {recovered_count} participants")
    return recovered_count

def main():
    """Main recovery function"""
    print("=== Study 4 Focused Recovery ===")
    
    models = ['gpt-4', 'gpt-4o', 'llama', 'deepseek', 'openai-gpt-3.5-turbo-0125']
    scenarios = ['moral', 'risk']
    formats = ['bfi_expanded', 'bfi_likert', 'bfi_binary_elaborated', 'bfi_binary_simple']
    
    total_recovered = 0
    total_files = len(models) * len(scenarios) * len(formats)
    processed = 0
    
    for model in models:
        for scenario in scenarios:
            for format_name in formats:
                processed += 1
                print(f"\n[{processed}/{total_files}] Processing...")
                try:
                    recovered = recover_single_file(model, scenario, format_name)
                    total_recovered += recovered
                except Exception as e:
                    print(f"  Error: {e}")
    
    print(f"\n=== Recovery Complete ===")
    print(f"Total participants recovered: {total_recovered}")
    print(f"Expected total: {total_files * 47}")

if __name__ == "__main__":
    # Test single case
    recover_single_file('gpt-4', 'moral', 'bfi_expanded')