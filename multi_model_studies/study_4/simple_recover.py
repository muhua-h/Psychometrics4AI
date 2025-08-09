#!/usr/bin/env python3
"""
Simple Recovery Script for Study 4 Generalized Framework

This script recovers the missing 47 participants in generalized framework simulations.
"""

import pandas as pd
import json
import sys
import time
from pathlib import Path

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

def prepare_personality_description(data_row):
    """Prepare personality description for all formats"""
    # Extract BFI-2 scores
    personality_str = f"Big Five personality: "
    personality_str += f"Extraversion={data_row.get('bfi_e', 'N/A')}, "
    personality_str += f"Agreeableness={data_row.get('bfi_a', 'N/A')}, "
    personality_str += f"Conscientiousness={data_row.get('bfi_c', 'N/A')}, "
    personality_str += f"Neuroticism={data_row.get('bfi_n', 'N/A')}, "
    personality_str += f"Openness={data_row.get('bfi_o', 'N/A')}"
    
    return personality_str

def recover_missing_participants(model_name, scenario_type, personality_format):
    """Recover missing participants for a specific model and format"""
    data = load_york_data()
    
    # Construct file path
    results_dir = Path(f'study_4_generalized_results/{personality_format}_format')
    scenario_dir = results_dir / scenario_type
    filename = f"{scenario_type}_{model_name}_temp1.0.json"
    file_path = scenario_dir / filename
    
    print(f"\nProcessing {model_name} - {scenario_type} - {personality_format}")
    print(f"File: {file_path}")
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    # Load existing results
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    print(f"Current responses: {len(results)}")
    
    if len(results) == 337:
        print("Already complete!")
        return
    
    # Determine missing participants
    missing_count = 337 - len(results)
    print(f"Missing participants: {missing_count}")
    
    # Get missing indices
    existing_indices = set(range(len(results)))  # 0-289
    all_indices = set(range(337))
    missing_indices = sorted(all_indices - existing_indices)
    
    print(f"Missing indices: {len(missing_indices)}")
    
    # Prepare recovery data
    participants_to_recover = []
    for idx in missing_indices:
        if idx < len(data):
            data_row = data.iloc[idx]
            personality = prepare_personality_description(data_row)
            participants_to_recover.append({
                'index': idx,
                'personality': personality,
                'data': data_row.to_dict()
            })
    
    print(f"Prepared {len(participants_to_recover)} participants for recovery")
    
    # Determine prompt function
    prompt_func = get_moral_prompt if scenario_type == 'moral' else get_risk_prompt
    
    # Recover participants
    recovered_count = 0
    for participant in participants_to_recover:
        try:
            prompt = prompt_func(participant['personality'])
            response = get_model_response(model_name, prompt, temperature=0.0)
            
            # Parse response
            if isinstance(response, str):
                import re
                json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
                matches = re.findall(json_pattern, response, re.DOTALL)
                if matches:
                    try:
                        response_dict = json.loads(matches[0])
                    except:
                        response_dict = {"error": "Invalid JSON format"}
                else:
                    response_dict = {"error": "No JSON found"}
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {"error": f"Unexpected type: {type(response)}"}
            
            # Add to results
            results.append(response_dict)
            recovered_count += 1
            print(f"  ✓ Recovered participant {participant['index']}")
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
            
        except Exception as e:
            print(f"  ✗ Failed participant {participant['index']}: {e}")
            results.append({"error": str(e)})
    
    print(f"Recovered {recovered_count}/{missing_count} participants")
    
    # Save updated results
    backup_path = file_path.with_suffix('.backup.json')
    with open(backup_path, 'w') as f:
        json.dump(results[:290], f, indent=2)  # Backup original 290
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Updated file saved: {file_path}")

def main():
    """Main recovery function"""
    print("=== Study 4 Simple Recovery ===")
    
    models = ['gpt-4', 'gpt-4o', 'llama', 'deepseek', 'openai-gpt-3.5-turbo-0125']
    scenarios = ['moral', 'risk']
    formats = ['bfi_expanded', 'bfi_likert', 'bfi_binary_elaborated', 'bfi_binary_simple']
    
    data = load_york_data()
    print(f"Total participants expected: {len(data)}")
    
    # Recovery summary
    total_recovered = 0
    
    for model in models:
        for scenario in scenarios:
            for format_name in formats:
                try:
                    recover_missing_participants(model, scenario, format_name)
                    total_recovered += 47  # Expected per format/model
                except Exception as e:
                    print(f"Error with {model}-{scenario}-{format_name}: {e}")
    
    print(f"\nRecovery complete! Total expected recoveries: {total_recovered}")

if __name__ == "__main__":
    # Test a single case
    recover_missing_participants('gpt-4', 'moral', 'bfi_expanded')