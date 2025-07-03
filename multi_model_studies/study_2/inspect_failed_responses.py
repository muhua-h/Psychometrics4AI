#!/usr/bin/env python3
"""
Inspect the actual JSON responses that caused participant loss
"""

import json
from pathlib import Path

def inspect_problematic_responses():
    print("="*80)
    print("INSPECTING PROBLEMATIC JSON RESPONSES")
    print("="*80)
    
    # Load the JSON file with most issues
    json_file = Path('study_2_likert_results/bfi_to_minimarker_openai_gpt_3.5_turbo_0125_temp1.json')
    
    if not json_file.exists():
        print(f"File not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Total participants in JSON: {len(data)}")
    
    # Check the problematic participants identified by the diagnostic
    problematic_indices = [44, 201]  # From diagnostic output
    
    for idx in problematic_indices:
        print(f"\n{'='*50}")
        print(f"PARTICIPANT {idx}")
        print(f"{'='*50}")
        
        if idx < len(data):
            participant_data = data[idx]
            print(f"Keys in response: {len(participant_data.keys())}")
            print(f"Sample keys: {list(participant_data.keys())[:10]}")
            
            # Show the actual response structure
            print(f"\nActual JSON structure:")
            for key, value in list(participant_data.items())[:5]:
                print(f"  '{key}': {value}")
            
            # Check for missing expected traits
            expected_traits = {
                'Bashful', 'Bold', 'Careless', 'Cold', 'Complex', 'Cooperative', 
                'Creative', 'Deep', 'Disorganized', 'Efficient', 'Energetic', 
                'Envious', 'Extraverted', 'Fretful', 'Harsh', 'Imaginative', 
                'Inefficient', 'Intellectual', 'Jealous', 'Kind', 'Moody', 
                'Organized', 'Philosophical', 'Practical', 'Quiet', 'Relaxed', 
                'Rude', 'Shy', 'Sloppy', 'Sympathetic', 'Systematic', 'Talkative', 
                'Temperamental', 'Touchy', 'Uncreative', 'Unenvious', 
                'Unintellectual', 'Unsympathetic', 'Warm', 'Withdrawn'
            }
            
            found_traits = set()
            for key in participant_data.keys():
                # Clean the key (remove numbers, whitespace, etc.)
                clean_key = key.lstrip()
                if '. ' in clean_key:
                    clean_key = clean_key.split('. ', 1)[1]
                found_traits.add(clean_key)
            
            missing_traits = expected_traits - found_traits
            print(f"\nMissing traits ({len(missing_traits)}): {sorted(list(missing_traits))}")
            
            if missing_traits:
                print(f"\nThis participant will be DROPPED due to missing traits")
            else:
                print(f"\nThis participant should be OK")
        else:
            print(f"Index {idx} out of range!")

    # Also check the other problematic file
    print(f"\n{'='*80}")
    print("CHECKING OTHER PROBLEMATIC FILE")
    print(f"{'='*80}")
    
    json_file2 = Path('study_2_likert_results/bfi_to_minimarker_openai_gpt_3.5_turbo_0125.json')
    if json_file2.exists():
        with open(json_file2, 'r') as f:
            data2 = json.load(f)
        
        print(f"Total participants: {len(data2)}")
        
        # Check one of the problematic participants
        idx = 127  # From diagnostic output
        if idx < len(data2):
            participant_data = data2[idx]
            print(f"\nParticipant {idx} keys: {len(participant_data.keys())}")
            print(f"Sample response: {dict(list(participant_data.items())[:3])}")

if __name__ == "__main__":
    inspect_problematic_responses() 