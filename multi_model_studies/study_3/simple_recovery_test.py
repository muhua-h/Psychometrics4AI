#!/usr/bin/env python3
"""
Simple Recovery Test Script

This is a simplified version to test the recovery logic without all the complexity.
"""

import pandas as pd
import json
import sys
from pathlib import Path

# Add shared modules to path
sys.path.append('../shared')

def test_format_conversion():
    """Test the format conversion functions."""
    print("Testing format conversion...")
    
    # Load data
    data_path = Path('facet_lvl_simulated_data.csv')
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return False
    
    data = pd.read_csv(data_path)
    print(f"Loaded data: {data.shape}")
    
    # Test with first row
    test_row = data.iloc[0]
    print(f"Test row domains: E={test_row['bfi_e']:.2f}, A={test_row['bfi_a']:.2f}, C={test_row['bfi_c']:.2f}, N={test_row['bfi_n']:.2f}, O={test_row['bfi_o']:.2f}")
    
    try:
        # Test binary format
        from binary_baseline_prompt import generate_binary_personality_description
        
        participant_data = {
            'bfi2_e': test_row['bfi_e'],
            'bfi2_a': test_row['bfi_a'],
            'bfi2_c': test_row['bfi_c'],
            'bfi2_n': test_row['bfi_n'],
            'bfi2_o': test_row['bfi_o']
        }
        
        binary_desc = generate_binary_personality_description(participant_data)
        print(f"Binary description: {binary_desc[:100]}...")
        
        # Test expanded format
        try:
            from schema_bfi2 import expanded_scale
            print("Successfully imported expanded_scale")
            
            # Simple test - just get a few descriptions
            bfi_columns = [f"bfi{i}" for i in range(1, 6)]  # Just first 5
            descriptions = []
            
            for col in bfi_columns:
                if col in expanded_scale and col in test_row:
                    value = int(test_row[col])
                    index = value - 1
                    if 0 <= index < len(expanded_scale[col]):
                        descriptions.append(expanded_scale[col][index])
            
            expanded_desc = ' '.join(descriptions)
            print(f"Expanded description (first 5 items): {expanded_desc[:100]}...")
            
        except Exception as e:
            print(f"Error with expanded format: {e}")
            return False
        
        # Test likert format  
        try:
            from schema_bfi2 import likert_scale
            print("Successfully imported likert_scale")
            
            # Simple test
            likert_descriptions = []
            for col in bfi_columns:
                if col in likert_scale and col in test_row:
                    value = int(test_row[col])
                    likert_descriptions.append(f"{likert_scale[col]} {value};")
            
            likert_desc = ' '.join(likert_descriptions)
            print(f"Likert description (first 5 items): {likert_desc[:100]}...")
            
        except Exception as e:
            print(f"Error with likert format: {e}")
            return False
            
        print("✓ All format conversions successful!")
        return True
        
    except Exception as e:
        print(f"Error in format conversion: {e}")
        return False

def test_problematic_file():
    """Test loading and analyzing the problematic llama file."""
    print("\nTesting problematic file...")
    
    problematic_file = "study_3_expanded_results/bfi_to_minimarker_llama_temp1.json"
    
    if not Path(problematic_file).exists():
        print(f"File not found: {problematic_file}")
        return False
    
    try:
        with open(problematic_file, 'r') as f:
            data = json.load(f)
        
        print(f"File loaded: {len(data)} responses")
        
        # Check first few responses
        valid_count = 0
        invalid_count = 0
        
        for i, response in enumerate(data[:10]):  # Check first 10
            if isinstance(response, dict):
                if len(response) > 30:  # Should have ~40 traits
                    valid_count += 1
                    if i == 0:
                        print(f"Sample valid response keys: {list(response.keys())[:5]}...")
                else:
                    invalid_count += 1
                    print(f"Invalid response {i}: {response}")
            else:
                invalid_count += 1
                print(f"Non-dict response {i}: {type(response)}")
        
        print(f"In first 10: {valid_count} valid, {invalid_count} invalid")
        return True
        
    except Exception as e:
        print(f"Error loading file: {e}")
        return False

if __name__ == "__main__":
    print("=== SIMPLE RECOVERY TEST ===")
    
    # Test 1: Format conversion
    format_ok = test_format_conversion()
    
    # Test 2: Problematic file
    file_ok = test_problematic_file()
    
    if format_ok and file_ok:
        print("\n✓ All tests passed! The issue might be in the recovery logic or API calls.")
    else:
        print("\n✗ Some tests failed. This indicates the problem.") 