#!/usr/bin/env python3
"""
Study 4 Multi-Model Risk-Taking Scenario Simulation

This script replicates the original Study 4 risk-taking simulation
with multiple LLM models using the unified portal.py interface.

Models to Test:
- GPT-4
- GPT-4o  
- Llama-3.3-70B-Instruct
- DeepSeek-V3

Data Flow:
1. Load York human behavioral data
2. Extract personality descriptions (bfi_combined)
3. Generate risk-taking scenario prompts using risk_taking.py
4. Run simulations across multiple models
5. Save results for behavioral validation analysis
"""

import pandas as pd
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add shared modules to path
sys.path.append('../shared')

from simulation_utils import SimulationConfig
from risk_taking import get_prompt
import sys
sys.path.append('../../')
from portal import get_model_response

def load_york_data():
    """Load and preprocess York behavioral data"""
    data_path = Path('../../study_4/simulation/data_w_simulation.csv')
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"Loaded York data shape: {data.shape}")
    
    # Filter for completed responses with good English comprehension
    data = data[data['Finished'] == 1]
    data = data[data['8) English language reading/comprehension ability:'] == 5]
    data = data.dropna(subset=[data.columns[17]])  # Remove null values in 18th column
    
    print(f"Filtered data shape: {data.shape}")
    return data

def process_participant_risk(participant_data, model, temperature=0.0):
    """Process a single participant for risk-taking scenarios"""
    try:
        # Get personality description
        personality = participant_data['bfi_combined']
        
        # Generate risk-taking scenario prompt
        prompt = get_prompt(personality)
        
        # Get response from model
        response = get_model_response(model, prompt, temperature=temperature)
        
        # Parse JSON response
        if isinstance(response, str):
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                try:
                    response_dict = json.loads(json_match.group())
                    return response_dict
                except json.JSONDecodeError:
                    return {"error": f"JSON parse error: {response}"}
            else:
                return {"error": f"No JSON found in response: {response}"}
        elif isinstance(response, dict):
            return response
        else:
            return {"error": f"Unexpected response type: {type(response)}"}
            
    except Exception as e:
        return {"error": str(e)}

def run_risk_simulation(participants_data, model, temperature, output_dir):
    """Run risk-taking simulation for a specific model"""
    print(f"\nStarting risk simulation: {model} with temperature {temperature}")
    
    results = []
    batch_size = 25
    
    # Process participants in batches
    for i in range(0, len(participants_data), batch_size):
        batch = participants_data[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} (participants {i+1}-{min(i+batch_size, len(participants_data))})")
        
        batch_results = []
        for participant in batch:
            result = process_participant_risk(participant, model, temperature)
            batch_results.append(result)
            time.sleep(0.1)  # Small delay to avoid rate limits
        
        results.extend(batch_results)
        print(f"Completed batch {i//batch_size + 1}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = f"risk_{model}_temp{temperature}.json"
    with open(output_path / filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Calculate success rate
    successful = sum(1 for r in results if not (isinstance(r, dict) and 'error' in r))
    success_rate = (successful / len(results)) * 100
    
    print(f"Completed {model}: {successful}/{len(results)} successful ({success_rate:.1f}%)")
    return results

def retry_failed_participants(results, participants_data, model, temperature):
    """Retry failed participants with exponential backoff"""
    print(f"Retrying failed participants for {model}")
    
    for i, result in enumerate(results):
        if isinstance(result, dict) and 'error' in result:
            print(f"Retrying participant {i}")
            for attempt in range(3):  # Max 3 retry attempts
                try:
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
                    
                    new_result = process_participant_risk(participants_data[i], model, temperature)
                    if not (isinstance(new_result, dict) and 'error' in new_result):
                        results[i] = new_result
                        print(f"Successfully retried participant {i}")
                        break
                except Exception as e:
                    print(f"Retry attempt {attempt + 1} failed for participant {i}: {str(e)}")
            
            if isinstance(results[i], dict) and 'error' in results[i]:
                print(f"All retry attempts failed for participant {i}")
    
    return results

def main():
    """Main execution function"""
    # Configuration
    models_to_test = [
        # 'openai-gpt-3.5-turbo-0125',
        "gpt-4",
        "gpt-4o",
        "llama",
        "deepseek"
    ]

    temperature = 1
    output_dir = "study_4_risk_results"
    
    print("="*60)
    print("Study 4 Multi-Model Risk-Taking Scenario Simulation")
    print("="*60)
    
    # Load data
    data = load_york_data()
    participants_data = data.to_dict('records')
    
    print(f"Prepared {len(participants_data)} participants for risk simulation")
    
    # Run simulations for all models
    all_results = {}
    
    for model in models_to_test:
        try:
            results = run_risk_simulation(participants_data, model, temperature, output_dir)
            all_results[model] = results
        except Exception as e:
            print(f"Error in simulation {model}: {str(e)}")
            all_results[model] = {"error": str(e)}
    
    # Retry failed participants
    for model, results in all_results.items():
        if isinstance(results, list):
            failed_count = sum(1 for r in results if isinstance(r, dict) and 'error' in r)
            if failed_count > 0:
                print(f"Retrying {failed_count} failed participants for {model}")
                updated_results = retry_failed_participants(results, participants_data, model, temperature)
                all_results[model] = updated_results
                
                # Save updated results
                output_path = Path(output_dir)
                filename = f"risk_{model}_temp{temperature}_retried.json"
                with open(output_path / filename, 'w') as f:
                    json.dump(updated_results, f, indent=2)
    
    # Results summary
    print("\n" + "="*50)
    print("Risk Simulation Results Summary:")
    print("="*50)
    
    for model, results in all_results.items():
        if isinstance(results, list):
            total = len(results)
            successful = sum(1 for r in results if not (isinstance(r, dict) and 'error' in r))
            failed = total - successful
            success_rate = (successful / total) * 100
            
            print(f"{model}:")
            print(f"  Total: {total}, Successful: {successful}, Failed: {failed}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print()
        else:
            print(f"{model}: FAILED - {results.get('error', 'Unknown error')}")
            print()
    
    # Save preprocessed data for reference
    output_path = Path(output_dir)
    data.to_csv(output_path / 'study4_risk_preprocessed_data.csv', index=False)
    print(f"Preprocessed data saved to {output_path / 'study4_risk_preprocessed_data.csv'}")
    
    print("\n" + "="*60)
    print("RISK SIMULATION COMPLETE!")
    print("\nNext steps:")
    print("1. Run study_4_risk_behavioral_analysis.py for validation analysis")
    print(f"2. Results are saved in {output_dir}/ directory")
    print("3. Use unified_behavioral_analysis.py for cross-scenario comparison")
    print("="*60)

if __name__ == "__main__":
    main()