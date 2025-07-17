#!/usr/bin/env python3
"""
Study 4 Multi-Model Moral Scenario Simulation

This script replicates the original Study 4 moral reasoning simulation
with multiple LLM models using the unified portal.py interface.

Models to Test:
- GPT-4
- GPT-4o  
- Llama-3.3-70B-Instruct
- DeepSeek-V3

Data Flow:
1. Load York human behavioral data
2. Extract personality descriptions (bfi_combined)
3. Generate moral scenario prompts using moral_stories.py
4. Run simulations across multiple models
5. Save results for behavioral validation analysis
"""

import pandas as pd
import sys
import json
import time
import re
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

# Add shared modules to path
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from moral_stories import get_prompt
from portal import get_model_response

# Thread-safe logging
log_lock = threading.Lock()

def safe_print(message, prefix="INFO"):
    """Thread-safe printing with timestamp and prefix"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    with log_lock:
        print(f"[{timestamp}] {prefix}: {message}")

def extract_jsons(text):
    """Extract and parse JSON objects from text response"""
    # Try to find JSON objects in the response
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)

    extracted_jsons = []
    for match in matches:
        try:
            json_obj = json.loads(match)
            extracted_jsons.append(json_obj)
        except json.JSONDecodeError:
            continue

    if not extracted_jsons:
        # Try to parse the entire response as JSON
        try:
            json_obj = json.loads(text.strip())
            extracted_jsons.append(json_obj)
        except json.JSONDecodeError:
            pass

    return extracted_jsons

def validate_moral_response(response_dict):
    """Validate that the moral response contains expected fields"""
    required_fields = ['scenario_1', 'scenario_2', 'scenario_3', 'scenario_4', 'scenario_5']

    if not isinstance(response_dict, dict):
        return False, "Response is not a dictionary"

    missing_fields = [field for field in required_fields if field not in response_dict]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"

    # Check if values are numeric (ratings)
    for field in required_fields:
        value = response_dict.get(field)
        if not isinstance(value, (int, float)) or not (1 <= value <= 7):
            return False, f"Invalid value for {field}: {value} (expected 1-7)"

    return True, "Valid response"

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def process_participant_moral_with_retry(participant_data, model, temperature=0.0):
    """Process a single participant with enhanced retry logic"""
    try:
        # Get personality description
        personality = participant_data['bfi_combined']
        
        # Generate moral scenario prompt
        prompt = get_prompt(personality)
        
        # Get response from model
        response = get_model_response(model, prompt, temperature=temperature)
        
        # Parse and validate JSON response
        if isinstance(response, str):
            extracted_jsons = extract_jsons(response)

            if extracted_jsons:
                # Use the first valid JSON found
                for json_obj in extracted_jsons:
                    is_valid, validation_msg = validate_moral_response(json_obj)
                    if is_valid:
                        return json_obj

                # If no valid JSON, return the first one with error info
                _, validation_msg = validate_moral_response(extracted_jsons[0])
                return {"error": f"Invalid response format: {validation_msg}", "raw_response": response}
            else:
                return {"error": f"No JSON found in response: {response}"}

        elif isinstance(response, dict):
            is_valid, validation_msg = validate_moral_response(response)
            if is_valid:
                return response
            else:
                return {"error": f"Invalid response format: {validation_msg}", "raw_response": response}
        else:
            return {"error": f"Unexpected response type: {type(response)}"}
            
    except Exception as e:
        # This will trigger the retry mechanism
        raise Exception(f"Processing error: {str(e)}")

def process_participant_moral(participant_data, model, temperature=0.0):
    """Process a single participant for moral reasoning scenarios with fallback"""
    try:
        return process_participant_moral_with_retry(participant_data, model, temperature)
    except Exception as e:
        # Final fallback after all retries failed
        return {"error": f"All retries failed: {str(e)}"}

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

def run_simulation_for_model(model, participants_data, temperature, output_dir):
    """Run simulation for a single model - designed for parallel execution"""
    simulation_id = f"{model}_temp{temperature}"

    # Start message
    safe_print(f"Starting simulation: {model} (temp={temperature})", "START")

    start_time = time.time()

    try:
        results = []
        batch_size = 25

        # Process participants in batches
        for i in range(0, len(participants_data), batch_size):
            batch = participants_data[i:i+batch_size]
            safe_print(f"[{model}] Processing batch {i//batch_size + 1} (participants {i+1}-{min(i+batch_size, len(participants_data))})", "BATCH")

            batch_results = []
            for participant in batch:
                result = process_participant_moral(participant, model, temperature)
                batch_results.append(result)
                time.sleep(0.1)  # Small delay to avoid rate limits

            results.extend(batch_results)
            safe_print(f"[{model}] Completed batch {i//batch_size + 1}", "BATCH")

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        filename = f"moral_{model}_temp{temperature}.json"
        with open(output_path / filename, 'w') as f:
            json.dump(results, f, indent=2)

        # Calculate success rate
        successful = sum(1 for r in results if not (isinstance(r, dict) and 'error' in r))
        failed_count = len(results) - successful
        success_rate = (successful / len(results)) * 100
        duration = time.time() - start_time

        if failed_count > 0:
            safe_print(f"Completed {simulation_id} in {duration:.1f}s - WARNING: {failed_count} participants failed", "WARN")
        else:
            safe_print(f"Completed {simulation_id} in {duration:.1f}s - All participants successful", "SUCCESS")

        return (simulation_id, results)

    except Exception as e:
        duration = time.time() - start_time
        safe_print(f"Failed {simulation_id} after {duration:.1f}s - Error: {str(e)}", "ERROR")
        return (simulation_id, {"error": str(e)})

def run_moral_simulation(participants_data, model, temperature, output_dir):
    """Run moral reasoning simulation for a specific model"""
    print(f"\nStarting moral simulation: {model} with temperature {temperature}")
    
    results = []
    batch_size = 25
    
    # Process participants in batches
    for i in range(0, len(participants_data), batch_size):
        batch = participants_data[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1} (participants {i+1}-{min(i+batch_size, len(participants_data))})")
        
        batch_results = []
        for participant in batch:
            result = process_participant_moral(participant, model, temperature)
            batch_results.append(result)
            time.sleep(0.1)  # Small delay to avoid rate limits
        
        results.extend(batch_results)
        print(f"Completed batch {i//batch_size + 1}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = f"moral_{model}_temp{temperature}.json"
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
                    
                    new_result = process_participant_moral(participants_data[i], model, temperature)
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
    """Main execution function with parallel model execution"""
    # Configuration
    models_to_test = [
        'openai-gpt-3.5-turbo-0125',
        "gpt-4",
        "gpt-4o",
        "llama",
        "deepseek"
    ]

    temperature = 1
    output_dir = "study_4_moral_results"
    
    print("="*80)
    print("Study 4 Multi-Model Moral Scenario Simulation")
    print("="*80)

    # Load data
    data = load_york_data()
    participants_data = data.to_dict('records')
    
    print(f"Prepared {len(participants_data)} participants for moral simulation")
    print(f"Models: {models_to_test}")
    print(f"Temperature: {temperature}")
    print(f"Total models to test: {len(models_to_test)}")
    print("="*80)

    all_results = {}
    start_time = time.time()

    # Use ThreadPoolExecutor for parallel execution across models
    with ThreadPoolExecutor(max_workers=len(models_to_test)) as executor:
        # Submit all model simulations
        futures = [
            executor.submit(run_simulation_for_model, model, participants_data, temperature, output_dir)
            for model in models_to_test
        ]

        # Collect results as they complete
        completed_count = 0
        total_jobs = len(futures)

        for future in as_completed(futures):
            key, result = future.result()
            all_results[key] = result
            completed_count += 1

            # Progress update
            safe_print(f"Progress: {completed_count}/{total_jobs} simulations completed", "PROGRESS")

    total_duration = time.time() - start_time

    # Enhanced retry logic for failed participants
    print("\n" + "="*60)
    print("RETRY PHASE - Processing Failed Participants")
    print("="*60)

    retry_results = {}
    for model in models_to_test:
        simulation_id = f"{model}_temp{temperature}"
        if simulation_id in all_results and isinstance(all_results[simulation_id], list):
            results = all_results[simulation_id]
            failed_count = sum(1 for r in results if isinstance(r, dict) and 'error' in r)

            if failed_count > 0:
                safe_print(f"Retrying {failed_count} failed participants for {model}", "RETRY")
                updated_results = retry_failed_participants(results, participants_data, model, temperature)
                all_results[simulation_id] = updated_results

                # Save updated results
                output_path = Path(output_dir)
                filename = f"moral_{model}_temp{temperature}_retried.json"
                with open(output_path / filename, 'w') as f:
                    json.dump(updated_results, f, indent=2)

                retry_results[model] = updated_results
            else:
                safe_print(f"No failed participants to retry for {model}", "RETRY")

    # Final summary
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    print(f"Total execution time: {total_duration:.1f} seconds")
    print(f"Completed simulations: {len(all_results)}")

    # Categorize results
    successful_sims = []
    failed_sims = []

    for model in models_to_test:
        simulation_id = f"{model}_temp{temperature}"
        if simulation_id in all_results:
            result = all_results[simulation_id]
            if isinstance(result, dict) and 'error' in result:
                failed_sims.append(simulation_id)
                print(f"  {simulation_id}: FAILED - {result.get('error', 'Unknown error')}")
            else:
                # Check for partial failures
                if isinstance(result, list):
                    total = len(result)
                    successful = sum(1 for r in result if not (isinstance(r, dict) and 'error' in r))
                    failed = total - successful
                    success_rate = (successful / total) * 100

                    if failed > 0:
                        print(f"  {simulation_id}: SUCCESS with {failed} failed participants ({success_rate:.1f}% success)")
                    else:
                        print(f"  {simulation_id}: SUCCESS (100% success)")
                    successful_sims.append(simulation_id)
                else:
                    successful_sims.append(simulation_id)
        else:
            failed_sims.append(simulation_id)
            print(f"  {simulation_id}: MISSING")

    if failed_sims:
        print(f"\nFailed simulations ({len(failed_sims)}): {failed_sims}")

    # Save preprocessed data for reference
    output_path = Path(output_dir)
    data.to_csv(output_path / 'study4_moral_preprocessed_data.csv', index=False)
    print(f"\nPreprocessed data saved to {output_path / 'study4_moral_preprocessed_data.csv'}")

    print("\n" + "="*80)
    print("MORAL SIMULATION COMPLETE!")
    print("\nNext steps:")
    print("1. Run study_4_moral_behavioral_analysis.py for validation analysis")
    print(f"2. Results are saved in {output_dir}/ directory")
    print("3. Use unified_behavioral_analysis.py for cross-scenario comparison")
    print("="*80)

if __name__ == "__main__":
    main()