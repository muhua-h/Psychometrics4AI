#!/usr/bin/env python3
"""
Study 4 Generalized Combined Multi-Model Simulation

This script runs both moral and risk-taking simulations for all 4 personality formats:
- expanded, likert, binary_elaborated, binary_simple

Models to Test:
- GPT-4
- GPT-4o  
- Llama-3.3-70B-Instruct
- DeepSeek-V3

Data Flow:
1. Load York human behavioral data
2. Extract personality descriptions for all formats
3. Generate moral and risk-taking scenario prompts
4. Run simulations across multiple models and formats
5. Save results in organized folder structure
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

from moral_stories import get_prompt as get_moral_prompt
from risk_taking import get_prompt as get_risk_prompt
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

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def process_participant_with_retry(participant_data, model, personality_format, scenario_type, temperature=0.0):
    """Process a single participant with enhanced retry logic"""
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
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        # Get response from model
        response = get_model_response(model, prompt, temperature=temperature)
        
        # Parse and validate JSON response
        if isinstance(response, str):
            extracted_jsons = extract_jsons(response)

            if extracted_jsons:
                # Use the first valid JSON found
                for json_obj in extracted_jsons:
                    is_valid, validation_msg = validate_func(json_obj)
                    if is_valid:
                        return json_obj

                # If no valid JSON, return the first one with error info
                _, validation_msg = validate_func(extracted_jsons[0])
                return {"error": f"Invalid response format: {validation_msg}", "raw_response": response}
            else:
                return {"error": f"No JSON found in response: {response}"}

        elif isinstance(response, dict):
            is_valid, validation_msg = validate_func(response)
            if is_valid:
                return response
            else:
                return {"error": f"Invalid response format: {validation_msg}", "raw_response": response}
        else:
            return {"error": f"Unexpected response type: {type(response)}"}
            
    except Exception as e:
        # This will trigger the retry mechanism
        raise Exception(f"Processing error: {str(e)}")

def process_participant(participant_data, model, personality_format, scenario_type, temperature=0.0):
    """Process a single participant for scenarios with fallback"""
    try:
        return process_participant_with_retry(participant_data, model, personality_format, scenario_type, temperature)
    except Exception as e:
        # Final fallback after all retries failed
        return {"error": f"All retries failed: {str(e)}"}

def load_york_data():
    """Load and preprocess York behavioral data"""
    data_path = Path('../../../raw_data/york_data_clean.csv')
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

def run_simulation_for_model_format_scenario(model, participants_data, personality_format, scenario_type, temperature, output_dir):
    """Run simulation for a single model, format, and scenario type"""
    simulation_id = f"{model}_{personality_format}_{scenario_type}_temp{temperature}"

    # Start message
    safe_print(f"Starting simulation: {model} with {personality_format} for {scenario_type} (temp={temperature})", "START")

    start_time = time.time()

    try:
        results = []
        batch_size = 25

        # Process participants in batches
        for i in range(0, len(participants_data), batch_size):
            batch = participants_data[i:i+batch_size]
            safe_print(f"[{model}_{personality_format}_{scenario_type}] Processing batch {i//batch_size + 1} (participants {i+1}-{min(i+batch_size, len(participants_data))})", "BATCH")

            batch_results = []
            for participant in batch:
                result = process_participant(participant, model, personality_format, scenario_type, temperature)
                batch_results.append(result)
                time.sleep(0.1)  # Small delay to avoid rate limits

            results.extend(batch_results)
            safe_print(f"[{model}_{personality_format}_{scenario_type}] Completed batch {i//batch_size + 1}", "BATCH")

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"{scenario_type}_{model}_temp{temperature}.json"
        with open(output_path / filename, 'w') as f:
            json.dump(results, f, indent=2)

        # Calculate success rate
        successful = sum(1 for r in results if not (isinstance(r, dict) and 'error' in r))
        failed_count = len(results) - successful
        success_rate = (successful / len(results)) * 100
        duration = time.time() - start_time

        safe_print(f"Completed {model} with {personality_format} for {scenario_type}: {successful}/{len(results)} successful ({success_rate:.1f}%) in {duration:.1f}s", "COMPLETE")

        return {
            'model': model,
            'format': personality_format,
            'scenario_type': scenario_type,
            'temperature': temperature,
            'total_participants': len(results),
            'successful': successful,
            'failed': failed_count,
            'success_rate': success_rate,
            'duration': duration,
            'output_file': str(output_path / filename)
        }

    except Exception as e:
        safe_print(f"Error in {model} with {personality_format} for {scenario_type}: {str(e)}", "ERROR")
        return {
            'model': model,
            'format': personality_format,
            'scenario_type': scenario_type,
            'temperature': temperature,
            'error': str(e)
        }

def main():
    """Main execution function with parallel processing"""
    print("=== Study 4 Generalized Combined Simulation (Parallelized) ===")
    
    # Configuration
    models = ['openai-gpt-3.5-turbo-0125', 'gpt-4', 'gpt-4o', 'llama', 'deepseek']
    personality_formats = ['bfi_expanded',  'bfi_binary_elaborated', 'bfi_binary_simple', 'bfi_likert']
    scenario_types = ['moral', 'risk']
    temperature = 1.0
    max_workers = 10  # Number of parallel workers (one per model)
    
    # Load data
    print("Loading York data...")
    data = load_york_data()
    
    # Convert to list of dictionaries for easier processing
    participants_data = data.to_dict('records')
    print(f"Processing {len(participants_data)} participants")
    
    # Create output directory structure
    base_output_dir = Path('../study_4_generalized_results')
    base_output_dir.mkdir(exist_ok=True)
    
    # Create format directories with scenario subdirectories
    format_dirs = {}
    for format_name in personality_formats:
        format_dir = base_output_dir / f"{format_name}_format"
        format_dir.mkdir(exist_ok=True)
        
        # Create scenario subdirectories
        for scenario_type in scenario_types:
            scenario_dir = format_dir / scenario_type
            scenario_dir.mkdir(exist_ok=True)
            
        format_dirs[format_name] = format_dir
    
    # Prepare all simulation tasks
    simulation_tasks = []
    for model in models:
        for personality_format in personality_formats:
            for scenario_type in scenario_types:
                output_dir = format_dirs[personality_format] / scenario_type
                simulation_tasks.append({
                    'model': model,
                    'participants_data': participants_data,
                    'personality_format': personality_format,
                    'scenario_type': scenario_type,
                    'temperature': temperature,
                    'output_dir': output_dir
                })
    
    print(f"Total simulation tasks: {len(simulation_tasks)}")
    print(f"Running with {max_workers} parallel workers")
    
    # Run simulations in parallel
    all_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                run_simulation_for_model_format_scenario,
                task['model'],
                task['participants_data'],
                task['personality_format'],
                task['scenario_type'],
                task['temperature'],
                task['output_dir']
            ): task for task in simulation_tasks
        }
        
        # Collect results as they complete
        completed_count = 0
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed_count += 1
            
            try:
                result = future.result()
                all_results.append(result)
                
                # Print progress
                if 'error' in result:
                    safe_print(f"❌ Completed {completed_count}/{len(simulation_tasks)}: {result['model']} ({result['format']}, {result['scenario_type']}): {result['error']}", "PROGRESS")
                else:
                    safe_print(f"✅ Completed {completed_count}/{len(simulation_tasks)}: {result['model']} ({result['format']}, {result['scenario_type']}): {result['successful']}/{result['total_participants']} ({result['success_rate']:.1f}%)", "PROGRESS")
                    
            except Exception as e:
                error_result = {
                    'model': task['model'],
                    'format': task['personality_format'],
                    'scenario_type': task['scenario_type'],
                    'temperature': task['temperature'],
                    'error': f"Execution failed: {str(e)}"
                }
                all_results.append(error_result)
                safe_print(f"❌ Failed {completed_count}/{len(simulation_tasks)}: {task['model']} ({task['personality_format']}, {task['scenario_type']}): {str(e)}", "ERROR")
    
    # Save simulation metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'models': models,
        'personality_formats': personality_formats,
        'scenario_types': scenario_types,
        'temperature': temperature,
        'max_workers': max_workers,
        'total_participants': len(participants_data),
        'results': all_results
    }
    
    metadata_path = base_output_dir / 'simulation_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print final summary
    print("\n=== Final Simulation Summary ===")
    successful_count = 0
    failed_count = 0
    
    for result in all_results:
        if 'error' in result:
            print(f"❌ {result['model']} ({result['format']}, {result['scenario_type']}): {result['error']}")
            failed_count += 1
        else:
            print(f"✅ {result['model']} ({result['format']}, {result['scenario_type']}): {result['successful']}/{result['total_participants']} ({result['success_rate']:.1f}%)")
            successful_count += 1
    
    print(f"\nOverall: {successful_count} successful, {failed_count} failed")
    print(f"Results saved to: {base_output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    
    # Print directory structure
    print(f"\n=== Directory Structure ===")
    for format_name in personality_formats:
        format_dir = format_dirs[format_name]
        print(f"{format_dir}/")
        for scenario_type in scenario_types:
            scenario_dir = format_dir / scenario_type
            print(f"  {scenario_type}/")
            print(f"    - {scenario_type}_[model]_temp{temperature}.json")

if __name__ == "__main__":
    main() 