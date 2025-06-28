"""
Simulation utilities for psychometrics AI studies.

This module provides unified functions for personality simulation across multiple LLM models,
replacing the scattered OpenAI API calls in Studies 2, 3, and 4 with a consistent interface
using the portal.py module.
"""

import json
import re
import ast
import time
import concurrent.futures
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from portal import get_model_response


class SimulationConfig:
    """Configuration class for simulation parameters."""
    
    def __init__(self, 
                 model: str = "gpt-4",
                 temperature: float = 0.0,
                 max_tokens: int = 2048,
                 batch_size: int = 35,
                 max_workers: int = 10):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.max_workers = max_workers


# def extract_json_from_response(text: str) -> List[Dict[str, Any]]:
#     """
#     Extract JSON objects from LLM response text.
#
#     Args:
#         text (str): The response text from the LLM
#
#     Returns:
#         List[Dict[str, Any]]: List of extracted JSON objects
#     """
#     json_pattern = r'(```json\s*)?\s*(\{[^}]+\})\s*(```)?'
#     matches = re.finditer(json_pattern, text, re.DOTALL)
#     extracted_jsons = []
#
#     for match in matches:
#         json_str = match.group(2)
#         try:
#             json_obj = json.loads(json_str)
#             extracted_jsons.append(json_obj)
#         except json.JSONDecodeError:
#             print(f"Warning: Could not parse JSON: {json_str}")
#
#     return extracted_jsons

def extract_json_from_response(text: str) -> dict:
    """
    Extract a JSON object from a messy LLM response.
    Handles:
      - Markdown fences (```json … ```)
      - Backticks around the JSON
      - Trailing commas before } or ]
      - Single quotes instead of double quotes
    Raises ValueError if no valid JSON can be parsed.
    """
    # 1) Strip markdown fences and backticks
    cleaned = text.strip()
    # remove ```json or ```js or plain ```
    cleaned = re.sub(r"```(?:json|js)?", "", cleaned,
                     flags=re.IGNORECASE).strip()
    # remove any leftover backticks
    cleaned = cleaned.strip("`").strip()

    # 2) Extract the first {...} or [...] block
    match = re.search(r"(\{.*\}|$begin:math:display$.*$end:math:display$)",
                      cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response.")
    json_str = match.group(1)

    # helper to remove trailing commas before } or ]
    def _strip_trailing_commas(s: str) -> str:
        # ,}  → }
        s = re.sub(r",\s*}", "}", s)
        # ,]  → ]
        s = re.sub(r",\s*\]", "]", s)
        return s

    # 3) First attempt: clean trailing commas, load as-is
    json_str = _strip_trailing_commas(json_str)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # 4) Second attempt: also normalize single → double quotes
    alt = json_str.replace("'", '"')
    alt = _strip_trailing_commas(alt)
    try:
        return json.loads(alt)
    except json.JSONDecodeError:
        pass

    # 5) Last resort: use Python literal eval (handles single quotes)
    try:
        return ast.literal_eval(json_str)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON:\n{text}\nError: {e}")

# @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
# def get_personality_response(prompt: str,
#                            personality_description: str,
#                            config: SimulationConfig) -> Dict[str, Any]:
#     """
#     Get a response from an LLM with personality traits using the unified portal interface.
#
#     Args:
#         prompt (str): The task/questionnaire prompt
#         personality_description (str): The personality description for the system prompt
#         config (SimulationConfig): Configuration for the simulation
#
#     Returns:
#         Dict[str, Any]: The parsed response from the LLM
#
#     Raises:
#         ValueError: If no valid JSON found in response
#         Exception: If API call fails after retries
#     """
#     system_prompt = f"You are a person with the given personality traits: {personality_description}"
#
#     try:
#         response_text = get_model_response(
#             model=config.model,
#             user_prompt=prompt,
#             system_prompt=system_prompt,
#             temperature=config.temperature,
#             max_tokens=config.max_tokens
#         )
#
#         # Try to extract JSON from response
#         extracted_jsons = extract_json_from_response(response_text)
#
#         if extracted_jsons:
#             return extracted_jsons[0]  # Return the first extracted JSON
#         else:
#             # If no JSON found, return the raw response for manual processing
#             return {"raw_response": response_text}
#
#     except Exception as e:
#         print(f"Error in get_personality_response: {str(e)}")
#         raise

# Global counter to track JSON parsing errors across all calls
_json_parsing_error_count = 0


def reset_json_parsing_error_count():
    """Reset the global JSON parsing error counter."""
    global _json_parsing_error_count
    _json_parsing_error_count = 0


def get_json_parsing_error_count():
    """Get the current count of JSON parsing errors."""
    return _json_parsing_error_count


@retry(stop=stop_after_attempt(10),
       wait=wait_exponential(multiplier=1, min=4, max=10))
def get_personality_response(prompt: str,
                             personality_description: str,
                             config: SimulationConfig) -> Dict[str, Any]:
    """
    Get a response from an LLM with personality traits using the unified portal interface.

    Args:
        prompt (str): The task/questionnaire prompt
        personality_description (str): The personality description for the system prompt
        config (SimulationConfig): Configuration for the simulation

    Returns:
        Dict[str, Any]: The parsed response from the LLM

    Raises:
        ValueError: If no valid JSON found in response after all retries
        Exception: If API call fails after retries
    """
    global _json_parsing_error_count
    # Use the exact system prompt from the original study
    system_prompt = "You are an agent participating in a research study. You will be given a personality profile."

    try:
        response_text = get_model_response(
            model=config.model,
            user_prompt=prompt,
            system_prompt=system_prompt,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

        # Try to extract JSON from response - this will raise ValueError if parsing fails
        extracted_json = extract_json_from_response(response_text)
        return extracted_json

    except ValueError as e:
        # JSON parsing failed - increment counter and let the retry decorator handle this
        _json_parsing_error_count += 1
        print(
            f"JSON parsing failed (attempt #{_json_parsing_error_count}): {str(e)}")
        raise
    except Exception as e:
        print(f"Error in get_personality_response: {str(e)}")
        raise

def process_single_participant(participant_data: Dict[str, Any],
                             prompt_generator: Callable,
                             config: SimulationConfig,
                             personality_key: str = 'combined_bfi2') -> Dict[str, Any]:
    """
    Process a single participant through the personality simulation.
    
    Args:
        participant_data (Dict[str, Any]): Data for one participant
        prompt_generator (Callable): Function to generate prompt from personality data
        config (SimulationConfig): Configuration for the simulation
        personality_key (str): Key in participant_data containing personality description
        
    Returns:
        Dict[str, Any]: The simulation result for this participant
    """
    try:
        personality_description = participant_data[personality_key]
        prompt = prompt_generator(personality_description)
        
        response = get_personality_response(prompt, personality_description, config)
        return response
        
    except Exception as e:
        print(f"Error processing participant: {str(e)}")
        return {"error": str(e)}


def run_batch_simulation(participants_data: List[Dict[str, Any]],
                        prompt_generator: Callable,
                        config: SimulationConfig,
                        personality_key: str = 'combined_bfi2',
                        output_dir: Optional[str] = None,
                        output_filename: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Run personality simulation for multiple participants in parallel batches.
    
    Args:
        participants_data (List[Dict[str, Any]]): List of participant data
        prompt_generator (Callable): Function to generate prompts
        config (SimulationConfig): Configuration for the simulation
        personality_key (str): Key in participant data containing personality description
        output_dir (Optional[str]): Directory to save results
        output_filename (Optional[str]): Filename for saving results
        
    Returns:
        List[Dict[str, Any]]: List of simulation results
    """
    num_participants = len(participants_data)
    results = [None] * num_participants
    
    print(f"Starting simulation for {num_participants} participants using {config.model}")
    print(f"Temperature: {config.temperature}, Batch size: {config.batch_size}")
    
    # Process participants in batches
    for batch_start in range(0, num_participants, config.batch_size):
        batch_end = min(batch_start + config.batch_size, num_participants)
        print(f"Processing participants {batch_start} to {batch_end - 1}")
        
        def process_participant_with_index(index):
            participant = participants_data[index]
            result = process_single_participant(participant, prompt_generator, config, personality_key)
            return index, result
        
        # Use ThreadPoolExecutor for concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(config.max_workers, config.batch_size)) as executor:
            future_to_index = {
                executor.submit(process_participant_with_index, i): i 
                for i in range(batch_start, batch_end)
            }
            
            for future in concurrent.futures.as_completed(future_to_index):
                index, result = future.result()
                results[index] = result
        
        print(f"Completed batch {batch_start} to {batch_end - 1}")
        
        # Small delay between batches to avoid overwhelming the API
        if batch_end < num_participants:
            time.sleep(2)
    
    # Save results if output parameters provided
    if output_dir and output_filename:
        save_simulation_results(results, output_dir, output_filename, config)
    
    return results


def retry_failed_participants(results: List[Dict[str, Any]],
                            participants_data: List[Dict[str, Any]],
                            prompt_generator: Callable,
                            config: SimulationConfig,
                            personality_key: str = 'combined_bfi2') -> List[Dict[str, Any]]:
    """
    Retry failed participants from a previous simulation run.
    
    Args:
        results (List[Dict[str, Any]]): Previous simulation results
        participants_data (List[Dict[str, Any]]): Original participant data
        prompt_generator (Callable): Function to generate prompts
        config (SimulationConfig): Configuration for the simulation
        personality_key (str): Key in participant data containing personality description
        
    Returns:
        List[Dict[str, Any]]: Updated results with retried participants
    """
    updated_results = results.copy()
    
    for index, result in enumerate(results):
        if isinstance(result, dict) and 'error' in result:
            print(f"Retrying participant {index}")
            try:
                participant = participants_data[index]
                new_response = process_single_participant(participant, prompt_generator, config, personality_key)
                updated_results[index] = new_response
                print(f"Successfully retried participant {index}")
            except Exception as e:
                print(f"Error retrying participant {index}: {str(e)}")
                updated_results[index] = {"error": str(e)}
            
            # Add delay between retries
            time.sleep(1)
    
    return updated_results


def save_simulation_results(results: List[Dict[str, Any]], 
                          output_dir: str, 
                          filename: str,
                          config: SimulationConfig) -> None:
    """
    Save simulation results to a JSON file.
    
    Args:
        results (List[Dict[str, Any]]): Simulation results
        output_dir (str): Directory to save the file
        filename (str): Base filename (will be modified with model and temperature info)
        config (SimulationConfig): Configuration used for simulation
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create filename with model and temperature info
    model_name = config.model.replace("-", "_")
    temp_str = str(config.temperature).replace(".", "_")
    full_filename = f"{filename}_{model_name}_temp{temp_str}.json"
    
    # Save results
    output_file = output_path / full_filename
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")


def load_simulation_results(filepath: str) -> List[Dict[str, Any]]:
    """
    Load simulation results from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file
        
    Returns:
        List[Dict[str, Any]]: Loaded simulation results
    """
    with open(filepath, "r") as f:
        return json.load(f)


def serialize_openai_completion(completion) -> Dict[str, Any]:
    """
    Serialize OpenAI ChatCompletion object to dictionary for backward compatibility.
    
    Args:
        completion: OpenAI ChatCompletion object
        
    Returns:
        Dict[str, Any]: Serialized completion data
    """
    def serialize_choice(choice):
        return {
            "finish_reason": choice.finish_reason,
            "index": choice.index,
            "logprobs": choice.logprobs,
            "message": {
                "content": choice.message.content,
                "role": choice.message.role,
                "function_call": choice.message.function_call,
                "tool_calls": choice.message.tool_calls
            }
        }
    
    return {
        "id": completion.id,
        "choices": [serialize_choice(choice) for choice in completion.choices],
        "created": completion.created,
        "model": completion.model,
        "object": completion.object,
        "usage": {
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens
        }
    }


# Convenience functions for each study type
def run_bfi_to_minimarker_simulation(participants_data: List[Dict[str, Any]],
                                   config: SimulationConfig,
                                   output_dir: str = "results") -> List[Dict[str, Any]]:
    """
    Run BFI-2 to Mini-Marker personality simulation (Study 2).
    
    Args:
        participants_data (List[Dict[str, Any]]): Participant data with 'combined_bfi2' key
        config (SimulationConfig): Simulation configuration
        output_dir (str): Directory to save results
        
    Returns:
        List[Dict[str, Any]]: Simulation results
    """
    from mini_marker_prompt import get_prompt
    
    return run_batch_simulation(
        participants_data=participants_data,
        prompt_generator=get_prompt,
        config=config,
        personality_key='combined_bfi2',
        output_dir=output_dir,
        output_filename="bfi_to_minimarker"
    )


def run_moral_simulation(participants_data: List[Dict[str, Any]],
                        config: SimulationConfig,
                        output_dir: str = "moral_results") -> List[Dict[str, Any]]:
    """
    Run moral decision-making simulation (Study 4).
    
    Args:
        participants_data (List[Dict[str, Any]]): Participant data with 'bfi_combined' key
        config (SimulationConfig): Simulation configuration  
        output_dir (str): Directory to save results
        
    Returns:
        List[Dict[str, Any]]: Simulation results
    """
    from moral_stories import get_prompt
    
    return run_batch_simulation(
        participants_data=participants_data,
        prompt_generator=get_prompt,
        config=config,
        personality_key='bfi_combined',
        output_dir=output_dir,
        output_filename="moral_simulation"
    )


def run_risk_simulation(participants_data: List[Dict[str, Any]],
                       config: SimulationConfig,
                       output_dir: str = "risk_results") -> List[Dict[str, Any]]:
    """
    Run risk-taking simulation (Study 4).
    
    Args:
        participants_data (List[Dict[str, Any]]): Participant data with 'bfi_combined' key
        config (SimulationConfig): Simulation configuration
        output_dir (str): Directory to save results
        
    Returns:
        List[Dict[str, Any]]: Simulation results
    """
    from risk_taking import get_prompt
    
    return run_batch_simulation(
        participants_data=participants_data,
        prompt_generator=get_prompt,
        config=config,
        personality_key='bfi_combined',
        output_dir=output_dir,
        output_filename="risk_simulation"
    )