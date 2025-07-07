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
from typing import List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
from pathlib import Path
# Add the project root to the path to import portal
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from portal import get_model_response


class SimulationConfig:
    """Configuration class for simulation parameters."""
    
    def __init__(self, 
                 model: str = "gpt-4",
                 temperature: float = 0.0,
                 max_tokens: int = 2048,
                 batch_size: int = 35,
                 max_workers: int = 10,
                 max_retries: int = 5,
                 base_wait_time: float = 2.0,
                 max_wait_time: float = 60.0):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.base_wait_time = base_wait_time
        self.max_wait_time = max_wait_time

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


class ResponseValidator:
    """Validates LLM responses for completeness and correctness."""
    
    # Expected Mini-Marker traits from the shared schema
    EXPECTED_TRAITS = {
        'Bashful', 'Bold', 'Careless', 'Cold', 'Complex', 'Cooperative', 
        'Creative', 'Deep', 'Disorganized', 'Efficient', 'Energetic', 
        'Envious', 'Extraverted', 'Fretful', 'Harsh', 'Imaginative', 
        'Inefficient', 'Intellectual', 'Jealous', 'Kind', 'Moody', 
        'Organized', 'Philosophical', 'Practical', 'Quiet', 'Relaxed', 
        'Rude', 'Shy', 'Sloppy', 'Sympathetic', 'Systematic', 'Talkative', 
        'Temperamental', 'Touchy', 'Uncreative', 'Unenvious', 
        'Unintellectual', 'Unsympathetic', 'Warm', 'Withdrawn'
    }
    
    VALID_VALUES = set(range(1, 10))  # 1-9 rating scale
    
    @classmethod
    def validate_response(cls, response: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a response for completeness and correctness.
        
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if response is a dict
        if not isinstance(response, dict):
            errors.append(f"Response is not a dictionary: {type(response)}")
            return False, errors
        
        # Check for expected traits
        found_traits = set()
        for key in response.keys():
            # Clean the key (remove numbers, whitespace, etc.)
            clean_key = cls._clean_trait_name(key)
            if clean_key in cls.EXPECTED_TRAITS:
                found_traits.add(clean_key)
        
        missing_traits = cls.EXPECTED_TRAITS - found_traits
        if missing_traits:
            errors.append(f"Missing traits: {sorted(list(missing_traits))}")
        
        # Check for extra/invalid traits  
        extra_traits = found_traits - cls.EXPECTED_TRAITS
        if extra_traits:
            errors.append(f"Extra/invalid traits: {sorted(list(extra_traits))}")
        
        # Check for unnamed/generic keys (like "Unnamed_1")
        unnamed_keys = [k for k in response.keys() if 'unnamed' in k.lower() or re.match(r'^(key|item|trait)_?\d+$', k.lower())]
        if unnamed_keys:
            errors.append(f"Generic/unnamed keys detected: {unnamed_keys}")
        
        # Check value validity
        invalid_values = []
        for key, value in response.items():
            try:
                int_value = int(value)
                if int_value not in cls.VALID_VALUES:
                    invalid_values.append(f"{key}={value}")
            except (ValueError, TypeError):
                invalid_values.append(f"{key}={value} (not convertible to int)")
        
        if invalid_values:
            errors.append(f"Invalid values (must be 1-9): {invalid_values}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @classmethod
    def _clean_trait_name(cls, key: str) -> str:
        """Clean trait name by removing numbers, whitespace, etc."""
        clean_key = key.strip()
        # Remove leading numbers and dots (e.g., "1. Bashful" -> "Bashful")
        clean_key = re.sub(r'^\d+\.\s*', '', clean_key)
        # Remove any trailing underscores or whitespace
        clean_key = clean_key.strip('_ ')
        return clean_key
    
    @classmethod
    def standardize_response(cls, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize a response by cleaning keys and ensuring proper format.
        
        Args:
            response: Raw response dictionary
            
        Returns:
            Standardized response dictionary
        """
        standardized = {}
        
        for key, value in response.items():
            clean_key = cls._clean_trait_name(key)
            
            # Only include expected traits
            if clean_key in cls.EXPECTED_TRAITS:
                try:
                    # Ensure value is an integer
                    int_value = int(value)
                    # Clamp to valid range
                    clamped_value = max(1, min(9, int_value))
                    standardized[clean_key] = clamped_value
                except (ValueError, TypeError):
                    # Default to neutral value for invalid responses
                    standardized[clean_key] = 5
        
        return standardized

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


def get_enhanced_personality_response(prompt: str,
                                    personality_description: str,
                                    config: SimulationConfig,
                                    participant_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get a validated response with automatic regeneration for failures.
    
    Args:
        prompt: The questionnaire prompt
        personality_description: Personality description for system prompt
        config: Configuration for the simulation
        participant_id: Optional participant ID for logging
        
    Returns:
        Valid response dictionary or error dict
    """
    validator = ResponseValidator()
    system_prompt = "You are an agent participating in a research study. You will be given a personality profile."
    
    for attempt in range(config.max_retries):
        try:
            # Get LLM response
            response_text = get_model_response(
                model=config.model,
                user_prompt=prompt,
                system_prompt=system_prompt,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
            
            # Extract JSON
            extracted_json = extract_json_from_response(response_text)
            
            # Validate response
            is_valid, errors = validator.validate_response(extracted_json)
            
            if is_valid:
                return validator.standardize_response(extracted_json)
            else:
                # Log validation errors
                participant_info = f"participant {participant_id}" if participant_id is not None else "participant"
                print(f"Validation failed for {participant_info} (attempt {attempt + 1}): {'; '.join(errors)}")
                
                # For partial failures (missing few traits), try to salvage on last attempt
                if attempt == config.max_retries - 1:
                    return _salvage_partial_response(extracted_json, errors, participant_id, validator)
                
                # Progressive prompt enhancement for retries
                if attempt > 0:
                    prompt = _enhance_prompt_for_retry(prompt, errors, attempt)
                
                # Add delay between retries
                wait_time = min(config.base_wait_time ** attempt, config.max_wait_time)
                print(f"  Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                
        except Exception as e:
            participant_info = f"participant {participant_id}" if participant_id is not None else "participant"
            print(f"Error for {participant_info} (attempt {attempt + 1}): {str(e)}")
            
            if attempt == config.max_retries - 1:
                # Last attempt failed - return error but don't lose the participant
                return {"error": str(e), "participant_id": participant_id, "recoverable": True}
            
            wait_time = min(config.base_wait_time ** attempt, config.max_wait_time)
            print(f"  Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)
    
    # If we get here, all retries failed
    return {"error": "All retry attempts failed", "participant_id": participant_id, "recoverable": True}


def _enhance_prompt_for_retry(original_prompt: str, errors: List[str], attempt: int) -> str:
    """Enhance the prompt based on the specific errors encountered."""
    
    enhancements = []
    
    # Check for missing traits error
    if any("Missing traits" in error for error in errors):
        enhancements.append(
            "IMPORTANT: Your response must include ALL 40 traits listed below. "
            "Do not skip any traits. Each trait must have a corresponding number from 1-9."
        )
    
    # Check for unnamed keys error
    if any("Generic/unnamed keys" in error for error in errors):
        enhancements.append(
            "CRITICAL: Use the EXACT trait names provided in the questionnaire. "
            "Do NOT use generic names like 'Unnamed_1' or 'Item_1'. "
            "Use the specific trait names: Bashful, Bold, Careless, etc."
        )
    
    # Check for invalid values
    if any("Invalid values" in error for error in errors):
        enhancements.append(
            "VALUES: Each trait must be rated with a number from 1 to 9 only. "
            "1=Extremely Inaccurate, 5=Neutral, 9=Extremely Accurate."
        )
    
    if enhancements:
        enhancement_text = "\n\n### RETRY INSTRUCTIONS ###\n" + "\n".join(enhancements) + "\n"
        # Insert enhancement before the questionnaire section
        enhanced_prompt = original_prompt.replace("### Questionnaire Item ###", 
                                                enhancement_text + "### Questionnaire Item ###")
        return enhanced_prompt
    
    return original_prompt


def _salvage_partial_response(response: Dict[str, Any], errors: List[str], participant_id: Optional[int], validator: ResponseValidator) -> Dict[str, Any]:
    """
    Attempt to salvage a partial response by filling in missing values.
    """
    participant_info = f"participant {participant_id}" if participant_id is not None else "participant"
    print(f"Attempting to salvage partial response for {participant_info}")
    
    standardized = validator.standardize_response(response)
    
    # Fill in missing traits with neutral values
    for trait in validator.EXPECTED_TRAITS:
        if trait not in standardized:
            standardized[trait] = 5  # Neutral value
    
    print(f"Salvaged response for {participant_info} - filled {len(validator.EXPECTED_TRAITS) - len(response)} missing traits")
    
    return standardized

def process_single_participant(participant_data: Dict[str, Any],
                             prompt_generator: Callable,
                             config: SimulationConfig,
                             personality_key: str = 'combined_bfi2',
                             use_enhanced: bool = False,
                             participant_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Process a single participant through the personality simulation.
    
    Args:
        participant_data (Dict[str, Any]): Data for one participant
        prompt_generator (Callable): Function to generate prompt from personality data
        config (SimulationConfig): Configuration for the simulation
        personality_key (str): Key in participant_data containing personality description
        use_enhanced (bool): Whether to use enhanced response validation
        participant_id (Optional[int]): Participant ID for logging
        
    Returns:
        Dict[str, Any]: The simulation result for this participant
    """
    try:
        personality_description = participant_data[personality_key]
        prompt = prompt_generator(personality_description)
        
        if use_enhanced:
            response = get_enhanced_personality_response(prompt, personality_description, config, participant_id)
        else:
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
                        output_filename: Optional[str] = None,
                        use_enhanced: bool = False) -> List[Dict[str, Any]]:
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
            result = process_single_participant(participant, prompt_generator, config, personality_key, use_enhanced, index)
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
                                   output_dir: str = "results",
                                   use_enhanced: bool = False,
                                   prompt_generator: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    Run BFI-2 to Mini-Marker personality simulation (Study 2).
    
    Args:
        participants_data (List[Dict[str, Any]]): Participant data with 'combined_bfi2' key
        config (SimulationConfig): Simulation configuration
        output_dir (str): Directory to save results
        use_enhanced (bool): Whether to use enhanced validation
        prompt_generator (Optional[Callable]): Prompt generator function (if None, uses default)
        
    Returns:
        List[Dict[str, Any]]: Simulation results
    """
    if prompt_generator is None:
        from mini_marker_prompt import get_prompt
        prompt_generator = get_prompt
    
    return run_batch_simulation(
        participants_data=participants_data,
        prompt_generator=prompt_generator,
        config=config,
        personality_key='combined_bfi2',
        output_dir=output_dir,
        output_filename="bfi_to_minimarker",
        use_enhanced=use_enhanced
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


# Convenience wrapper function for enhanced BFI simulation
def run_enhanced_bfi_to_minimarker_simulation(participants_data: List[Dict[str, Any]],
                                            config: SimulationConfig,
                                            output_dir: str = "results",
                                            use_enhanced: bool = True,
                                            prompt_generator: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """
    Run enhanced BFI-2 to Mini-Marker personality simulation with validation and auto-retry.
    
    This is a convenience wrapper around run_bfi_to_minimarker_simulation with enhanced=True.
    
    Args:
        participants_data (List[Dict[str, Any]]): Participant data with 'combined_bfi2' key
        config (SimulationConfig): Simulation configuration
        output_dir (str): Directory to save results
        use_enhanced (bool): Whether to use enhanced validation (default: True)
        prompt_generator (Optional[Callable]): Prompt generator function (if None, uses default)
        
    Returns:
        List[Dict[str, Any]]: Simulation results with enhanced validation
    """
    return run_bfi_to_minimarker_simulation(
        participants_data=participants_data,
        config=config,
        output_dir=output_dir,
        use_enhanced=use_enhanced,
        prompt_generator=prompt_generator
    )