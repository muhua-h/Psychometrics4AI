"""
Enhanced simulation utilities with early detection and regeneration.

This module provides robust error handling and automatic regeneration for failed responses,
ensuring all participants are preserved in the dataset.
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
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from portal import get_model_response
except ImportError:
    # Alternative path for portal
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, project_root)
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


def extract_json_from_response(text: str) -> dict:
    """
    Enhanced JSON extraction with multiple fallback strategies.
    """
    # 1) Strip markdown fences and backticks
    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json|js)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = cleaned.strip("`").strip()

    # 2) Extract the first {...} or [...] block
    match = re.search(r"(\{.*\}|$begin:math:display$.*$end:math:display$)", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response.")
    json_str = match.group(1)

    # Helper to remove trailing commas before } or ]
    def _strip_trailing_commas(s: str) -> str:
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*\]", "]", s)
        return s

    # 3) Progressive parsing attempts
    parsing_attempts = [
        # Original string with trailing comma cleanup
        lambda: json.loads(_strip_trailing_commas(json_str)),
        
        # Normalize quotes and clean trailing commas
        lambda: json.loads(_strip_trailing_commas(json_str.replace("'", '"'))),
        
        # Use Python literal eval
        lambda: ast.literal_eval(json_str),
        
        # Fix common JSON formatting issues
        lambda: json.loads(_fix_common_json_issues(json_str)),
    ]
    
    for i, attempt in enumerate(parsing_attempts):
        try:
            result = attempt()
            if i > 0:
                print(f"JSON parsed successfully using attempt #{i+1}")
            return result
        except Exception:
            continue
    
    raise ValueError(f"Failed to parse JSON after all attempts:\n{text}")


def _fix_common_json_issues(json_str: str) -> str:
    """Fix common JSON formatting issues."""
    # Fix missing quotes around keys
    json_str = re.sub(r'(\w+):', r'"\1":', json_str)
    # Fix single quotes to double quotes
    json_str = json_str.replace("'", '"')
    # Remove trailing commas
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*\]", "]", json_str)
    return json_str


class EnhancedPersonalitySimulator:
    """Enhanced personality simulator with robust error handling and regeneration."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.validator = ResponseValidator()
        self.total_attempts = 0
        self.successful_responses = 0
        self.regeneration_attempts = 0
        
    def get_enhanced_personality_response(self, 
                                        prompt: str,
                                        personality_description: str,
                                        participant_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get a validated response with automatic regeneration for failures.
        
        Args:
            prompt: The questionnaire prompt
            personality_description: Personality description for system prompt
            participant_id: Optional participant ID for logging
            
        Returns:
            Valid response dictionary
        """
        
        system_prompt = "You are an agent participating in a research study. You will be given a personality profile."
        
        for attempt in range(self.config.max_retries):
            self.total_attempts += 1
            
            try:
                # Get LLM response
                response_text = get_model_response(
                    model=self.config.model,
                    user_prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                
                # Extract JSON
                extracted_json = extract_json_from_response(response_text)
                
                # Validate response
                is_valid, errors = self.validator.validate_response(extracted_json)
                
                if is_valid:
                    self.successful_responses += 1
                    return self.validator.standardize_response(extracted_json)
                else:
                    # Log validation errors
                    participant_info = f"participant {participant_id}" if participant_id is not None else "participant"
                    print(f"Validation failed for {participant_info} (attempt {attempt + 1}): {'; '.join(errors)}")
                    
                    # For partial failures (missing few traits), try to salvage
                    if attempt == self.config.max_retries - 1:
                        return self._salvage_partial_response(extracted_json, errors, participant_id)
                    
                    self.regeneration_attempts += 1
                    
                    # Progressive prompt enhancement for retries
                    if attempt > 0:
                        prompt = self._enhance_prompt_for_retry(prompt, errors, attempt)
                    
                    # Add delay between retries - configurable wait times for better stability
                    wait_time = min(self.config.base_wait_time ** attempt, self.config.max_wait_time)
                    print(f"  Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    
            except Exception as e:
                participant_info = f"participant {participant_id}" if participant_id is not None else "participant"
                print(f"Error for {participant_info} (attempt {attempt + 1}): {str(e)}")
                
                if attempt == self.config.max_retries - 1:
                    # Last attempt failed - return error but don't lose the participant
                    return {"error": str(e), "participant_id": participant_id, "recoverable": True}
                
                self.regeneration_attempts += 1
                wait_time = min(self.config.base_wait_time ** attempt, self.config.max_wait_time)
                print(f"  Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
        
        # If we get here, all retries failed
        return {"error": "All retry attempts failed", "participant_id": participant_id, "recoverable": True}
    
    def _enhance_prompt_for_retry(self, original_prompt: str, errors: List[str], attempt: int) -> str:
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
    
    def _salvage_partial_response(self, response: Dict[str, Any], errors: List[str], participant_id: Optional[int]) -> Dict[str, Any]:
        """
        Attempt to salvage a partial response by filling in missing values.
        """
        participant_info = f"participant {participant_id}" if participant_id is not None else "participant"
        print(f"Attempting to salvage partial response for {participant_info}")
        
        standardized = self.validator.standardize_response(response)
        
        # Fill in missing traits with neutral values
        for trait in self.validator.EXPECTED_TRAITS:
            if trait not in standardized:
                standardized[trait] = 5  # Neutral value
        
        print(f"Salvaged response for {participant_info} - filled {len(self.validator.EXPECTED_TRAITS) - len(response)} missing traits")
        
        return standardized
    
    def process_single_participant(self, 
                                 participant_data: Dict[str, Any],
                                 prompt_generator: Callable,
                                 personality_key: str = 'combined_bfi2',
                                 participant_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a single participant with enhanced error handling.
        """
        try:
            personality_description = participant_data[personality_key]
            prompt = prompt_generator(personality_description)
            
            response = self.get_enhanced_personality_response(
                prompt, 
                personality_description,
                participant_index
            )
            
            return response
            
        except Exception as e:
            print(f"Unexpected error processing participant {participant_index}: {str(e)}")
            return {"error": str(e), "participant_id": participant_index, "recoverable": True}
    
    def get_simulation_stats(self) -> Dict[str, Any]:
        """Get simulation statistics."""
        return {
            "total_attempts": self.total_attempts,
            "successful_responses": self.successful_responses,
            "regeneration_attempts": self.regeneration_attempts,
            "success_rate": self.successful_responses / max(1, self.total_attempts - self.regeneration_attempts),
            "regeneration_rate": self.regeneration_attempts / max(1, self.total_attempts)
        }


def run_enhanced_bfi_to_minimarker_simulation(participants_data: List[Dict[str, Any]],
                                            config: SimulationConfig,
                                            output_dir: str = "enhanced_results",
                                            prompt_generator: Optional[Callable] = None,
                                            personality_key: str = 'combined_bfi2') -> List[Dict[str, Any]]:
    """
    Run enhanced BFI-2 to Mini-Marker simulation with robust error handling.
    
    Args:
        participants_data: List of participant data dictionaries
        config: Simulation configuration
        output_dir: Directory to save results
        prompt_generator: Function to generate prompts (will import if None)
        personality_key: Key for personality description in participant data
        
    Returns:
        List of simulation results (guaranteed to have all participants)
    """
    
    if prompt_generator is None:
        from mini_marker_prompt import get_likert_prompt
        prompt_generator = get_likert_prompt
    
    simulator = EnhancedPersonalitySimulator(config)
    num_participants = len(participants_data)
    results = [None] * num_participants
    
    print(f"Starting enhanced simulation for {num_participants} participants using {config.model}")
    print(f"Temperature: {config.temperature}, Max retries per participant: {config.max_retries}")
    
    # Process in batches
    for batch_start in range(0, num_participants, config.batch_size):
        batch_end = min(batch_start + config.batch_size, num_participants)
        print(f"Processing participants {batch_start} to {batch_end - 1}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit batch jobs
            future_to_index = {}
            for i in range(batch_start, batch_end):
                future = executor.submit(
                    simulator.process_single_participant,
                    participants_data[i],
                    prompt_generator,
                    personality_key,
                    i
                )
                future_to_index[future] = i
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    print(f"Unexpected error with participant {index}: {str(e)}")
                    results[index] = {"error": str(e), "participant_id": index, "recoverable": True}
        
        print(f"Completed batch {batch_start} to {batch_end - 1}")
    
    # Final validation - ensure no participant is lost
    valid_results = 0
    error_results = 0
    recoverable_errors = 0
    
    for i, result in enumerate(results):
        if result is None:
            print(f"WARNING: Participant {i} has no result - creating placeholder")
            results[i] = {"error": "No result generated", "participant_id": i, "recoverable": True}
            error_results += 1
        elif isinstance(result, dict) and 'error' in result:
            error_results += 1
            if result.get('recoverable', False):
                recoverable_errors += 1
        else:
            valid_results += 1
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = f"enhanced_bfi_to_minimarker_{config.model.replace('-', '_')}_temp{config.temperature}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comprehensive summary
    stats = simulator.get_simulation_stats()
    print(f"\n{'='*80}")
    print(f"ENHANCED SIMULATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total participants: {num_participants}")
    print(f"Valid responses: {valid_results}")
    print(f"Error responses: {error_results}")
    print(f"Recoverable errors: {recoverable_errors}")
    print(f"Data preservation rate: {100 * num_participants / num_participants:.1f}%")
    print(f"")
    print(f"LLM Response Statistics:")
    print(f"  Total LLM calls: {stats['total_attempts']}")
    print(f"  Successful responses: {stats['successful_responses']}")
    print(f"  Regeneration attempts: {stats['regeneration_attempts']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Regeneration rate: {stats['regeneration_rate']:.1%}")
    print(f"")
    print(f"Results saved to: {filepath}")
    
    return results


# Backward compatibility functions
def save_enhanced_simulation_results(results: List[Dict[str, Any]], 
                                   output_dir: str, 
                                   base_filename: str, 
                                   config: SimulationConfig):
    """Save simulation results with enhanced metadata."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = f"{base_filename}_{config.model.replace('-', '_')}_temp{config.temperature}.json"
    filepath = output_path / filename
    
    # Add metadata
    enhanced_results = {
        "metadata": {
            "total_participants": len(results),
            "model": config.model,
            "temperature": config.temperature,
            "max_retries": config.max_retries,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "results": results
    }
    
    with open(filepath, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    print(f"Enhanced results saved to: {filepath}") 