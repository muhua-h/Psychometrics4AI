"""
Binary Baseline Personality Prompt Generator

This module creates simplified binary personality descriptions based on BFI-2 domain scores.
If a domain score is above 50% (2.5 on 1-5 scale), the person is classified as "high" on that trait.
If a domain score is at or below 50%, the person is classified as "low" on that trait.

This serves as a baseline comparison to the more complex expanded and likert formats.
"""

import schema_tda

# # Baseline Simple: Domain descriptions for binary personality traits
# DOMAIN_DESCRIPTIONS = {
#     'bfi2_e': {
#         'high': "You are high in Extraversion.",
#         'low': "You are low in Extraversion."
#     },
#     'bfi2_a': {
#         'high': "You are high in Agreeableness.",
#         'low': "You are low in Agreeableness."
#     },
#     'bfi2_c': {
#         'high': "You are high in Conscientiousness.",
#         'low': "You are low in Conscientiousness."
#     },
#     'bfi2_n': {
#         'high': "You are high in Neuroticism.",
#         'low': "You are low in Neuroticism."
#     },
#     'bfi2_o': {
#         'high': "You are high in Openness.",
#         'low': "You are low in Openness."
#     }
# }
## Note: uncomment the following lines to use the elaborated descriptions instead of the simple ones
# Baseline Elaborated: Domain names and their corresponding binary descriptions
DOMAIN_DESCRIPTIONS = {
    'bfi2_e': {
        'high': "You are high in Extraversion. You are outgoing, sociable, assertive, and energetic.",
        'low': "You are low in Extraversion. You are reserved, quiet, and prefer smaller social settings."
    },
    'bfi2_a': {
        'high': "You are high in Agreeableness. You are compassionate, cooperative, trusting, and kind to others.",
        'low': "You are low in Agreeableness. You tend to be skeptical, competitive, and direct in your interactions."
    },
    'bfi2_c': {
        'high': "You are high in Conscientiousness. You are organized, responsible, hardworking, and reliable.",
        'low': "You are low in Conscientiousness. You tend to be spontaneous, flexible, and less focused on organization."
    },
    'bfi2_n': {
        'high': "You are high in Neuroticism. You tend to experience anxiety, worry, and emotional sensitivity.",
        'low': "You are low in Neuroticism. You are emotionally stable, calm, and resilient under stress."
    },
    'bfi2_o': {
        'high': "You are high in Openness. You are curious, creative, open to new experiences, and intellectually engaged.",
        'low': "You are low in Openness. You prefer familiar routines, practical approaches, and conventional ideas."
    }
}

def generate_binary_personality_description(participant_data, threshold=2.5):
    """
    Generate a binary personality description based on domain scores.
    
    Args:
        participant_data (dict): Dictionary containing BFI-2 domain scores
        threshold (float): Threshold for high/low classification (default: 2.5 for 50%)
        
    Returns:
        str: Binary personality description
    """
    domain_cols = ['bfi2_e', 'bfi2_a', 'bfi2_c', 'bfi2_n', 'bfi2_o']
    descriptions = []
    
    for domain in domain_cols:
        if domain in participant_data:
            score = participant_data[domain]
            if score > threshold:
                descriptions.append(DOMAIN_DESCRIPTIONS[domain]['high'])
            else:
                descriptions.append(DOMAIN_DESCRIPTIONS[domain]['low'])
        else:
            print(f"Warning: Domain {domain} not found in participant data")
    
    return " ".join(descriptions)

def get_binary_prompt(personality_description):
    """
    Generate a prompt for binary baseline personality assessment.
    
    Args:
        personality_description (str): Binary personality description
        
    Returns:
        str: Complete prompt for personality assessment
    """
    tda_list = list(schema_tda.tda_dict.values())
    
    def format_tda_list(tda_list):
        formatted_list = ""
        for i, trait in enumerate(tda_list, 1):
            formatted_list += f"{i}. {trait} _\n"
        return formatted_list
    
    instruction = (f"I will provide you a list of descriptive traits. For each "
                   f"trait, take a deep breath and think about what personality "
                   f"you are assigned with then, choose a number indicating how "
                   f"accurately that trait describes you. Using the following "
                   f"rating scale:\n"
                   f"1 - Extremely Inaccurate \n"
                   f"2 - Very Inaccurate\n"
                   f"3 - Moderately Inaccurate\n"
                   f"4 - Slightly Inaccurate\n"
                   f"5 - Neutral / Not Applicable\n"
                   f"6 - Slightly Accurate\n"
                   f"7 - Moderately Accurate\n"
                   f"8 - Very Accurate\n"
                   f"9 - Extremely Accurate\n")
    
    binary_context = ("Based on your personality profile below, please rate yourself "
                     "on the following traits.\n")
    
    # Create example JSON to show expected format
    example_json = '{\n    "Bashful": 7,\n    "Bold": 3,\n    "Careless": 2,\n    ...\n    "Withdrawn": 4\n}'
    
    prompt = (f"### Your Assigned Personality ### \n"
              f"{binary_context}{personality_description}\n\n"
              
              f"### Context and Objective ###\n"
              f"You are participating in a study to help us understand human personality.\n\n"
              "Your job is to fill out a personality questionnaire below. Your questionnaire answers "
              "should be reflective of your assigned personalities.\n\n"
              
              f"### Response Format ###\n"
              f"IMPORTANT: You must provide ratings for ALL 40 traits listed below.\n"
              f"Return ONLY a JSON object where:\n"
              f"- Keys are the exact trait names (e.g., \"Bashful\", \"Bold\", etc.)\n"
              f"- Values are numbers from 1-9 based on the rating scale\n"
              f"- Include ALL 40 traits - no more, no less\n"
              f"- Do NOT include personality domains like \"Extraversion\" or \"Agreeableness\"\n"
              f"- Do NOT add any text outside the JSON\n\n"
              f"Example format:\n{example_json}\n\n"
              
              f"### Questionnaire Instruction ###\n"
              f"{instruction}\n"
              
              f"### Questionnaire Item ###\n"
              f"{format_tda_list(tda_list)}")
    
    return prompt

def create_binary_participant_data(original_participants_data, threshold=2.5):
    """
    Create binary personality descriptions for a list of participants.
    
    Args:
        original_participants_data (list): List of participant dictionaries with BFI domain scores
        threshold (float): Threshold for high/low classification
        
    Returns:
        list: List of participant dictionaries with added binary personality descriptions
    """
    updated_participants = []
    
    for participant in original_participants_data:
        # Create a copy of the participant data
        updated_participant = participant.copy()
        
        # Generate binary personality description
        binary_description = generate_binary_personality_description(participant, threshold)
        
        # Add the binary description to the participant data
        updated_participant['binary_personality'] = binary_description
        
        updated_participants.append(updated_participant)
    
    return updated_participants

def validate_minimarker_response(response):
    """
    Validate that the response contains all 40 Mini-Marker traits.
    
    Args:
        response: The response to validate (should be a dict)
        
    Returns:
        tuple: (is_valid, error_message)
    """
    expected_traits = list(schema_tda.tda_dict.values())
    
    if not isinstance(response, dict):
        return False, "Response is not a dictionary"
    
    # Check for missing traits
    missing_traits = [trait for trait in expected_traits if trait not in response]
    if missing_traits:
        return False, f"Missing traits: {missing_traits}"
    
    # Check for extra traits (including Big Five domains)
    extra_traits = [key for key in response.keys() if key not in expected_traits]
    if extra_traits:
        return False, f"Unexpected traits found: {extra_traits}"
    
    # Validate that all values are integers between 1-9
    for trait, value in response.items():
        if not isinstance(value, (int, float)):
            return False, f"Invalid value type for {trait}: {type(value)}"
        if value < 1 or value > 9:
            return False, f"Invalid value for {trait}: {value} (must be 1-9)"
    
    return True, "Valid response"


# Example usage and testing
if __name__ == "__main__":
    # Test with sample participant data
    sample_participant = {
        'bfi2_e': 3.2,  # High extraversion
        'bfi2_a': 2.1,  # Low agreeableness  
        'bfi2_c': 4.0,  # High conscientiousness
        'bfi2_n': 1.8,  # Low neuroticism
        'bfi2_o': 3.8   # High openness
    }
    
    print("=== Binary Baseline Personality Description ===")
    binary_desc = generate_binary_personality_description(sample_participant)
    print(binary_desc)
    
    print("\n=== Binary Baseline Prompt ===")
    binary_prompt = get_binary_prompt(binary_desc)
    print(binary_prompt)