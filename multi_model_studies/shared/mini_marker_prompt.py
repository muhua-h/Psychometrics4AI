import schema_tda

tda_list = list(schema_tda.tda_dict.values())

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

# Context explanations for different personality description formats
likert_context = ("The number indicates the extent to which you agree or "
                 "disagree with that statement. 1 means 'Disagree Strongly', "
                 "3 means 'Neutral', and 5 means 'Agree Strongly'.\n")

expanded_context = ("Based on your detailed personality description below, please rate yourself "
                   "on the following traits. Consider how each trait applies to you based on "
                   "the personality profile provided.\n")

binary_context = ("Based on your personality profile below, please rate yourself "
                 "on the following traits.\n")

# Domain names and their corresponding binary descriptions for binary format
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


def format_tda_list(tda_list):
    formatted_list = ""
    for i, trait in enumerate(tda_list, 1):
        formatted_list += f"{i}. {trait} _\n"
    return formatted_list


# create the prompt with format-specific context
def get_prompt(personality, format_type="default"):
    """
    Generate a prompt for personality assessment.

    Args:
        personality (str): The personality description
        format_type (str): Either "likert", "expanded", "binary", or "default"
    """

    # Format-specific personality section
    if format_type == "likert":
        personality_section = f"{likert_context}\n{personality}"
    elif format_type == "expanded":
        personality_section = f"{expanded_context}{personality}"
    elif format_type == "binary":
        personality_section = f"{binary_context}{personality}"
    else:  # default
        personality_section = personality

    # Use enhanced response format for all formats (originally from binary)
    example_json = '{\n    "Bashful": 7,\n    "Bold": 3,\n    "Careless": 2,\n    ...\n    "Withdrawn": 4\n}'
    
    response_format = (f"### Response Format ###\n"
                      f"IMPORTANT: You must provide ratings for ALL 40 traits listed below.\n"
                      f"Return ONLY a JSON object where:\n"
                      f"- Keys are the exact trait names (e.g., \"Bashful\", \"Bold\", etc.)\n"
                      f"- Values are numbers from 1-9 based on the rating scale\n"
                      f"- Include ALL 40 traits - no more, no less\n"
                      f"- Do NOT include personality domains like \"Extraversion\" or \"Agreeableness\"\n"
                      f"- Do NOT add any text outside the JSON\n\n"
                      f"Example format:\n{example_json}\n\n")

    # Build the common prompt structure
    prompt = (f"### Your Assigned Personality ### \n"
              f"{personality_section}\n\n"

              f"### Context and Objective ###\n"
              f"You are participating in a study to help us understand human personality.\n\n"
              "Your job is to fill out a personality questionnaire below. Your questionnaire answers "
              "should be reflective of your assigned personalities.\n\n"

              f"{response_format}"

              f"### Questionnaire Instruction ###\n"
              f"{instruction}\n"

              f"### Questionnaire Item ###\n"
              f"{format_tda_list(tda_list)}")

    return prompt


# Convenience functions for specific formats
def get_likert_prompt(personality):
    """Generate a prompt specifically for likert-format personality descriptions."""
    return get_prompt(personality, format_type="likert")


def get_expanded_prompt(personality):
    """Generate a prompt specifically for expanded-format personality descriptions."""
    return get_prompt(personality, format_type="expanded")


def get_binary_prompt(personality):
    """Generate a prompt specifically for binary-format personality descriptions."""
    return get_prompt(personality, format_type="binary")


# Binary personality description generation functions
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
