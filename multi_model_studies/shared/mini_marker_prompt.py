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
        format_type (str): Either "likert", "expanded", or "default"
    """
    
    if format_type == "likert":
        # Use original Likert format structure exactly
        prompt = (f"### Context ###\n"
                  f"You are participating in a personality psychology study. You "
                  f"have been assigned with personality traits.\n\n"

                  f"### Your Assigned Personality ### \n"
                  f"{likert_context}\n"
                  f"{personality}\n\n"

                  f"### Objective ###\n"
                  f"Fill out a personality questionnaire. Your "
                  f"questionnaire answers should be reflective of your "
                  f"assigned personalities.\n\n"

                  f"### Response Format ###\n"
                  f"ONLY return your response as a JSON file where the keys are the "
                  f"traits and the numbers indicate your endorsement to the "
                  f"statements.\n\n"

                  f"### Questionnaire Instruction ###\n"
                  f"{instruction}\n"

                  f"### Questionnaire Item ###\n"
                  f"{format_tda_list(tda_list)}"
                  )
    elif format_type == "expanded":
        # Keep existing expanded format structure
        prompt = (f"### Your Assigned Personality ### \n"
                  f"{expanded_context}"
                  f"{personality}\n\n"
           
                  f"### Context and Objective ###\n"
                  f"You are participating in a study to help us understand human personality.\n\n"
                  "Your job is to fill out a personality questionnaire below. Your questionnaire answers "
                  "should be reflective of your assigned personalities.\n\n"

                  f"### Response Format ###\n"
                  f"ONLY return your response as a JSON file where the keys are the "
                  f"traits and the number that best describes you. Do not say anything else.\n\n"
                 
                  f"### Questionnaire Instruction ###\n"
                  f"{instruction}\n"
                
                  f"### Questionnaire Item ###\n"
                  f"{format_tda_list(tda_list)}"
                  )
    else:
        # Default behavior (backward compatibility)
        prompt = (f"### Your Assigned Personality ### \n"
                  f"{personality}\n\n"
           
                  f"### Context and Objective ###\n"
                  f"You are participating in a study to help us understand human personality.\n\n"
                  "Your job is to fill out a personality questionnaire below. Your questionnaire answers "
                  "should be reflective of your assigned personalities.\n\n"

                  f"### Response Format ###\n"
                  f"ONLY return your response as a JSON file where the keys are the "
                  f"traits and the number that best describes you. Do not say anything else.\n\n"
                 
                  f"### Questionnaire Instruction ###\n"
                  f"{instruction}\n"
                
                  f"### Questionnaire Item ###\n"
                  f"{format_tda_list(tda_list)}"
                  )
    return prompt


# Convenience functions for specific formats
def get_likert_prompt(personality):
    """Generate a prompt specifically for likert-format personality descriptions."""
    return get_prompt(personality, format_type="likert")


def get_expanded_prompt(personality):
    """Generate a prompt specifically for expanded-format personality descriptions."""
    return get_prompt(personality, format_type="expanded")
