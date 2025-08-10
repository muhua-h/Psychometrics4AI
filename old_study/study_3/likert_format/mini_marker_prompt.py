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


def format_tda_list(tda_list):
    formatted_list = ""
    for i, trait in enumerate(tda_list, 1):
        formatted_list += f"{i}. {trait} _\n"
    return formatted_list


# create the prompt
def get_prompt(personality):
    prompt = (f"### Context ###\n"
              f"You are participating in a personality psychology study. You "
              f"have been assigned with personality traits.\n\n"

              f"### Your Assigned Personality ### \n"
              f"The number indicates the extent to which you agree or "
              f"disagree with that statement. 1 means 'Disagree Strongly', "
              f"3 means 'Neural', and 5 means 'Agree Strongly'.\n\n"
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
    return prompt
