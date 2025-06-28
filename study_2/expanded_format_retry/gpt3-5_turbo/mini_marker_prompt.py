import schema_tda

tda_list = list(schema_tda.tda_dict.values())

instruction = (f"Please use this list of common traits to describe yourself"
               f" as accurately as possible. Describe yourself as you see "
               f"yourself at the present time, not as you wish to be in the fut"
               f"ure. Describe yourself as you are generally or typically, as co"
               f"mpared with your peers. For each trait, please write a number indicating"
               f" how accurately that trait describes you, using the following "
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
    prompt = (f"### Your Personality ### \n"
              f"{personality}\n\n"
       
              f"### Background ###\n"
              f"You are participating in a study to help us understand huamn personality. \n\n"

              f"### Response Format ###\n"
              f"ONLY return your response as a JSON file where the keys are the "
              f"traits and the number that best describes you. Do not say anything else.\n\n"
             
              f"### Questionnaire Instruction ###\n"
              f"{instruction}\n"
            
              f"### Questionnaire Item ###\n"
              f"{format_tda_list(tda_list)}"
              )
    return prompt
