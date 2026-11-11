"""
This module contains a list of moral stories that are used in the moral
"""

investment = ("Scenario: Investment\n"
              "You have recently come into a modest sum of money and are "
              "contemplating how to invest it. You're torn between a "
              "high-risk, high-reward stock that has the potential to either "
              "significantly increase your wealth or decrease in value, "
              "and a low-risk government bond that offers guaranteed but "
              "modest returns. On a scale of 1 to 10, where 1 means fully "
              "embracing the potential high rewards of the risky stock, "
              "and 10 means preferring the safety and guaranteed return of "
              "the government bond, how would act in this situation?")

extreme_sports = ("Scenario: Extreme_Sports \n"
                  "You've always been fascinated by extreme sports and are "
                  "considering taking up base jumping, drawn by the thrill "
                  "and the challenge it presents. However, you're fully aware "
                  "of the significant physical risks involved, including the "
                  "potential for severe injury. On a scale of 1 to 10, "
                  "where 1 means diving into the sport for the ultimate "
                  "thrill despite the risks, and 10 means prioritizing your "
                  "physical safety and choosing not to participate, how would "
                  "act in this situation?")

entrepreneurial_venture = ("Scenario: Entrepreneurial_Venture \n"
                           "You've developed an innovative business idea that "
                           "you believe has the potential to be highly "
                           "successful. Pursuing this would require you to "
                           "quit your stable job and invest a substantial "
                           "portion of your savings into the venture. While "
                           "the idea could revolutionize the industry, "
                           "there's also a significant risk of failure, "
                           "which could leave you without a job and deplete "
                           "your savings. On a scale of 1 to 10, where 1 "
                           "means pursuing the entrepreneurial venture with "
                           "all your resources, embracing the risk for the "
                           "chance of substantial reward, and 10 means "
                           "conserving your current job and financial "
                           "security, how would act in this situation?")

confessing_feeling = ("Scenario: Confessing_Feelings \n"
                      "You find yourself developing romantic feelings for "
                      "your best friend. These feelings have grown stronger "
                      "over time, and you're contemplating whether to "
                      "confess. Doing so could potentially lead to a deeper, "
                      "romantic relationship, enriching your connection. "
                      "However, there's also the risk that your confession "
                      "could make things awkward and possibly harm the "
                      "friendship you deeply value. On a scale of 1 to 10, "
                      "where 1 means opening up about your feelings, risking "
                      "the friendship for the possibility of something more, "
                      "and 10 means preserving the current friendship without "
                      "risking discomfort or loss, how would act in this "
                      "situation?")

study_overseas = ("Scenario: Study_Overseas\n"
                  "You have been offered admission to a prestigious "
                  "university in a foreign country, an opportunity that "
                  "promises to significantly advance your career and personal "
                  "development. This educational pursuit, however, comes with "
                  "a substantial sacrifice: you would need to leave behind "
                  "your family, friends, and the comfort of your familiar "
                  "environment. The move entails not only physical relocation "
                  "but also adapting to a new culture, potentially facing "
                  "language barriers, and starting anew without your "
                  "established support network. On a scale of 1 to 10, "
                  "where 1 means embracing the opportunity to study abroad, "
                  "fully committing to the personal and professional growth "
                  "it offers despite the sacrifices, and 10 means choosing to "
                  "stay in your home country to maintain your current "
                  "relationships and stability, how would act in this "
                  "situation?")

risk_story = [investment, extreme_sports, entrepreneurial_venture,
              confessing_feeling, study_overseas]


# design a function to return a randomized list of moral stories
def randomize_stories():
    import random
    duplicate = risk_story.copy()
    random.shuffle(duplicate)
    return duplicate


# design a function to add a scenario number to each moral story
def get_scenarios():
    scenario = []
    for index, value in enumerate(risk_story):
        scenario.append(f"##{value}\n")
    return scenario


# create the prompt
def get_prompt(personality):
    scenario = get_scenarios()

    prompt = (f"### Personality### \n"
              f"{personality}\n\n"
              f"### Background ###\n"
              f"Your decisions and actions are impacted by your personality.\n\n"
              f"### Objective ###\n"
              f"I will ask you a sequence of 5 scenario-based questions. "
              f"Please choose a value between 1 and 10 to indicate your "
              f"likelihood of an action.\n\n"
              f"### Response Format ###\n"
              f"Respond with a JSON file in the format of "
              f"{{Scenario Name: likelihood_value}}. \n\n"
              f"### Scenarios ###\n"
              f"{scenario[0]}\n"
              f"{scenario[1]}\n"
              f"{scenario[2]}\n"
              f"{scenario[3]}\n"
              f"{scenario[4]}\n\n")

    return prompt