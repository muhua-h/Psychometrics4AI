# Designing LLM-Agents with Personalities: A Psychometric Approach

## Citation 
Huang, M., Zhang, X., Soto, C., & Evans, J. (2024). Designing LLM-agents with personalities: A psychometric approach. arXiv. [https://arxiv.org/abs/2410.19238](https://arxiv.org/abs/2410.19238)

## Objective 
This project aims to design LLM-Agents with Big Five personalities using psychometric tests. The study involves:
1. Understanding 
the psychometric tests using embeddings
2. Assigning personalities to LLM-Agents and validating the assignment
3. Evaluating the performance of LLM-Agents behavior using decision-making tasks. 

## Main Findings & Contribution
- Study 1 shows several different Big Five personality tests are measuring the same latent constructs in the embedding space. 
The embedding approach has two main contributions: 
  - It provides a new tool to understand similar psychometric tests in addition to traditional methods (e.g., 
qualitative analysis of different tests _or_ convergent correlations in participants' answers to different tests).
  - It lays the groundwork for understanding personality and psychometric tests in the embedding space, the space where 
LLM and AI models operate.
- Study 2 and Study 3 show that we can assign personalities to LLM-Agents using psychometric tests, while Study 2 starts 
with empirical data collected from human participants,
  and Study 3 starts with simulated data generated from sample statistics. 
  - Study 2 shows that LLM-Agents could absorb and manifest psychometrically validated personality traits, demenstrating 
their responses were in line with established personality data.
  - Study 3 introduces a parametric method to design and validate LLM-Agents with personalities, which enhances 
the process’s eﬀiciency and broad applicability
  - Additionally, the introduction of **Expanded Format** as a prompting strategies in these studies also marks a significant 
advancement.
    By validating that the Expanded format is superior for personality assignments in LLMs,
    our research suggests modifications to current psychometric tests
    that could lead to more nuanced and accurate personality replications. 
- Study 4 shows that LLM-Agents with diﬀerent personalities behave similarly to human participants with corresponding personality 
in decision-making tasks.
  -  This examination underpins the external validity, suggesting that LLMs equipped with accurately simulated personalities can effectively model human-like behavior across diverse scenarios. This bridges a crucial gap between theoretical personality constructs and their practical manifestations, promising to refine the predictive power and utility of personality models in digital and real-world applications.

## Folder & File Organization 
- `raw_data` folder contains the raw data used in Study 2, which is private data collected by authors of the paper cited below.
  - Soto CJ, John OP. The next Big Five Inventory (BFI-2): Developing and assessing a hierarchical model with 15 facets to enhance bandwidth, fidelity, and predictive power. J Pers Soc Psychol. 2017 Jul;113(1):117-143. doi: 10.1037/pspp0000096. Epub 2016 Apr 7. PMID: 27055049.
- `study_1` folder contains several files: 
  - In `scale_content.ipynb`, we transcribed and organize the content of the psychometric tests into a structured format, which leads 
to `all_scales.csv`, contains 5 Big Five personality tests, their items, domains and facets (when applicable).
  - In `scale_obtain_embedding.ipynb` we made API calls to OpenAI's embedding to obtain the embeddings of different psychometric tests by domain, which leads to `scales_embedding.csv`.
  - In `embedding_analysis.ipynb`, we performed dimension reduction using t-SNE projection and calculated the cosine 
similarity between the tests' embeddings.
    Two `png` files are generated to visualize the t-SNE projection and the cosine similarity matrix.
- `study_2` folder contains two subfolders: `expanded_format` and `likert_format`. Both subfolders are organized in the same way.
  - `simulation_bfi2_miniMarker.ipynb` processes the raw data, calls OpenAI's LLM API to generate responses and produces `bfi_to_mini_temp0.json`.
  - `process_json_bfi_miniMarker.ipynb` processes `bfi_to_mini_temp0.json` and concatenates the responses to the raw 
data `study1_data_no_simulation.csv`, which leads to `study1_data_with_simulation_result.csv`.
  - `analysis.ipynb` analyzes `study1_data_with_simulation_result.csv` and generates the statistics in Study 2.
  - `schema_bfi2.py`, 'schema_tda.py' and 'mini_marker_prompt.py' are used to generate the prompts for LLMs.
- `study_3` folder contains two subfolders: `expanded_format` and `likert_format`. Both subfolders are organized in the same way.
  - `data.csv` contains the raw data, the same as the one stored under `raw_data/Soto_data.xlsx`.
  - `bfi2-facet-level-parameter-extraction-and-simulation.ipynb` extracts the parameters from the raw data and simulates
    the responses using OpenAI's LLM API, which leads to `bfi_to_mini_temp0.json`.
  - The rest of process is identical to Study 2's. 
- `study_4` folder contains three subfolders: `simulation`, `data`, and `analysis`.
  - `data`,
    - `york_data_clean.csv` is the raw data used in Study 4 after de-identification.
  - `simulation`
    - It contains the simulation code to generate the responses from LLMs, clean LLM's returned json files, and generate the final dataset named `data_w_simulation.csv`.
## Required packages
- https://github.com/h-karyn/llm_psychometrics/blob/main/requirement.txt
