# Designing AI-Agents with Personalities: A Psychometric Approach

## Citation 
Huang, M., Zhang, X., Soto, C., & Evans, J. (2024). Designing LLM-agents with personalities: A psychometric approach. arXiv. [https://arxiv.org/abs/2410.19238](https://arxiv.org/abs/2410.19238)

## Objective 
This project aims to design AI-Agents with Big Five personalities using psychometric tests. The study involves:
1. Understanding the psychometric tests using embeddings
2. Assigning personalities to AI-Agents and validating the assignment
3. Evaluating the performance of AI-Agents behavior using decision-making tasks

## Main Findings & Contributions

### Study 1: Embedding Analysis of Personality Scales
- Shows that different Big Five personality tests measure the same latent constructs in the embedding space
- **Contributions**:
  - Provides a new tool to understand similar psychometric tests beyond traditional methods (qualitative analysis or convergent correlations)
  - Lays groundwork for understanding personality and psychometric tests in the embedding space where LLMs operate

### Study 2a: Personality Assignment Validation (Empirical Data)
- Demonstrates that AI-Agents can absorb and manifest psychometrically validated personality traits
- Shows responses align with established personality data from human participants
- **Key Innovation**: Introduction of **Expanded Format** as a superior prompting strategy for personality assignments in LLMs

### Study 2b: Facet-Level Parameter Extraction (Simulated Data)
- Introduces a parametric method to design and validate AI-Agents with personalities
- Enhances process efficiency and broad applicability through statistical simulation
- Validates personality structure at the facet level across multiple response formats

### Study 3: Behavioral Validation
- Shows that AI-Agents with different personalities behave similarly to human participants in decision-making tasks
- Examines external validity across moral reasoning and risk-taking scenarios
- Bridges the gap between theoretical personality constructs and practical manifestations

## Project Structure

### Core Studies
```
├── raw_data/                   # Original datasets
├── portal.py                   # Unified API interface
├── requirement.txt             # Python dependencies
└── multi_model_studies/        # Multi-model extension implementation
    ├── shared/                 # Common utilities and modules
    ├── study_1/                # Embedding analysis
    ├── study_2a/               # Personality assignment (empirical)
    ├── study_2b/               # Facet-level extraction (simulated)
    ├── study_3/                # Behavioral validation
    ├── binary_dichotomization_analysis/  # Binary format analysis
    ├── correlation_significance_test/    # Statistical testing
    └── factor_analysis/        # Psychometric validation
```

### Multi-Model Extension
The `multi_model_studies/` directory contains enhanced implementations that extend the research to multiple state-of-the-art LLM models:
- **GPT-4** (Azure OpenAI)
- **GPT-4o** (Azure OpenAI)  
- **Llama-3.3-70B-Instruct** (Azure AI)
- **DeepSeek-V3** (Azure AI)
- **GPT-3.5-Turbo** (OpenAI)

## Detailed File Organization

### Study 1: Embedding Analysis
- `scale_content.ipynb` - Transcription of psychometric tests into structured format
- `all_scales.csv` - 5 Big Five personality tests with items, domains and facets
- `scale_obtain_embedding.ipynb` - OpenAI API calls to obtain embeddings
- `scales_embedding.csv` - Embeddings of different psychometric tests by domain
- `embedding_analysis.ipynb` - t-SNE projection and cosine similarity analysis
- Visualization outputs: t-SNE projections and similarity matrices

### Study 2a: Empirical Validation
- Multiple format implementations: Binary (simple/elaborated), Expanded, Likert
- Simulation notebooks for each format with multi-model support
- Unified convergent validity analysis across formats and models
- Shared preprocessed data and results directories
- Recovery scripts for missing participants

### Study 2b: Simulated Parameter Extraction
- Facet-level BFI-2 data generation (200 simulated participants)
- Multi-format personality assignment validation
- Format-specific simulation notebooks
- Unified cross-format analysis framework
- Statistical parameter extraction and simulation

### Study 3: Behavioral Validation
- Moral reasoning and risk-taking scenario simulations
- Generalized framework for personality-behavior relationships
- Format-specific results for each model
- Comprehensive behavioral analysis scripts
- Human baseline comparison and validation

### Multi-Model Studies
Complete replication with multiple models:
- `shared/` - Common utilities and simulation framework
- `study_1/` - Multi-model embedding analysis
- `study_2a/` - Multi-model empirical validation
- `study_2b/` - Multi-model facet-level extraction
- `study_3/` - Multi-model behavioral validation
- See `multi_model_studies/README.md` for detailed documentation

## Raw Data
- `raw_data/Soto_data.xlsx` - BFI-2 data from Soto CJ, John OP (2017) study
- `raw_data/york_data_clean.csv` - Behavioral decision-making data

## Installation & Requirements

### Python Dependencies
```bash
pip install -r requirement.txt
```

Key packages:
- Python 3.10+
- openai 1.10.0
- pandas 2.1.4
- numpy 1.26.3
- matplotlib 3.8.2
- statsmodels 0.14.0

### R Dependencies
- psych (factor analysis)
- lavaan (confirmatory factor analysis)
- semTools (structural equation modeling)
- readr (CSV reading)

## API Configuration
Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_key_here
AZURE_OPENAI_KEY=your_key_here
AZURE_AI_INFERENCE_KEY=your_key_here
```

Configure endpoints in `portal.py` for multi-model studies.

## Key Innovations

### Expanded Format
A novel prompting strategy that provides detailed personality descriptions with full context and examples, showing superior performance for personality assignments in LLMs compared to traditional Likert scales.

### Multi-Format Support
- **Binary Format**: Yes/No personality descriptions (simple and elaborated variants)
- **Expanded Format**: Detailed personality descriptions with context
- **Likert Format**: Traditional rating scale format

### Comprehensive Analysis Framework
- Individual format analysis for each model
- Unified cross-format and cross-model comparisons
- Convergent validity testing
- Behavioral validation across decision-making scenarios

## Results Summary

### Convergent Validity (Studies 2a & 2b)
- **Expanded Format**: 0.87+ average BFI-Sim correlation (best performance)
- **Likert Format**: 0.85+ average correlation
- **Binary Format**: Lower correlations, as expected

### Behavioral Validation (Study 3)
- Successful replication of human personality-behavior relationships
- Consistent effects across moral reasoning and risk-taking scenarios
- Model performance varies by format and scenario type

## Citation & References

### Primary Citation
Huang, M., Zhang, X., Soto, C., & Evans, J. (2024). Designing LLM-agents with personalities: A psychometric approach. arXiv. https://arxiv.org/abs/2410.19238

### Data Source
Soto CJ, John OP. The next Big Five Inventory (BFI-2): Developing and assessing a hierarchical model with 15 facets to enhance bandwidth, fidelity, and predictive power. J Pers Soc Psychol. 2017;113(1):117-143.

## License & Contact
For questions about the implementation or to report issues, please open an issue on GitHub or contact the authors through the paper's corresponding author information.