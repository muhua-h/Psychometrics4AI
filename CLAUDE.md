# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a psychometrics research project that designs LLM-Agents with Big Five personalities using psychometric tests. The codebase contains multiple studies that explore embedding-based analysis of personality tests, personality assignment to LLMs, and behavioral validation.

## Project Structure

- **study_1/**: Embedding analysis of personality scales
  - `scale_content.ipynb`: Transcription of psychometric tests into structured format
  - `scale_obtain_embedding.ipynb`: OpenAI API calls to obtain embeddings
  - `embedding_analysis.ipynb`: t-SNE and cosine similarity analysis
  
- **study_2/** & **study_3/**: Personality assignment validation
  - Both have `expanded_format/` and `likert_format/` subdirectories
  - `simulation_bfi2_miniMarker.ipynb`: LLM API calls for response generation
  - `process_json_bfi_miniMarker.ipynb`: JSON processing and data concatenation
  - `analysis.ipynb`: Statistical analysis of results
  - `factor_analysis.R`: Confirmatory factor analysis using R
  
- **study_4/**: Behavioral validation in decision-making tasks
  - `simulation/`: LLM response generation for moral and risk scenarios
  - `analysis/`: Statistical analysis and demographics
  - `data/`: Cleaned empirical datasets

- **raw_data/**: Original datasets (Soto BFI-2 data)

## Development Environment

### Python Dependencies
Install from `requirement.txt`:
- Python 3.10
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

## Key Code Patterns

### Schema Files
- `schema_bfi2.py` and `schema_tda.py`: Define expanded format personality scales
- `mini_marker_prompt.py`: Generate prompts for LLM personality assignment

### Data Processing
- Reverse coding is applied to specific personality items
- JSON responses from LLMs are processed and merged with empirical data
- Factor analysis validates personality structure in both formats

### LLM Integration
- OpenAI API calls for embedding generation and personality simulation
- Temperature parameter (temp0) used for consistent responses
- JSON format enforced for structured personality responses

## Common Workflows

### Running Jupyter Notebooks
```bash
jupyter notebook
```

### Running R Scripts
```bash
Rscript study_2/factor_analysis.R
```

### Installing Dependencies
```bash
pip install -r requirement.txt
```

## Data Flow

1. **Study 1**: Personality scales → Embeddings → Similarity analysis
2. **Study 2**: Empirical data → LLM simulation → Validation analysis
3. **Study 3**: Parameter extraction → Facet-level simulation → Validation
4. **Study 4**: Personality profiles → Decision scenarios → Behavioral analysis

## Important Notes

- File paths in R scripts may need adjustment based on local directory structure
- OpenAI API keys required for embedding and simulation tasks
- Factor analysis requires careful handling of reverse-coded items
- Expanded format shows superior performance for personality assignment