"""
Organize Factor Analysis Results into CSV Format
Includes ALL formats: binary, expanded, and Likert

This script reads the unified factor analysis JSON results and organizes them
into CSV files with one model per table, separated by format directories.
"""

import json
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_format_from_filename(filename):
    """Extract format type from filename"""
    filename_lower = filename.lower()
    if 'binary' in filename_lower:
        return 'binary_format'
    elif 'expanded' in filename_lower or 'i_am' in filename_lower:
        return 'expanded_format'  
    elif 'likert' in filename_lower:
        return 'likert_format'
    else:
        # Try to determine from file structure or default
        return 'expanded_format'  # Default assumption

def extract_model_name(filename):
    """Extract model name from filename"""
    filename_lower = filename.lower()
    
    # Map different model name variants
    if 'gpt_4o' in filename_lower or 'gpt-4o' in filename_lower:
        return 'gpt_4o'
    elif 'gpt_4' in filename_lower or 'gpt-4' in filename_lower:
        return 'gpt_4'
    elif 'gpt_3.5' in filename_lower or 'openai_gpt_3' in filename_lower:
        return 'gpt_3.5_turbo'
    elif 'llama' in filename_lower:
        return 'llama_3.3_70b'
    elif 'deepseek' in filename_lower:
        return 'deepseek_v3'
    else:
        return 'unknown_model'

def organize_factor_results():
    """Organize factor analysis results into CSV format by model and format"""
    
    # Load the unified factor analysis results
    base_dir = Path(__file__).parent
    results_file = base_dir / "unified_factor_analysis_results.json"
    
    if not results_file.exists():
        logger.error(f"Results file not found: {results_file}")
        return
    
    logger.info(f"Loading results from: {results_file}")
    with open(results_file, 'r') as f:
        all_results = json.load(f)
    
    # Create output directory structure
    output_dir = base_dir / "factor_analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    # Track all data for CSV creation
    all_loadings_data = []
    all_summary_data = []
    
    # Process each study
    for study_name, study_results in all_results.items():
        logger.info(f"Processing {study_name}")
        
        # Create study directory
        study_dir = output_dir / study_name.lower()
        study_dir.mkdir(exist_ok=True)
        
        # Process each file result
        for file_key, file_results in study_results.items():
            
            # Extract format and model from filename
            format_type = extract_format_from_filename(file_key)
            model_name = extract_model_name(file_key)
            
            logger.info(f"  Processing {file_key} -> Format: {format_type}, Model: {model_name}")
            
            # Create format directory
            format_dir = study_dir / format_type
            format_dir.mkdir(exist_ok=True)
            
            # Process both original and modified structures
            for structure_type, result_data in file_results.items():
                if not isinstance(result_data, dict):
                    continue
                
                domains = result_data.get('domains', {})
                
                # Process each domain/factor
                for domain_name, domain_data in domains.items():
                    if not isinstance(domain_data, dict):
                        continue
                    
                    loadings = domain_data.get('loadings', {}) or domain_data.get('factor_loadings', {})
                    reliability = domain_data.get('reliability', {})
                    # Handle direct reliability fields in domain_data
                    if not reliability:
                        reliability = {
                            'alpha': domain_data.get('alpha_reliability', 0),
                            'omega': domain_data.get('omega_reliability', 0)
                        }
                    fit_indices = domain_data.get('fit_indices', {})
                    
                    # Create summary entry
                    summary_entry = {
                        'Study': study_name,
                        'Format': format_type.replace('_format', ''),
                        'Model': model_name,
                        'Structure_Type': structure_type.title(),
                        'Factor_Domain': domain_name,
                        'N_Items': len(loadings) if loadings else 0,
                        'N_Participants': domain_data.get('n_participants', 0),
                        'Alpha': reliability.get('alpha', 0),
                        'Omega': reliability.get('omega', 0),
                        'Eigenvalue': domain_data.get('eigenvalue', 0),
                        'Variance_Explained': domain_data.get('variance_explained', 0),
                        'Mean_Loading_Abs': sum(abs(v) for v in loadings.values()) / len(loadings) if loadings else 0,
                        'Max_Loading_Abs': max(abs(v) for v in loadings.values()) if loadings else 0,
                        'Min_Loading_Abs': min(abs(v) for v in loadings.values()) if loadings else 0,
                        'RMSEA': fit_indices.get('RMSEA', 0),
                        'CFI': fit_indices.get('CFI', 0),
                        'TLI': fit_indices.get('TLI', 0),
                        'SRMR': fit_indices.get('SRMR', 0),
                        'Total_Variance_Explained': result_data.get('total_variance_explained', 0),
                        'N_Factors_Total': result_data.get('n_factors', 0),
                        'File_Source': file_key
                    }
                    all_summary_data.append(summary_entry)
                    
                    # Create loading entries
                    for item_name, loading_value in loadings.items():
                        loading_entry = {
                            'Study': study_name,
                            'Format': format_type.replace('_format', ''),
                            'Model': model_name,
                            'Structure_Type': structure_type.title(),
                            'Factor_Domain': domain_name,
                            'Item': item_name,
                            'Loading': loading_value,
                            'Loading_Abs': abs(loading_value),
                            'Alpha': reliability.get('alpha', 0),
                            'Omega': reliability.get('omega', 0),
                            'Eigenvalue': domain_data.get('eigenvalue', 0),
                            'Variance_Explained': domain_data.get('variance_explained', 0),
                            'N_Items': len(loadings),
                            'N_Participants': domain_data.get('n_participants', 0),
                            'RMSEA': fit_indices.get('RMSEA', 0),
                            'CFI': fit_indices.get('CFI', 0),
                            'TLI': fit_indices.get('TLI', 0),
                            'SRMR': fit_indices.get('SRMR', 0),
                            'File_Source': file_key
                        }
                        all_loadings_data.append(loading_entry)
    
    # Convert to DataFrames
    loadings_df = pd.DataFrame(all_loadings_data)
    summary_df = pd.DataFrame(all_summary_data)
    
    logger.info(f"Created DataFrames: {len(loadings_df)} loading entries, {len(summary_df)} summary entries")
    
    # Organize by model within each format
    if not loadings_df.empty:
        for study in loadings_df['Study'].unique():
            study_dir = output_dir / study.lower()
            
            for format_type in loadings_df[loadings_df['Study'] == study]['Format'].unique():
                format_dir = study_dir / f"{format_type}_format"
                format_dir.mkdir(exist_ok=True)
                
                # Filter data for this study and format
                study_format_loadings = loadings_df[
                    (loadings_df['Study'] == study) & 
                    (loadings_df['Format'] == format_type)
                ]
                study_format_summary = summary_df[
                    (summary_df['Study'] == study) & 
                    (summary_df['Format'] == format_type)
                ]
                
                # Create one file per model
                for model in study_format_loadings['Model'].unique():
                    model_loadings = study_format_loadings[study_format_loadings['Model'] == model]
                    model_summary = study_format_summary[study_format_summary['Model'] == model]
                    
                    # Save model files
                    loadings_file = format_dir / f"{model}_factor_loadings.csv"
                    summary_file = format_dir / f"{model}_factor_summary.csv"
                    
                    model_loadings.to_csv(loadings_file, index=False)
                    model_summary.to_csv(summary_file, index=False)
                    
                    logger.info(f"Created: {loadings_file} ({len(model_loadings)} rows)")
                    logger.info(f"Created: {summary_file} ({len(model_summary)} rows)")
    
    # Create cross-format comparison
    comparison_dir = output_dir / "cross_format_comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    if not loadings_df.empty:
        # Comprehensive comparison
        comprehensive_file = comparison_dir / "comprehensive_model_format_comparison.csv"
        loadings_df.to_csv(comprehensive_file, index=False)
        logger.info(f"Created comprehensive comparison: {comprehensive_file}")
        
        # Format model summary
        format_summary = summary_df.groupby(['Study', 'Format', 'Model', 'Structure_Type']).agg({
            'Alpha': 'mean',
            'Omega': 'mean',
            'Eigenvalue': 'mean',
            'Variance_Explained': 'mean',
            'RMSEA': 'mean',
            'CFI': 'mean',
            'N_Items': 'sum',
            'N_Factors_Total': 'first'
        }).reset_index()
        
        format_summary_file = comparison_dir / "format_model_summary.csv"
        format_summary.to_csv(format_summary_file, index=False)
        logger.info(f"Created format summary: {format_summary_file}")
    
    logger.info("Factor analysis results organization complete!")
    return output_dir

if __name__ == "__main__":
    organize_factor_results()