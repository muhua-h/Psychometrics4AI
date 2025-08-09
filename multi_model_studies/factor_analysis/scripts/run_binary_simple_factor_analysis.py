#!/usr/bin/env python3
"""
Run Factor Analysis for Binary Simple Format Data

This script processes the missing binary simple format factor analyses for Studies 2 and 3.
The current "binary" format in factor_analysis_results is actually "binary elaborated".
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Dict, List, Any

# Add shared directory to path
sys.path.append(str(Path(__file__).parent / "shared"))

# Import the factor analysis framework
try:
    from factor_analysis_utils import FactorAnalysisFramework
except ImportError as e:
    print(f"Error importing factor analysis utilities: {e}")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_model_name(filename: str) -> str:
    """Extract standardized model name from filename"""
    filename_lower = filename.lower()
    
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

def convert_factor_results_to_csv(analysis_results: Dict[str, Any], study_name: str, format_name: str, output_dir: Path):
    """Convert factor analysis results to CSV format matching existing structure"""
    
    if not analysis_results:
        logger.warning(f"No results to convert for {study_name} {format_name}")
        return
    
    # Create output directory
    study_dir = output_dir / study_name.lower()
    format_dir = study_dir / f"{format_name}_format"
    format_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each model's results
    for model_key, model_results in analysis_results.items():
        if not isinstance(model_results, dict):
            continue
        
        model_name = extract_model_name(model_key)
        logger.info(f"  Converting results for {model_name}")
        
        loading_entries = []
        summary_entries = []
        
        # Get participant count from file_info
        file_info = model_results.get('file_info', {})
        n_participants = file_info.get('n_participants', 0)
        
        # Get domain results
        domain_results = model_results.get('domain_results', {})
        
        # Process each domain
        for domain_name, domain_data in domain_results.items():
            if not isinstance(domain_data, dict):
                continue
            
            # Get factor loadings (stored as DataFrame in factor_loadings)
            factor_loadings_df = domain_data.get('factor_loadings')
            if factor_loadings_df is None or factor_loadings_df.empty:
                continue
            
            # Convert DataFrame to dict for consistency
            loadings = dict(zip(factor_loadings_df['item'], factor_loadings_df['loading']))
            
            # Get reliability measures (stored directly as float)
            alpha = domain_data.get('reliability', 0)
            omega = 0  # Not available in current framework
            
            # Get fit indices (from fit_measures)
            fit_measures = domain_data.get('fit_measures', {})
            rmsea = 0  # Not available in sklearn fallback
            cfi = 0    # Not available in sklearn fallback
            tli = 0    # Not available in sklearn fallback
            srmr = 0   # Not available in sklearn fallback
            
            # Calculate eigenvalue and variance explained from loadings
            loading_values = list(loadings.values())
            eigenvalue = sum(l**2 for l in loading_values) if loading_values else 0
            variance_explained = eigenvalue / len(loading_values) if loading_values else 0
            
            # Create summary entry
            summary_entry = {
                'Study': study_name.upper(),
                'Format': format_name,
                'Model': model_name,
                'Structure_Type': 'Original',  # Only original structure analysis
                'Factor_Domain': domain_name,
                'N_Items': len(loadings),
                'N_Participants': n_participants,
                'Alpha': alpha,
                'Omega': omega,
                'Eigenvalue': eigenvalue,
                'Variance_Explained': variance_explained,
                'Mean_Loading_Abs': np.mean([abs(v) for v in loadings.values()]) if loadings else 0,
                'Max_Loading_Abs': max([abs(v) for v in loadings.values()]) if loadings else 0,
                'Min_Loading_Abs': min([abs(v) for v in loadings.values()]) if loadings else 0,
                'RMSEA': rmsea,
                'CFI': cfi,
                'TLI': tli,
                'SRMR': srmr,
                'Total_Variance_Explained': 0,  # Not available in single domain analysis
                'N_Factors_Total': 5,  # Big Five domains
                'File_Source': f"{study_name.lower()}_{format_name}_results_{model_key}"
            }
            summary_entries.append(summary_entry)
            
            # Create loading entries
            for item_name, loading_value in loadings.items():
                loading_entry = {
                    'Study': study_name.upper(),
                    'Format': format_name,
                    'Model': model_name,
                    'Structure_Type': 'Original',
                    'Factor_Domain': domain_name,
                    'Item': item_name,
                    'Loading': loading_value,
                    'Loading_Abs': abs(loading_value),
                    'Alpha': alpha,
                    'Omega': omega,
                    'Eigenvalue': eigenvalue,
                    'Variance_Explained': variance_explained,
                    'N_Items': len(loadings),
                    'N_Participants': n_participants,
                    'RMSEA': rmsea,
                    'CFI': cfi,
                    'TLI': tli,
                    'SRMR': srmr,
                    'File_Source': f"{study_name.lower()}_{format_name}_results_{model_key}"
                }
                loading_entries.append(loading_entry)
        
        # Save CSV files
        if loading_entries:
            loadings_df = pd.DataFrame(loading_entries)
            loadings_file = format_dir / f"{model_name}_factor_loadings.csv"
            loadings_df.to_csv(loadings_file, index=False)
            logger.info(f"    Saved: {loadings_file} ({len(loadings_df)} rows)")
        
        if summary_entries:
            summary_df = pd.DataFrame(summary_entries)
            summary_file = format_dir / f"{model_name}_factor_summary.csv"
            summary_df.to_csv(summary_file, index=False)
            logger.info(f"    Saved: {summary_file} ({len(summary_df)} rows)")
        
        if not loading_entries and not summary_entries:
            logger.warning(f"    No data to save for {model_name} - check domain structure")

def run_binary_simple_factor_analysis():
    """Run factor analysis for binary simple format data in Studies 2 and 3"""
    
    base_dir = Path(__file__).parent
    results_dir = base_dir / "factor_analysis_results"
    factor_framework = FactorAnalysisFramework()
    
    logger.info("=== Running Binary Simple Format Factor Analysis ===")
    
    # Study 2 Binary Simple format
    logger.info("\n--- Study 2 Binary Simple Format ---")
    study_2_binary_simple_dir = base_dir / "study_2" / "study_2_simple_binary_results"
    
    if study_2_binary_simple_dir.exists():
        logger.info(f"Processing directory: {study_2_binary_simple_dir}")
        
        # Use the framework's batch analysis method
        analysis_results = factor_framework.analyze_all_model_results(
            study_2_binary_simple_dir, 
            pattern="*.json"
        )
        
        if analysis_results:
            logger.info(f"✅ Completed factor analysis for {len(analysis_results)} models")
            convert_factor_results_to_csv(analysis_results, "Study_2", "binary_simple", results_dir)
        else:
            logger.warning("❌ No results returned from factor analysis")
    else:
        logger.warning("❌ Study 2 Binary Simple results directory not found")
    
    # Study 3 Binary Simple format  
    logger.info("\n--- Study 3 Binary Simple Format ---")
    study_3_binary_simple_dir = base_dir / "study_3" / "study_3_binary_simple_results"
    
    if study_3_binary_simple_dir.exists():
        logger.info(f"Processing directory: {study_3_binary_simple_dir}")
        
        # Use the framework's batch analysis method
        analysis_results = factor_framework.analyze_all_model_results(
            study_3_binary_simple_dir, 
            pattern="*.json"
        )
        
        if analysis_results:
            # Filter out metadata files
            filtered_results = {k: v for k, v in analysis_results.items() 
                              if 'metadata' not in k.lower()}
            
            logger.info(f"✅ Completed factor analysis for {len(filtered_results)} models")
            convert_factor_results_to_csv(filtered_results, "Study_3", "binary_simple", results_dir)
        else:
            logger.warning("❌ No results returned from factor analysis")
    else:
        logger.warning("❌ Study 3 Binary Simple results directory not found")
    
    logger.info("\n=== Binary Simple Factor Analysis Complete ===")
    logger.info("Now we need to rename the current 'binary' format to 'binary_elaborated'")

def main():
    """Main execution function"""
    run_binary_simple_factor_analysis()

if __name__ == "__main__":
    main()