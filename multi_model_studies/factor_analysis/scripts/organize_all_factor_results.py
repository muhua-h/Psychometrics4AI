"""
Organize Factor Analysis Results into CSV Format
Includes ALL formats: binary, expanded, and Likert

This script manages factor analysis results by:
1. Checking existing organized CSV results
2. Identifying missing formats/models
3. Optionally running factor analysis for missing data
4. Creating comprehensive cross-format comparisons
"""

import json
import pandas as pd
from pathlib import Path
import logging
import sys
from typing import Dict, List, Set, Tuple

# Add shared directory to path for imports
sys.path.append(str(Path(__file__).parent / "shared"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_existing_results():
    """Check what factor analysis results already exist"""
    base_dir = Path(__file__).parent
    results_dir = base_dir / "factor_analysis_results"
    
    existing_results = {}
    
    if not results_dir.exists():
        logger.info("No existing factor analysis results directory found")
        return existing_results
    
    for study_dir in results_dir.iterdir():
        if study_dir.is_dir() and study_dir.name.startswith('study_'):
            study_name = study_dir.name
            existing_results[study_name] = {}
            
            for format_dir in study_dir.iterdir():
                if format_dir.is_dir() and format_dir.name.endswith('_format'):
                    format_name = format_dir.name.replace('_format', '')
                    existing_results[study_name][format_name] = []
                    
                    # Find all model files
                    for csv_file in format_dir.glob('*_factor_summary.csv'):
                        model_name = csv_file.name.replace('_factor_summary.csv', '')
                        existing_results[study_name][format_name].append(model_name)
    
    return existing_results

def identify_missing_analyses():
    """Identify what factor analyses are missing based on simulation data"""
    base_dir = Path(__file__).parent
    
    # Expected models and formats based on simulation data
    expected_models = ['gpt_4', 'gpt_4o', 'gpt_3.5_turbo', 'llama_3.3_70b', 'deepseek_v3']
    expected_formats = ['binary', 'expanded', 'likert']
    expected_studies = ['study_2', 'study_3']
    
    # Check what exists
    existing = check_existing_results()
    
    missing = {}
    for study in expected_studies:
        missing[study] = {}
        study_data = existing.get(study, {})
        
        for format_name in expected_formats:
            existing_models = set(study_data.get(format_name, []))
            expected_models_set = set(expected_models)
            missing_models = expected_models_set - existing_models
            
            if missing_models or format_name not in study_data:
                missing[study][format_name] = list(missing_models) if missing_models else expected_models
    
    return missing, existing

def get_simulation_data_files():
    """Get available simulation data files for factor analysis"""
    base_dir = Path(__file__).parent
    simulation_files = {}
    
    # Study 2 simulation files
    study_2_dir = base_dir / "study_2"
    if study_2_dir.exists():
        simulation_files['study_2'] = {
            'binary': list(study_2_dir.glob("study_2_*binary_results*/*.json")) + 
                     list(study_2_dir.glob("study_2_simple_binary_results*/*.json")),
            'expanded': list(study_2_dir.glob("study_2_expanded_results*/*.json")),
            'likert': list(study_2_dir.glob("study_2_likert_results*/*.json"))
        }
    
    # Study 3 simulation files  
    study_3_dir = base_dir / "study_3"
    if study_3_dir.exists():
        simulation_files['study_3'] = {
            'binary': list(study_3_dir.glob("study_3_binary_*results*/*.json")),
            'expanded': list(study_3_dir.glob("study_3_expanded_results*/*.json")),
            'likert': list(study_3_dir.glob("study_3_likert_results*/*.json"))
        }
    
    return simulation_files

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

def organize_existing_results():
    """Organize existing factor analysis results and create comprehensive comparisons"""
    base_dir = Path(__file__).parent
    results_dir = base_dir / "factor_analysis_results"
    
    if not results_dir.exists():
        logger.error("No factor analysis results directory found")
        return None
    
    # Collect all existing CSV data
    all_loadings_data = []
    all_summary_data = []
    
    # Process existing CSV files
    for study_dir in results_dir.iterdir():
        if not study_dir.is_dir() or not study_dir.name.startswith('study_'):
            continue
            
        study_name = study_dir.name.upper()
        logger.info(f"Processing existing results for {study_name}")
        
        for format_dir in study_dir.iterdir():
            if not format_dir.is_dir() or not format_dir.name.endswith('_format'):
                continue
                
            format_name = format_dir.name.replace('_format', '')
            
            # Read all CSV files in this format directory
            for loadings_file in format_dir.glob('*_factor_loadings.csv'):
                try:
                    df_loadings = pd.read_csv(loadings_file)
                    if not df_loadings.empty:
                        all_loadings_data.append(df_loadings)
                        logger.info(f"  Loaded {len(df_loadings)} loading entries from {loadings_file.name}")
                except Exception as e:
                    logger.error(f"  Error reading {loadings_file}: {e}")
            
            for summary_file in format_dir.glob('*_factor_summary.csv'):
                try:
                    df_summary = pd.read_csv(summary_file)
                    if not df_summary.empty:
                        all_summary_data.append(df_summary)
                        logger.info(f"  Loaded {len(df_summary)} summary entries from {summary_file.name}")
                except Exception as e:
                    logger.error(f"  Error reading {summary_file}: {e}")
    # Combine all DataFrames
    if not all_loadings_data:
        logger.warning("No loading data found in existing results")
        return None
        
    loadings_df = pd.concat(all_loadings_data, ignore_index=True)
    
    if not all_summary_data:
        logger.warning("No summary data found in existing results")
        return None
        
    summary_df = pd.concat(all_summary_data, ignore_index=True)
    
    logger.info(f"Combined DataFrames: {len(loadings_df)} loading entries, {len(summary_df)} summary entries")
    
    # Update cross-format comparison with combined data
    comparison_dir = results_dir / "cross_format_comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    # Comprehensive comparison - all loadings
    comprehensive_file = comparison_dir / "comprehensive_model_format_comparison.csv"
    loadings_df.to_csv(comprehensive_file, index=False)
    logger.info(f"Updated comprehensive comparison: {comprehensive_file} ({len(loadings_df)} rows)")
    
    # Format model summary - aggregate statistics
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
    logger.info(f"Updated format summary: {format_summary_file} ({len(format_summary)} rows)")
    
    return results_dir

def run_missing_factor_analysis(missing_analyses):
    """Run factor analysis for missing formats and models"""
    logger.info("Factor analysis for missing data would be implemented here")
    logger.info("Missing analyses:")
    for study, formats in missing_analyses.items():
        for format_name, models in formats.items():
            if models:
                logger.info(f"  {study} - {format_name}: {len(models)} models missing: {models}")
    
    logger.warning("Automatic factor analysis execution not yet implemented")
    logger.info("Please run factor analysis manually for missing data using factor_analysis_utils.py")

def main():
    """Main function to organize factor analysis results and identify missing data"""
    logger.info("=== Factor Analysis Results Organization ===")
    
    # Check what exists and what's missing
    missing, existing = identify_missing_analyses()
    
    logger.info("\n=== CURRENT STATUS ===")
    for study, formats in existing.items():
        logger.info(f"{study.upper()}:")
        for format_name, models in formats.items():
            logger.info(f"  {format_name}: {len(models)} models - {models}")
    
    logger.info("\n=== MISSING ANALYSES ===")
    total_missing = 0
    for study, formats in missing.items():
        study_missing = 0
        for format_name, models in formats.items():
            if models:
                model_count = len(models)
                total_missing += model_count
                study_missing += model_count
                logger.info(f"{study.upper()} - {format_name}: {model_count} models missing: {models}")
        if study_missing == 0:
            logger.info(f"{study.upper()}: ✅ All analyses complete")
    
    if total_missing > 0:
        logger.warning(f"\n⚠️  TOTAL MISSING: {total_missing} model-format combinations need factor analysis")
        logger.info("\n📋 PRIORITY MISSING:")
        logger.info("  1. Study 2 Likert Format - 5 models (complete format missing)")
        logger.info("  2. Study 3 Likert Format - 5 models (complete format missing)")
        logger.info("  3. Study 2 Expanded Format - 1 model (GPT-3.5-turbo)")
        logger.info("  4. Study 3 Binary Format - 2 models (DeepSeek-V3, Llama-3.3-70B)\n")
    else:
        logger.info("\n✅ All expected factor analyses are complete!")
    
    # Organize existing results
    logger.info("\n=== ORGANIZING EXISTING RESULTS ===")
    result_dir = organize_existing_results()
    
    if result_dir:
        logger.info(f"\n✅ Organization complete! Results in: {result_dir}")
        logger.info("\n📁 KEY FILES:")
        logger.info("  • Individual model CSVs: factor_analysis_results/study_X/format/model_factor_*.csv")
        logger.info("  • Cross-format comparison: factor_analysis_results/cross_format_comparison/")
        logger.info("  • Comprehensive data: comprehensive_model_format_comparison.csv")
        logger.info("  • Summary statistics: format_model_summary.csv")
    
    # Show simulation data available for missing analyses
    if total_missing > 0:
        logger.info("\n=== SIMULATION DATA AVAILABLE ===")
        sim_files = get_simulation_data_files()
        for study, formats in sim_files.items():
            logger.info(f"{study.upper()}:")
            for format_name, files in formats.items():
                logger.info(f"  {format_name}: {len(files)} JSON files available")
                if format_name in missing.get(study.lower().replace('_', '_'), {}):
                    logger.info(f"    → Ready for factor analysis")
    
    return result_dir, missing

if __name__ == "__main__":
    main()