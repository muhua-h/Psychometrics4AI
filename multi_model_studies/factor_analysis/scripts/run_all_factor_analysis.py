#!/usr/bin/env python3
"""
Master Factor Analysis Runner

This script runs factor analysis for all formats (binary_simple, binary_elaborated, expanded, likert)
for both Studies 2 and 3 using the organized directory structure.
"""

import sys
from pathlib import Path
import logging

# Add the shared directory to path
sys.path.append(str(Path(__file__).parent.parent / "shared"))

# Import the factor analysis framework
from factor_analysis_utils import FactorAnalysisFramework

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_factor_analysis_for_format(study_name: str, format_name: str, base_dir: Path, results_dir: Path):
    """Run factor analysis for a specific study and format"""
    
    # Map format names to directory names
    dir_map = {
        'binary_simple': 'study_2_simple_binary_results' if study_name == 'Study_2' else 'study_3_binary_simple_results',
        'binary_elaborated': 'study_2_elaborated_binary_results' if study_name == 'Study_2' else 'study_3_binary_elaborated_results',
        'expanded': 'study_2_expanded_results' if study_name == 'Study_2' else 'study_3_expanded_results',
        'likert': 'study_2_likert_results' if study_name == 'Study_2' else 'study_3_likert_results'
    }
    
    study_dir = base_dir / study_name.lower()
    format_dir_name = dir_map[format_name]
    data_dir = study_dir / format_dir_name
    
    if not data_dir.exists():
        logger.warning(f"❌ {study_name} {format_name} directory not found: {data_dir}")
        return
    
    logger.info(f"\n--- {study_name} {format_name.title()} Format ---")
    logger.info(f"Processing: {data_dir}")
    
    # Initialize framework
    framework = FactorAnalysisFramework()
    
    # Run analysis
    results = framework.analyze_all_model_results(data_dir, pattern="*.json")
    
    if not results:
        logger.warning(f"❌ No results for {study_name} {format_name}")
        return
    
    # Filter out metadata and backup files
    filtered_results = {k: v for k, v in results.items() 
                       if 'metadata' not in k.lower() and 'backup' not in k.lower()}
    
    if not filtered_results:
        logger.warning(f"❌ No valid results for {study_name} {format_name}")
        return
    
    # Save results using the individual format runner
    from run_binary_simple_factor_analysis import convert_factor_results_to_csv
    convert_factor_results_to_csv(filtered_results, study_name, format_name, results_dir)
    
    logger.info(f"✅ Completed {study_name} {format_name} - {len(filtered_results)} models processed")

def main():
    """Run factor analysis for all formats and studies"""
    
    base_dir = Path(__file__).parent.parent.parent  # multi_model_studies root
    results_dir = base_dir / "factor_analysis" / "results"
    
    logger.info("=== Running Complete Factor Analysis Suite ===")
    
    studies = ["Study_2", "Study_3"]
    formats = ["binary_simple", "binary_elaborated", "expanded", "likert"]
    
    for study in studies:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {study}")
        logger.info(f"{'='*50}")
        
        for format_name in formats:
            try:
                run_factor_analysis_for_format(study, format_name, base_dir, results_dir)
            except Exception as e:
                logger.error(f"❌ Error processing {study} {format_name}: {e}")
                continue
    
    logger.info("\n" + "="*60)
    logger.info("🎉 All factor analysis completed!")
    logger.info("="*60)

if __name__ == "__main__":
    main()