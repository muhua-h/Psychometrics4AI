#!/usr/bin/env python3
"""
Run R-based CFA analysis for all simulation results
This script uses the R-based CFA to generate proper fit indices
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
R_SCRIPT_PATH = Path(__file__).parent / "cfa_analysis.R"
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"

# Map format names to expected directory names
FORMAT_MAPPING = {
    'binary_simple': ['binary_simple', 'simple_binary'],
    'binary_elaborated': ['binary_elaborated', 'elaborated_binary'], 
    'expanded': ['expanded'],
    'likert': ['likert']
}

# Map model names to expected file patterns
MODEL_PATTERNS = {
    'gpt_4': ['gpt_4', 'gpt-4'],
    'gpt_4o': ['gpt_4o', 'gpt-4o'],
    'gpt_3.5_turbo': ['gpt_3.5_turbo', 'gpt-3.5-turbo', 'openai_gpt_3.5'],
    'llama_3.3_70b': ['llama_3.3_70b', 'llama-3.3-70b', 'llama'],
    'deepseek_v3': ['deepseek_v3', 'deepseek-v3', 'deepseek']
}

def find_json_files(study_dir: Path):
    """Find all JSON simulation result files"""
    json_files = []
    
    for format_dir in study_dir.glob("*_results*"):
        format_name = format_dir.name.replace('_results', '').replace('_', '')
        
        # Map to standard format name
        for standard_name, patterns in FORMAT_MAPPING.items():
            if any(pattern in format_dir.name.lower() for pattern in patterns):
                format_name = standard_name
                break
        
        # Find JSON files
        for json_file in format_dir.glob("*.json"):
            # Determine model name from filename
            model_name = None
            filename_lower = json_file.stem.lower()
            
            for standard_model, patterns in MODEL_PATTERNS.items():
                if any(pattern.lower() in filename_lower for pattern in patterns):
                    model_name = standard_model
                    break
            
            if model_name:
                json_files.append({
                    'path': json_file,
                    'study': study_dir.name.upper(),
                    'format': format_name,
                    'model': model_name
                })
    
    return json_files

def run_r_cfa(json_path: Path, output_dir: Path, scale_range=(1, 9)):
    """Run R-based CFA analysis for a single JSON file"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "Rscript", 
        str(R_SCRIPT_PATH),
        str(json_path),
        str(output_dir),
        str(scale_range[0]),
        str(scale_range[1])
    ]
    
    try:
        logger.info(f"Running R CFA for {json_path.name}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.stdout:
            logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
            
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"R script failed for {json_path.name}: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def process_study(study_name: str):
    """Process all results for a specific study"""
    study_dir = RESULTS_DIR / study_name.lower()
    if not study_dir.exists():
        logger.warning(f"Study directory not found: {study_dir}")
        return
    
    logger.info(f"Processing {study_name} results...")
    
    json_files = find_json_files(study_dir)
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    for file_info in json_files:
        # Determine output directory based on format
        format_dir = study_dir / f"{file_info['format']}_format"
        format_dir.mkdir(exist_ok=True)
        
        # Run R CFA analysis
        success = run_r_cfa(
            file_info['path'],
            format_dir
        )
        
        if success:
            logger.info(f"Successfully analyzed {file_info['model']} in {file_info['format']} format")
        else:
            logger.error(f"Failed to analyze {file_info['model']} in {file_info['format']} format")

def main():
    """Main function to run R-based CFA analysis for all studies"""
    
    logger.info("=== Starting R-based CFA Analysis ===")
    
    # Check if Rscript is available
    try:
        subprocess.run(["Rscript", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Rscript not found. Please install R and ensure it's in PATH")
        sys.exit(1)
    
    # Check if required R packages are available
    try:
        subprocess.run([
            "Rscript", "-e", 
            "suppressPackageStartupMessages({library(lavaan); library(psych)})"
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        logger.error("Required R packages (lavaan, psych) not found. Install with:")
        logger.error("R -e 'install.packages(c(\"lavaan\", \"psych\"))'")
        sys.exit(1)
    
    # Process each study
    studies = ['study_2', 'study_3']
    
    for study in studies:
        process_study(study)
    
    logger.info("=== R-based CFA Analysis Complete ===")

if __name__ == "__main__":
    main()