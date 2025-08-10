#!/usr/bin/env python3
"""
Enhanced R-based CFA analysis runner
Uses the enhanced R script to provide comprehensive factor analysis
with proper fit indices and reliability measures
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import logging
import sys
import os
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
R_SCRIPT_PATH = Path(__file__).parent / "cfa_analysis_enhanced.R"
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
R_RESULTS_DIR = BASE_DIR / "results_r"

# Ensure R results directory exists
R_RESULTS_DIR.mkdir(exist_ok=True)

class EnhancedRCFAAnalyzer:
    """Enhanced R-based CFA analysis wrapper"""
    
    def __init__(self):
        self.required_r_packages = ["lavaan", "psych", "semTools", "jsonlite", "dplyr", "tidyr", "purrr", "readr"]
        self.validate_environment()
    
    def validate_environment(self):
        """Validate that R and required packages are available"""
        try:
            # Check Rscript availability
            subprocess.run(["Rscript", "--version"], capture_output=True, check=True)
            logger.info("✓ Rscript is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("❌ Rscript not found. Please install R and ensure it's in PATH")
            sys.exit(1)
        
        # Check required packages
        missing_packages = []
        for package in self.required_r_packages:
            try:
                subprocess.run([
                    "Rscript", "-e", 
                    f"suppressPackageStartupMessages(library('{package}'))"
                ], check=True, capture_output=True)
                logger.info(f"✓ {package} package is available")
            except subprocess.CalledProcessError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"❌ Missing R packages: {', '.join(missing_packages)}")
            logger.error("Install with: R -e 'install.packages(c(\"lavaan\", \"psych\", \"semTools\", \"jsonlite\", \"dplyr\", \"tidyr\", \"purrr\", \"readr\"))'")
            sys.exit(1)
    
    def find_json_files(self) -> List[Dict[str, Any]]:
        """Find all JSON simulation result files across studies"""
        json_files = []
        
        # Process study directories
        for study_dir in RESULTS_DIR.glob("study_*"):
            if not study_dir.is_dir():
                continue
                
            study_name = study_dir.name.upper()
            
            # Find format directories
            for format_dir in study_dir.glob("*_format"):
                format_name = format_dir.name.replace('_format', '').replace('_', '')
                
                # Map to standard format names
                format_mapping = {
                    'binarysimple': 'binary_simple',
                    'binaryelaborated': 'binary_elaborated', 
                    'expanded': 'expanded',
                    'likert': 'likert'
                }
                
                format_name = format_mapping.get(format_name, format_name)
                
                # Find JSON files
                for json_file in format_dir.glob("*.json"):
                    # Determine model name from filename
                    model_name = self.extract_model_name(json_file.stem)
                    
                    if model_name:
                        json_files.append({
                            'path': json_file,
                            'study': study_name,
                            'format': format_name,
                            'model': model_name,
                            'relative_path': str(json_file.relative_to(RESULTS_DIR))
                        })
        
        return json_files
    
    def extract_model_name(self, filename: str) -> str:
        """Extract standardized model name from filename"""
        filename_lower = filename.lower()
        
        model_patterns = {
            'gpt_4': ['gpt_4', 'gpt-4', 'openai_gpt_4'],
            'gpt_4o': ['gpt_4o', 'gpt-4o', 'openai_gpt_4o'],
            'gpt_3.5_turbo': ['gpt_3.5_turbo', 'gpt-3.5-turbo', 'openai_gpt_3.5'],
            'llama_3.3_70b': ['llama_3.3_70b', 'llama-3.3-70b', 'llama'],
            'deepseek_v3': ['deepseek_v3', 'deepseek-v3', 'deepseek']
        }
        
        for standard_model, patterns in model_patterns.items():
            if any(pattern in filename_lower for pattern in patterns):
                return standard_model
        
        return None
    
    def determine_scale_range(self, json_path: Path) -> tuple:
        """Determine appropriate scale range based on data"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Flatten data to get all values
            if isinstance(data, dict):
                values = []
                for v in data.values():
                    if isinstance(v, dict):
                        values.extend(v.values())
                    elif isinstance(v, list):
                        values.extend(v)
                    else:
                        values.append(v)
            else:
                values = list(data.values()) if hasattr(data, 'values') else list(data)
            
            # Determine scale range
            all_values = [v for v in values if isinstance(v, (int, float))]
            min_val, max_val = min(all_values), max(all_values)
            
            if max_val <= 5:
                return (1, 5)
            elif max_val <= 7:
                return (1, 7)
            else:
                return (1, 9)
                
        except Exception as e:
            logger.warning(f"Could not determine scale range for {json_path}: {e}")
            return (1, 9)
    
    def run_single_analysis(self, json_file: Dict[str, Any]) -> bool:
        """Run R CFA analysis for a single JSON file"""
        
        # Create output directory structure
        output_dir = R_RESULTS_DIR / json_file['study'].lower() / f"{json_file['format']}_format"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine scale range
        scale_range = self.determine_scale_range(json_file['path'])
        
        # Build command
        cmd = [
            "Rscript", 
            str(R_SCRIPT_PATH),
            "single",
            str(json_file['path']),
            str(output_dir),
            str(scale_range[0]),
            str(scale_range[1])
        ]
        
        try:
            logger.info(f"Analyzing {json_file['study']} - {json_file['format']} - {json_file['model']}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            if result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Analysis failed for {json_file['path']}: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
    
    def run_batch_analysis(self) -> None:
        """Run analysis for all JSON files"""
        
        logger.info("=== Starting Enhanced R-based CFA Analysis ===")
        
        # Find all JSON files
        json_files = self.find_json_files()
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        if not json_files:
            logger.warning("No JSON files found for analysis")
            return
        
        # Process each file
        successful = 0
        failed = 0
        
        for json_file in json_files:
            if self.run_single_analysis(json_file):
                successful += 1
            else:
                failed += 1
        
        logger.info(f"=== Analysis Complete ===")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total: {len(json_files)}")
        
        # Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self) -> None:
        """Generate comprehensive summary report"""
        
        try:
            # Find all R results
            summary_files = list(R_RESULTS_DIR.glob("**/"*_factor_summary_R.csv"))
            
            if not summary_files:
                logger.warning("No R results found for summary")
                return
            
            # Combine all results
            all_results = []
            for file in summary_files:
                df = pd.read_csv(file)
                all_results.append(df)
            
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=True)
                
                # Save comprehensive summary
                summary_file = R_RESULTS_DIR / "comprehensive_factor_analysis_summary_R.csv"
                combined_results.to_csv(summary_file, index=False)
                
                # Generate summary statistics
                summary_stats = combined_results.groupby(['Study', 'Format', 'Model']).agg({
                    'Alpha': ['mean', 'std'],
                    'Omega': ['mean', 'std'],
                    'CFI': ['mean', 'std'],
                    'RMSEA': ['mean', 'std'],
                    'SRMR': ['mean', 'std'],
                    'N_Items': 'sum',
                    'N_Participants': 'mean'
                }).round(3)
                
                stats_file = R_RESULTS_DIR / "summary_statistics_R.csv"
                summary_stats.to_csv(stats_file)
                
                logger.info(f"Comprehensive summary saved to: {summary_file}")
                logger.info(f"Summary statistics saved to: {stats_file}")
        
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")

def main():
    """Main execution function"""
    
    analyzer = EnhancedRCFAAnalyzer()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--single" and len(sys.argv) >= 3:
            # Single file mode
            json_path = Path(sys.argv[2])
            if json_path.exists():
                model_name = analyzer.extract_model_name(json_path.stem)
                study = "STUDY_2"  # Default, adjust based on path
                format = "expanded"  # Default
                
                json_file = {
                    'path': json_path,
                    'study': study,
                    'format': format,
                    'model': model_name or 'unknown'
                }
                
                analyzer.run_single_analysis(json_file)
            else:
                logger.error(f"File not found: {json_path}")
        else:
            logger.error("Invalid arguments. Use --single <json_path> for single file analysis")
    else:
        # Batch mode (default)
        analyzer.run_batch_analysis()

if __name__ == "__main__":
    main()