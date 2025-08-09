#!/usr/bin/env python3
"""
Run Missing Factor Analysis for Likert Format Data

This script processes the missing Likert format factor analyses for Studies 2 and 3,
and any other missing model analyses identified by organize_all_factor_results.py.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from typing import Dict, List, Tuple, Any

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

class MissingFactorAnalysisRunner:
    """Run factor analysis for missing formats and models"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / "factor_analysis_results"
        self.factor_framework = FactorAnalysisFramework()
        
    def extract_model_name(self, filename: str) -> str:
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
    
    def load_simulation_data(self, json_file: Path) -> Tuple[pd.DataFrame, str]:
        """Load simulation data from JSON file and return processed DataFrame"""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract simulation results
            if 'simulation_results' in data:
                sim_data = data['simulation_results']
            elif isinstance(data, list):
                sim_data = data
            else:
                sim_data = data
            
            # Convert to DataFrame
            if isinstance(sim_data, list):
                df = pd.DataFrame(sim_data)
            else:
                df = pd.DataFrame([sim_data])
            
            # Get model name from filename
            model_name = self.extract_model_name(json_file.name)
            
            return df, model_name
            
        except Exception as e:
            logger.error(f"Error loading {json_file}: {e}")
            return pd.DataFrame(), "unknown_model"
    
    def process_study_format(self, study_name: str, format_name: str, json_files: List[Path]) -> Dict[str, Any]:
        """Process factor analysis for a specific study and format"""
        logger.info(f"Processing {study_name} {format_name} format - {len(json_files)} files")
        
        results = {}
        
        for json_file in json_files:
            logger.info(f"  Processing {json_file.name}")
            
            try:
                # Run factor analysis using the framework's built-in method
                factor_results = self.run_factor_analysis_for_file(json_file)
                
                if factor_results:
                    model_name = factor_results['model_name']
                    results[model_name] = factor_results
                else:
                    model_name = self.extract_model_name(json_file.name)
                    logger.warning(f"    ❌ Factor analysis failed for {model_name}")
                    
            except Exception as e:
                model_name = self.extract_model_name(json_file.name)
                logger.error(f"    ❌ Error processing {model_name}: {e}")
                continue
        
        return results
    
    def run_factor_analysis_for_file(self, json_file: Path) -> Dict[str, Any]:
        """Run factor analysis for a single JSON simulation file"""
        try:
            # Use the framework's built-in method to analyze simulation results
            results = self.factor_framework.analyze_simulation_results(json_file)
            
            if results:
                model_name = self.extract_model_name(json_file.name)
                results['model_name'] = model_name
                results['file_source'] = str(json_file)
                logger.info(f"    ✅ Completed factor analysis for {model_name}")
                return results
            else:
                logger.warning(f"    ❌ Factor analysis returned no results for {json_file.name}")
                return None
                
        except Exception as e:
            logger.error(f"Factor analysis failed for {json_file.name}: {e}")
            return None
    
    def save_factor_results(self, results: Dict[str, Any], study_name: str, format_name: str):
        """Save factor analysis results to CSV files"""
        if not results:
            logger.warning(f"No results to save for {study_name} {format_name}")
            return
        
        # Create directories
        study_dir = self.results_dir / study_name.lower()
        format_dir = study_dir / f"{format_name}_format"
        format_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each model's results
        for model_name, model_results in results.items():
            if not isinstance(model_results, dict):
                continue
                
            # Prepare data for CSV export
            loading_entries = []
            summary_entries = []
            
            # Process original and modified structures
            for structure_type in ['original', 'modified']:
                if structure_type not in model_results:
                    continue
                    
                structure_data = model_results[structure_type]
                if not isinstance(structure_data, dict) or 'domains' not in structure_data:
                    continue
                
                # Process each domain/factor
                for domain_name, domain_data in structure_data['domains'].items():
                    if not isinstance(domain_data, dict):
                        continue
                    
                    loadings = domain_data.get('loadings', {}) or domain_data.get('factor_loadings', {})
                    reliability = domain_data.get('reliability', {})
                    fit_indices = domain_data.get('fit_indices', {})
                    
                    # Create summary entry
                    summary_entry = {
                        'Study': study_name.upper(),
                        'Format': format_name,
                        'Model': model_name,
                        'Structure_Type': structure_type.title(),
                        'Factor_Domain': domain_name,
                        'N_Items': len(loadings) if loadings else 0,
                        'N_Participants': model_results.get('n_participants', 0),
                        'Alpha': reliability.get('alpha', 0),
                        'Omega': reliability.get('omega', 0),
                        'Eigenvalue': domain_data.get('eigenvalue', 0),
                        'Variance_Explained': domain_data.get('variance_explained', 0),
                        'Mean_Loading_Abs': np.mean([abs(v) for v in loadings.values()]) if loadings else 0,
                        'Max_Loading_Abs': max([abs(v) for v in loadings.values()]) if loadings else 0,
                        'Min_Loading_Abs': min([abs(v) for v in loadings.values()]) if loadings else 0,
                        'RMSEA': fit_indices.get('RMSEA', 0),
                        'CFI': fit_indices.get('CFI', 0),
                        'TLI': fit_indices.get('TLI', 0),
                        'SRMR': fit_indices.get('SRMR', 0),
                        'Total_Variance_Explained': structure_data.get('total_variance_explained', 0),
                        'N_Factors_Total': structure_data.get('n_factors', 0),
                        'File_Source': f"{study_name.lower()}_{format_name}_results_{model_name}"
                    }
                    summary_entries.append(summary_entry)
                    
                    # Create loading entries
                    for item_name, loading_value in loadings.items():
                        loading_entry = {
                            'Study': study_name.upper(),
                            'Format': format_name,
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
                            'N_Participants': model_results.get('n_participants', 0),
                            'RMSEA': fit_indices.get('RMSEA', 0),
                            'CFI': fit_indices.get('CFI', 0),
                            'TLI': fit_indices.get('TLI', 0),
                            'SRMR': fit_indices.get('SRMR', 0),
                            'File_Source': f"{study_name.lower()}_{format_name}_results_{model_name}"
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
    
    def run_missing_analyses(self):
        """Run factor analysis for all missing formats"""
        logger.info("=== Running Missing Factor Analysis ===")
        
        # Study 2 Likert format
        logger.info("\n--- Study 2 Likert Format ---")
        study_2_likert_dir = self.base_dir / "study_2" / "study_2_likert_results"
        if study_2_likert_dir.exists():
            json_files = list(study_2_likert_dir.glob("*.json"))
            json_files = [f for f in json_files if not f.name.startswith('.')]
            
            if json_files:
                results = self.process_study_format("Study_2", "likert", json_files)
                self.save_factor_results(results, "Study_2", "likert")
                logger.info(f"✅ Completed Study 2 Likert format - {len(results)} models processed")
            else:
                logger.warning("❌ No JSON files found in Study 2 Likert results")
        else:
            logger.warning("❌ Study 2 Likert results directory not found")
        
        # Study 3 Likert format  
        logger.info("\n--- Study 3 Likert Format ---")
        study_3_likert_dir = self.base_dir / "study_3" / "study_3_likert_results" 
        if study_3_likert_dir.exists():
            json_files = list(study_3_likert_dir.glob("*.json"))
            # Filter out metadata and backup files
            json_files = [f for f in json_files if not f.name.startswith('.') and 
                         'metadata' not in f.name.lower() and 'backup' not in f.name.lower()]
            
            if json_files:
                results = self.process_study_format("Study_3", "likert", json_files)
                self.save_factor_results(results, "Study_3", "likert")
                logger.info(f"✅ Completed Study 3 Likert format - {len(results)} models processed")
            else:
                logger.warning("❌ No valid JSON files found in Study 3 Likert results")
        else:
            logger.warning("❌ Study 3 Likert results directory not found")
        
        logger.info("\n=== Factor Analysis Complete ===")
        logger.info("Run organize_all_factor_results.py to update comprehensive results")

def main():
    """Main execution function"""
    runner = MissingFactorAnalysisRunner()
    runner.run_missing_analyses()

if __name__ == "__main__":
    main()