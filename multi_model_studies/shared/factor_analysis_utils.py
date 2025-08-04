"""
Factor Analysis Utilities for Multi-Model Psychometrics Studies

This module provides a unified framework for performing factor analysis on multi-model 
personality simulation results, replicating and extending the original R-based analysis.

Features:
- CFA (Confirmatory Factor Analysis) using Python's semopy
- Domain-specific factor models for Big Five personality traits
- Robust error handling for models that fail to converge
- Near-zero variance detection and handling
- Multi-model result processing and comparison
- Compatible with original R analysis results

Author: Multi-Model Extension Team
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# Import required libraries for factor analysis
try:
    import semopy
    from semopy import Model
    from semopy.stats import fit_indices
except ImportError:
    print("Warning: semopy not installed. Install with: pip install semopy")
    semopy = None

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import FactorAnalysis
    import scipy.stats as stats
except ImportError:
    print("Warning: sklearn/scipy not installed for fallback factor analysis")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactorAnalysisFramework:
    """
    Unified framework for factor analysis across multiple models and formats.
    
    This class replicates the original R-based factor analysis methodology
    while extending it to handle multi-model results systematically.
    """
    
    # Define Big Five personality domains and their corresponding Mini-Marker items
    BIG_FIVE_DOMAINS = {
        'Extraversion': {
            'positive': ['Bold', 'Energetic', 'Extraverted', 'Talkative'],
            'negative': ['Bashful', 'Quiet', 'Shy', 'Withdrawn']
        },
        'Agreeableness': {
            'positive': ['Cooperative', 'Kind', 'Sympathetic', 'Warm'],
            'negative': ['Cold', 'Harsh', 'Rude', 'Unsympathetic']
        },
        'Conscientiousness': {
            'positive': ['Efficient', 'Organized', 'Practical', 'Systematic'],
            'negative': ['Careless', 'Disorganized', 'Inefficient', 'Sloppy']
        },
        'Neuroticism': {
            'positive': ['Envious', 'Fretful', 'Jealous', 'Moody', 'Temperamental', 'Touchy'],
            'negative': ['Relaxed', 'Unenvious']
        },
        'Openness': {
            'positive': ['Complex', 'Creative', 'Deep', 'Imaginative', 'Intellectual', 'Philosophical'],
            'negative': ['Uncreative', 'Unintellectual']
        }
    }
    
    # Revised Neuroticism items (as used in original analysis)
    NEUROTICISM_REVISED = ['Jealous', 'Fretful', 'Moody', 'Temperamental', 'Touchy', 'Relaxed', 'Unenvious']
    
    def __init__(self, use_semopy: bool = True):
        """
        Initialize the factor analysis framework.
        
        Args:
            use_semopy: Whether to use semopy for CFA (True) or fallback to sklearn (False)
        """
        self.use_semopy = use_semopy and semopy is not None
        if not self.use_semopy:
            logger.warning("Using fallback factor analysis methods (sklearn-based)")
    
    def reverse_code_items(self, data: pd.DataFrame, scale_range: Tuple[int, int] = (1, 9)) -> pd.DataFrame:
        """
        Apply reverse coding to negative items in Mini-Marker scale.
        
        Args:
            data: DataFrame with Mini-Marker item responses
            scale_range: (min, max) values for the response scale
            
        Returns:
            DataFrame with reverse-coded items
        """
        data_reversed = data.copy()
        
        # Get all negative items across domains
        negative_items = []
        for domain_items in self.BIG_FIVE_DOMAINS.values():
            negative_items.extend(domain_items['negative'])
        
        # Apply reverse coding: new_value = (scale_max + scale_min) - old_value
        scale_min, scale_max = scale_range
        for item in negative_items:
            if item in data_reversed.columns:
                data_reversed[item] = (scale_max + scale_min) - data_reversed[item]
        
        logger.info(f"Reverse coded {len(negative_items)} negative items")
        return data_reversed
    
    def check_and_remove_near_zero_variance(self, data: pd.DataFrame, 
                                          threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
        """
        Identify and remove variables with near-zero variance.
        
        Args:
            data: Input DataFrame
            threshold: Variance threshold below which variables are removed
            
        Returns:
            Tuple of (cleaned_data, removed_variables)
        """
        variances = data.var()
        near_zero_var = variances[variances < threshold].index.tolist()
        
        if near_zero_var:
            logger.warning(f"Removing {len(near_zero_var)} variables with near-zero variance: {near_zero_var}")
            data_cleaned = data.drop(columns=near_zero_var)
        else:
            data_cleaned = data.copy()
            
        return data_cleaned, near_zero_var
    
    def load_simulation_results(self, result_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load simulation results from JSON file and convert to DataFrame.
        
        Args:
            result_path: Path to JSON file containing simulation results
            
        Returns:
            DataFrame with participants as rows and Mini-Marker items as columns
        """
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Ensure all expected Mini-Marker items are present
        expected_items = []
        for domain_items in self.BIG_FIVE_DOMAINS.values():
            expected_items.extend(domain_items['positive'])
            expected_items.extend(domain_items['negative'])
        
        missing_items = set(expected_items) - set(df.columns)
        if missing_items:
            logger.warning(f"Missing expected items in {result_path}: {missing_items}")
        
        logger.info(f"Loaded {len(df)} participants from {result_path}")
        return df
    
    def create_domain_model_semopy(self, domain: str, items: List[str]) -> str:
        """
        Create a CFA model specification for semopy.
        
        Args:
            domain: Name of the personality domain
            items: List of item names for this domain
            
        Returns:
            Model specification string for semopy
        """
        # Create latent variable specification
        model_spec = f"{domain} =~ " + " + ".join(items)
        return model_spec
    
    def fit_single_domain_cfa_semopy(self, domain: str, items: List[str], 
                                   data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit a single-domain CFA model using semopy.
        
        Args:
            domain: Name of the personality domain
            items: List of item names for the domain
            data: DataFrame containing the item responses
            
        Returns:
            Dictionary containing fit results, loadings, and reliability
        """
        try:
            # Check if all items are available
            available_items = [item for item in items if item in data.columns]
            if len(available_items) < 3:
                return {'error': f'Insufficient items available for {domain}: {available_items}'}
            
            # Create model specification
            model_spec = self.create_domain_model_semopy(domain, available_items)
            
            # Fit the model
            model = Model(model_spec)
            model.fit(data[available_items])
            
            # Check convergence
            if not model.optimizer.success:
                return {'error': 'Model did not converge'}
            
            # Get fit indices
            fit_stats = fit_indices(model)
            
            # Get standardized loadings
            loadings = model.inspect(std_est=True)
            factor_loadings = loadings[loadings['op'] == '=~'][['rhs', 'Std. Est']].copy()
            factor_loadings.columns = ['item', 'loading']
            
            # Calculate reliability (Cronbach's alpha approximation)
            item_data = data[available_items]
            alpha = self.calculate_cronbach_alpha(item_data)
            
            return {
                'fit_measures': {
                    'chisq': fit_stats.get('chi2', None),
                    'df': fit_stats.get('df', None),
                    'pvalue': fit_stats.get('p', None),
                    'cfi': fit_stats.get('CFI', None),
                    'tli': fit_stats.get('TLI', None),
                    'rmsea': fit_stats.get('RMSEA', None),
                    'srmr': fit_stats.get('SRMR', None)
                },
                'factor_loadings': factor_loadings,
                'reliability': alpha,
                'items_used': available_items,
                'converged': True
            }
            
        except Exception as e:
            return {'error': f'Error in fitting model for {domain}: {str(e)}'}
    
    def calculate_cronbach_alpha(self, data: pd.DataFrame) -> float:
        """
        Calculate Cronbach's alpha reliability coefficient.
        
        Args:
            data: DataFrame with item responses
            
        Returns:
            Cronbach's alpha value
        """
        # Remove rows with missing data
        data_clean = data.dropna()
        
        if len(data_clean) < 2 or data_clean.shape[1] < 2:
            return np.nan
        
        # Calculate Cronbach's alpha
        n_items = data_clean.shape[1]
        item_variances = data_clean.var(axis=0, ddof=1).sum()
        total_variance = data_clean.sum(axis=1).var(ddof=1)
        
        if total_variance == 0:
            return np.nan
        
        alpha = (n_items / (n_items - 1)) * (1 - item_variances / total_variance)
        return alpha
    
    def fit_single_domain_cfa_fallback(self, domain: str, items: List[str], 
                                     data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fallback factor analysis using sklearn when semopy is not available.
        
        Args:
            domain: Name of the personality domain
            items: List of item names for the domain
            data: DataFrame containing the item responses
            
        Returns:
            Dictionary containing basic factor analysis results
        """
        try:
            # Check if all items are available
            available_items = [item for item in items if item in data.columns]
            if len(available_items) < 2:
                return {'error': f'Insufficient items available for {domain}: {available_items}'}
            
            item_data = data[available_items].dropna()
            if len(item_data) < 3:
                return {'error': f'Insufficient valid responses for {domain}'}
            
            # Standardize the data
            scaler = StandardScaler()
            data_std = scaler.fit_transform(item_data)
            
            # Perform factor analysis
            fa = FactorAnalysis(n_components=1, random_state=42)
            fa.fit(data_std)
            
            # Get loadings
            loadings = pd.DataFrame({
                'item': available_items,
                'loading': fa.components_[0]
            })
            
            # Calculate reliability
            alpha = self.calculate_cronbach_alpha(item_data)
            
            # Basic fit measures (limited without full SEM)
            return {
                'fit_measures': {
                    'loglik': fa.loglike_[-1] if hasattr(fa, 'loglike_') else None,
                    'noise_variance': fa.noise_variance_.mean()
                },
                'factor_loadings': loadings,
                'reliability': alpha,
                'items_used': available_items,
                'converged': True,
                'method': 'sklearn_fallback'
            }
            
        except Exception as e:
            return {'error': f'Error in fallback factor analysis for {domain}: {str(e)}'}
    
    def fit_single_domain_cfa(self, domain: str, items: List[str], 
                            data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit a single-domain CFA model using the available method.
        
        Args:
            domain: Name of the personality domain
            items: List of item names for the domain
            data: DataFrame containing the item responses
            
        Returns:
            Dictionary containing fit results, loadings, and reliability
        """
        if self.use_semopy:
            return self.fit_single_domain_cfa_semopy(domain, items, data)
        else:
            return self.fit_single_domain_cfa_fallback(domain, items, data)
    
    def analyze_simulation_results(self, result_path: Union[str, Path], 
                                 use_revised_neuroticism: bool = True) -> Dict[str, Any]:
        """
        Perform complete factor analysis on simulation results.
        
        Args:
            result_path: Path to simulation results JSON file
            use_revised_neuroticism: Whether to use revised Neuroticism items
            
        Returns:
            Dictionary containing analysis results for all domains
        """
        logger.info(f"Analyzing simulation results from {result_path}")
        
        # Load and preprocess data
        data = self.load_simulation_results(result_path)
        data_reversed = self.reverse_code_items(data)
        data_clean, removed_vars = self.check_and_remove_near_zero_variance(data_reversed)
        
        results = {
            'file_info': {
                'path': str(result_path),
                'n_participants': len(data),
                'removed_variables': removed_vars
            },
            'domain_results': {}
        }
        
        # Analyze each domain
        for domain, domain_items in self.BIG_FIVE_DOMAINS.items():
            # Combine positive and negative items
            all_items = domain_items['positive'] + domain_items['negative']
            
            # Use revised Neuroticism items if specified
            if domain == 'Neuroticism' and use_revised_neuroticism:
                all_items = self.NEUROTICISM_REVISED
            
            logger.info(f"Analyzing {domain} with {len(all_items)} items")
            domain_result = self.fit_single_domain_cfa(domain, all_items, data_clean)
            results['domain_results'][domain] = domain_result
        
        return results
    
    def analyze_all_model_results(self, study_dir: Union[str, Path], 
                                pattern: str = "*.json") -> Dict[str, Any]:
        """
        Analyze factor structure across all model results in a study directory.
        
        Args:
            study_dir: Directory containing simulation result files
            pattern: File pattern to match (default: "*.json")
            
        Returns:
            Dictionary containing analysis results for all models
        """
        study_path = Path(study_dir)
        result_files = list(study_path.glob(pattern))
        
        if not result_files:
            logger.warning(f"No result files found in {study_dir} matching {pattern}")
            return {}
        
        logger.info(f"Found {len(result_files)} result files to analyze")
        
        all_results = {}
        for result_file in result_files:
            model_name = result_file.stem
            logger.info(f"Analyzing {model_name}")
            
            try:
                model_results = self.analyze_simulation_results(result_file)
                all_results[model_name] = model_results
            except Exception as e:
                logger.error(f"Failed to analyze {model_name}: {str(e)}")
                all_results[model_name] = {'error': str(e)}
        
        return all_results
    
    def create_summary_report(self, analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a summary report of factor analysis results across models.
        
        Args:
            analysis_results: Results from analyze_all_model_results
            
        Returns:
            DataFrame with summary statistics for each model and domain
        """
        summary_data = []
        
        for model_name, model_results in analysis_results.items():
            if 'error' in model_results:
                continue
                
            for domain, domain_results in model_results.get('domain_results', {}).items():
                if 'error' in domain_results:
                    row = {
                        'model': model_name,
                        'domain': domain,
                        'converged': False,
                        'reliability': np.nan,
                        'n_items': 0,
                        'error': domain_results['error']
                    }
                else:
                    fit_measures = domain_results.get('fit_measures', {})
                    row = {
                        'model': model_name,
                        'domain': domain,
                        'converged': domain_results.get('converged', False),
                        'reliability': domain_results.get('reliability', np.nan),
                        'n_items': len(domain_results.get('items_used', [])),
                        'cfi': fit_measures.get('cfi', np.nan),
                        'tli': fit_measures.get('tli', np.nan),
                        'rmsea': fit_measures.get('rmsea', np.nan),
                        'srmr': fit_measures.get('srmr', np.nan),
                        'chisq': fit_measures.get('chisq', np.nan),
                        'pvalue': fit_measures.get('pvalue', np.nan)
                    }
                
                summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def save_results(self, results: Dict[str, Any], output_path: Union[str, Path]):
        """
        Save analysis results to JSON file.
        
        Args:
            results: Analysis results dictionary
            output_path: Output file path
        """
        # Convert any numpy objects to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            return obj
        
        # Deep convert the results
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        results_converted = deep_convert(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_converted, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """
    Example usage of the FactorAnalysisFramework.
    """
    # Initialize the framework
    fa_framework = FactorAnalysisFramework()
    
    # Example: Analyze a single result file
    result_path = "path/to/simulation/results.json"
    if Path(result_path).exists():
        results = fa_framework.analyze_simulation_results(result_path)
        print("Single file analysis completed")
    
    # Example: Analyze all results in a directory
    study_dir = "path/to/study/results"
    if Path(study_dir).exists():
        all_results = fa_framework.analyze_all_model_results(study_dir)
        summary_df = fa_framework.create_summary_report(all_results)
        print("Multi-model analysis completed")
        print(summary_df.head())


if __name__ == "__main__":
    main()