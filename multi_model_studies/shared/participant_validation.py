#!/usr/bin/env python3
"""
Participant validation utilities for detecting and handling participant loss
during Mini-Marker simulation and analysis.

This module provides functions to:
1. Validate simulation results for completeness
2. Detect participants with invalid or missing data
3. Generate detailed reports on participant loss
4. Provide recovery suggestions for failed participants
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Mini-Marker domain mapping (same as used in analysis)
MINIMARKER_DOMAIN_MAPPING = {
    # Extraversion (E)
    'Bashful': 'E', 'Bold': 'E', 'Energetic': 'E', 'Extraverted': 'E',
    'Quiet': 'E', 'Shy': 'E', 'Talkative': 'E', 'Withdrawn': 'E',
    # Agreeableness (A)
    'Cold': 'A', 'Cooperative': 'A', 'Envious': 'A', 'Harsh': 'A',
    'Jealous': 'A', 'Kind': 'A', 'Rude': 'A', 'Sympathetic': 'A', 
    'Unsympathetic': 'A', 'Warm': 'A',
    # Conscientiousness (C)
    'Careless': 'C', 'Disorganized': 'C', 'Efficient': 'C', 'Inefficient': 'C',
    'Organized': 'C', 'Practical': 'C', 'Sloppy': 'C', 'Systematic': 'C',
    # Neuroticism (N)
    'Fretful': 'N', 'Moody': 'N', 'Relaxed': 'N', 'Temperamental': 'N', 'Touchy': 'N',
    # Openness (O)
    'Complex': 'O', 'Creative': 'O', 'Deep': 'O', 'Imaginative': 'O',
    'Intellectual': 'O', 'Philosophical': 'O', 'Uncreative': 'O', 
    'Unenvious': 'O', 'Unintellectual': 'O'
}

REVERSE_CODED_TRAITS = {
    'Bashful', 'Quiet', 'Shy', 'Withdrawn',  # Extraversion (reverse)
    'Cold', 'Harsh', 'Rude', 'Unsympathetic',  # Agreeableness (reverse)
    'Careless', 'Disorganized', 'Inefficient', 'Sloppy',  # Conscientiousness (reverse)
    'Relaxed',  # Neuroticism (reverse)
    'Uncreative', 'Unintellectual'  # Openness (reverse)
}

class ParticipantValidationReport:
    """Holds validation results for a simulation."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.initial_count = 0
        self.after_cleaning = 0
        self.after_aggregation = 0
        self.after_alignment = 0
        self.final_count = 0
        self.cleaning_issues = 0
        self.parsing_errors = 0
        self.missing_traits = 0
        self.invalid_participants = []
        self.loss_breakdown = {}
    
    def print_summary(self):
        """Print a detailed summary of participant loss."""
        print(f"\n{'='*60}")
        print(f"VALIDATION REPORT: {self.model_name.upper()}")
        print(f"{'='*60}")
        print(f"Initial participants: {self.initial_count}")
        print(f"After key cleaning: {self.after_cleaning}")
        print(f"After aggregation: {self.after_aggregation}")
        print(f"After alignment: {self.after_alignment}")
        print(f"Final valid count: {self.final_count}")
        print(f"Total lost: {self.initial_count - self.final_count}")
        
        if self.cleaning_issues > 0:
            print(f"Cleaning issues: {self.cleaning_issues}")
        if self.parsing_errors > 0:
            print(f"Parsing errors: {self.parsing_errors}")
        if self.missing_traits > 0:
            print(f"Missing trait values: {self.missing_traits}")
        
        if self.invalid_participants:
            print(f"Invalid participants: {len(self.invalid_participants)}")
            for i, (idx, issues) in enumerate(self.invalid_participants[:5]):
                print(f"  Participant {idx}: {', '.join(issues)}")
            if len(self.invalid_participants) > 5:
                print(f"  ... and {len(self.invalid_participants) - 5} more")
        
        if self.loss_breakdown:
            print("Loss breakdown:")
            for reason, count in self.loss_breakdown.items():
                print(f"  {reason}: {count}")

def validate_simulation_completeness(
    simulation_results: List[Dict[str, Any]], 
    model_name: str,
    expected_count: Optional[int] = None
) -> ParticipantValidationReport:
    """
    Validate a simulation result for completeness and data quality.
    
    Args:
        simulation_results: List of simulation results (JSON responses)
        model_name: Name of the model for reporting
        expected_count: Expected number of participants (optional)
    
    Returns:
        ParticipantValidationReport with detailed validation results
    """
    report = ParticipantValidationReport(model_name)
    report.initial_count = len(simulation_results)
    
    if expected_count and report.initial_count != expected_count:
        print(f"Warning: Expected {expected_count} participants, got {report.initial_count}")
    
    # Step 1: Clean keys and check for missing traits
    cleaned_results = []
    
    for i, item in enumerate(simulation_results):
        if isinstance(item, dict) and 'error' in item:
            # Skip failed participants
            continue
            
        cleaned_item = {}
        
        # Clean keys (remove leading whitespace, numbering)
        for k, v in item.items():
            k_clean = k.lstrip()  # Remove leading whitespace/newlines
            if '. ' in k_clean:
                k_clean = k_clean.split('. ', 1)[1]
            cleaned_item[k_clean] = v
        
        # Check for missing traits
        expected_traits = set(MINIMARKER_DOMAIN_MAPPING.keys())
        found_traits = set(cleaned_item.keys())
        missing_traits = expected_traits - found_traits
        
        if missing_traits:
            report.cleaning_issues += 1
        
        cleaned_results.append(cleaned_item)
    
    report.after_cleaning = len(cleaned_results)
    
    # Step 2: Aggregate to domain scores and detect parsing errors
    if cleaned_results:
        sim_df = pd.DataFrame(cleaned_results)
        report.after_aggregation = len(sim_df)
        
        # Check for empty rows
        empty_rows = sim_df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            report.loss_breakdown['Empty rows'] = empty_rows
        
        # Aggregate and check for parsing errors
        aggregated_df = aggregate_minimarker_with_validation(sim_df, report)
        
        # Step 3: Check for invalid values (NaN, inf)
        nan_rows = aggregated_df.isna().any(axis=1)
        inf_rows = np.isinf(aggregated_df.values).any(axis=1)
        invalid_rows = nan_rows | inf_rows
        
        if invalid_rows.any():
            invalid_indices = aggregated_df.index[invalid_rows].tolist()
            for idx in invalid_indices:
                row_data = aggregated_df.iloc[idx]
                issues = []
                for col in row_data.index:
                    if pd.isna(row_data[col]):
                        issues.append(f"{col}=NaN")
                    elif np.isinf(row_data[col]):
                        issues.append(f"{col}=inf")
                report.invalid_participants.append((idx, issues))
        
        # Final valid count
        valid_mask = ~invalid_rows
        report.final_count = sum(valid_mask)
        
        # Calculate loss breakdown
        if report.initial_count > report.after_aggregation:
            report.loss_breakdown['Aggregation issues'] = report.initial_count - report.after_aggregation
        if report.after_aggregation > report.final_count:
            report.loss_breakdown['Invalid data filtering'] = report.after_aggregation - report.final_count
    
    return report

def aggregate_minimarker_with_validation(df: pd.DataFrame, report: ParticipantValidationReport) -> pd.DataFrame:
    """
    Aggregate Mini-Marker responses to domain scores with validation tracking.
    Same logic as the original analysis but with error tracking.
    """
    domain_scores = {d: [] for d in ['E', 'A', 'C', 'N', 'O']}
    
    for idx, row in df.iterrows():
        trait_by_domain = {d: [] for d in ['E', 'A', 'C', 'N', 'O']}
        
        for trait, value in row.items():
            if trait not in MINIMARKER_DOMAIN_MAPPING:
                continue
            domain = MINIMARKER_DOMAIN_MAPPING[trait]

            # Cast value to int if it's a string
            if isinstance(value, str):
                try:
                    value = int(value)
                except ValueError:
                    report.parsing_errors += 1
                    continue
            
            # Check for NaN values
            if pd.isna(value):
                report.missing_traits += 1
                continue
                
            # Reverse code if needed
            if trait in REVERSE_CODED_TRAITS:
                value = 10 - value
            trait_by_domain[domain].append(value)
        
        # Aggregate by domain
        for d in trait_by_domain:
            if trait_by_domain[d]:
                domain_scores[d].append(sum(trait_by_domain[d]))
            else:
                domain_scores[d].append(np.nan)
    
    return pd.DataFrame(domain_scores)

def get_failed_participant_indices(simulation_results: List[Dict[str, Any]]) -> List[int]:
    """
    Get indices of participants that failed during simulation.
    
    Args:
        simulation_results: List of simulation results
    
    Returns:
        List of indices for failed participants
    """
    failed_indices = []
    for i, result in enumerate(simulation_results):
        if isinstance(result, dict) and 'error' in result:
            failed_indices.append(i)
    return failed_indices

def validate_all_simulation_results(
    results_dir: Path, 
    expected_participant_count: Optional[int] = None
) -> Dict[str, ParticipantValidationReport]:
    """
    Validate all simulation results in a directory.
    
    Args:
        results_dir: Directory containing simulation JSON files
        expected_participant_count: Expected number of participants
    
    Returns:
        Dictionary mapping model names to validation reports
    """
    import glob
    
    json_files = glob.glob(str(results_dir / 'bfi_to_minimarker_*.json'))
    
    if not json_files:
        print(f"No simulation files found in {results_dir}")
        return {}
    
    reports = {}
    
    print(f"Found {len(json_files)} simulation files:")
    for f in json_files:
        print(f"  - {Path(f).name}")
    
    for json_file in json_files:
        model_name = Path(json_file).stem.replace('bfi_to_minimarker_', '')
        
        # Load simulation results
        with open(json_file, 'r') as f:
            sim_results = json.load(f)
        
        # Validate
        report = validate_simulation_completeness(
            sim_results, 
            model_name, 
            expected_participant_count
        )
        
        reports[model_name] = report
        report.print_summary()
    
    return reports

def generate_recovery_suggestions(reports: Dict[str, ParticipantValidationReport]) -> List[str]:
    """
    Generate suggestions for recovering lost participants based on validation reports.
    
    Args:
        reports: Dictionary of validation reports
    
    Returns:
        List of recovery suggestions
    """
    suggestions = []
    
    # Analyze common issues across models
    total_parsing_errors = sum(r.parsing_errors for r in reports.values())
    total_missing_traits = sum(r.missing_traits for r in reports.values())
    total_cleaning_issues = sum(r.cleaning_issues for r in reports.values())
    
    if total_parsing_errors > 0:
        suggestions.append(
            f"Fix {total_parsing_errors} parsing errors: Check for non-integer responses, "
            "malformed JSON, or unexpected value formats"
        )
    
    if total_missing_traits > 0:
        suggestions.append(
            f"Handle {total_missing_traits} missing trait values: "
            "Consider imputation or re-prompting for missing responses"
        )
    
    if total_cleaning_issues > 0:
        suggestions.append(
            f"Resolve {total_cleaning_issues} key cleaning issues: "
            "Check for missing Mini-Marker traits in simulation responses"
        )
    
    # Model-specific suggestions
    for model_name, report in reports.items():
        loss_rate = (report.initial_count - report.final_count) / report.initial_count * 100
        if loss_rate > 10:  # More than 10% loss
            suggestions.append(
                f"{model_name}: High loss rate ({loss_rate:.1f}%) - "
                "Consider adjusting prompts or temperature settings"
            )
    
    if not suggestions:
        suggestions.append("All simulations show good data quality with minimal participant loss")
    
    return suggestions