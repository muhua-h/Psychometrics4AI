#!/usr/bin/env python3
"""
BFI-2 Facet-Level Parameter Extraction and Simulation

This script replicates the original Study 3 facet-level data generation approach,
extracting parameters from Soto's empirical data and generating synthetic BFI-2 
responses that preserve the correlation structure.

Based on: study_3/likert_format/bfi2-facet-level-parameter-extraction-and-simulation.ipynb
"""

import pandas as pd
import numpy as np
import random
import os
from pathlib import Path

# Set random seeds for reproducibility
random.seed(1234)
np.random.seed(1234)

def load_original_data():
    """Load the original Study 3 data with proper path handling"""
    # Try multiple possible paths for the original data
    possible_paths = [
        '../../study_3/likert_format/data.csv',
        '../../../study_3/likert_format/data.csv',
        'data.csv'
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading data from: {path}")
            return pd.read_csv(path)
    
    raise FileNotFoundError(f"Could not find original Study 3 data. Tried paths: {possible_paths}")

def apply_reverse_coding(data):
    """Apply reverse coding to BFI-2 items based on the original scheme"""
    print("Applying reverse coding to BFI-2 items...")
    
    # Define the mapping for reverse coding based on the scheme provided
    reverse_coding_map = {
        'bfi1': 'reversed_bfi1', 'bfi2': 'reversed_bfi2', 'bfi3R': 'reversed_bfi3', 'bfi4R': 'reversed_bfi4',
        'bfi5R': 'reversed_bfi5', 'bfi6': 'reversed_bfi6', 'bfi7': 'reversed_bfi7', 'bfi8R': 'reversed_bfi8',
        'bfi9R': 'reversed_bfi9', 'bfi10': 'reversed_bfi10', 'bfi11R': 'reversed_bfi11', 'bfi12R': 'reversed_bfi12',
        'bfi13': 'reversed_bfi13', 'bfi14': 'reversed_bfi14', 'bfi15': 'reversed_bfi15', 'bfi16R': 'reversed_bfi16',
        'bfi17R': 'reversed_bfi17', 'bfi18': 'reversed_bfi18', 'bfi19': 'reversed_bfi19', 'bfi20': 'reversed_bfi20',
        'bfi21': 'reversed_bfi21', 'bfi22R': 'reversed_bfi22', 'bfi23R': 'reversed_bfi23', 'bfi24R': 'reversed_bfi24',
        'bfi25R': 'reversed_bfi25', 'bfi26R': 'reversed_bfi26', 'bfi27': 'reversed_bfi27', 'bfi28R': 'reversed_bfi28',
        'bfi29R': 'reversed_bfi29', 'bfi30R': 'reversed_bfi30', 'bfi31R': 'reversed_bfi31', 'bfi32': 'reversed_bfi32',
        'bfi33': 'reversed_bfi33', 'bfi34': 'reversed_bfi34', 'bfi35': 'reversed_bfi35', 'bfi36R': 'reversed_bfi36',
        'bfi37R': 'reversed_bfi37', 'bfi38': 'reversed_bfi38', 'bfi39': 'reversed_bfi39', 'bfi40': 'reversed_bfi40',
        'bfi41': 'reversed_bfi41', 'bfi42R': 'reversed_bfi42', 'bfi43': 'reversed_bfi43', 'bfi44R': 'reversed_bfi44',
        'bfi45R': 'reversed_bfi45', 'bfi46': 'reversed_bfi46', 'bfi47R': 'reversed_bfi47', 'bfi48R': 'reversed_bfi48',
        'bfi49R': 'reversed_bfi49', 'bfi50R': 'reversed_bfi50', 'bfi51R': 'reversed_bfi51', 'bfi52': 'reversed_bfi52',
        'bfi53': 'reversed_bfi53', 'bfi54': 'reversed_bfi54', 'bfi55R': 'reversed_bfi55', 'bfi56': 'reversed_bfi56',
        'bfi57': 'reversed_bfi57', 'bfi58R': 'reversed_bfi58', 'bfi59': 'reversed_bfi59', 'bfi60': 'reversed_bfi60'
    }

    # Perform reverse coding
    for original, reversed_var in reverse_coding_map.items():
        if original.endswith('R'):  # Reverse coded
            data[reversed_var] = 6 - data[original[:-1]]
        else:  # Not reverse coded
            data[reversed_var] = data[original]
    
    return data

def domain_stats(df, domain_prefix):
    """
    Compute the mean and standard deviation for columns that start with the specified domain prefix.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - domain_prefix (str): The prefix of the columns for which to compute the statistics.
    
    Returns:
    - pd.DataFrame: A DataFrame containing the mean and standard deviation for each column that starts with the specified prefix.
    """
    # Filter the columns that start with the domain prefix
    filtered_columns = [col for col in df.columns if col.startswith(domain_prefix)]
    domain_df = df[filtered_columns]
    
    # Compute mean and standard deviation
    means = domain_df.mean()
    std_devs = domain_df.std()
    
    # Create a result DataFrame
    result_df = pd.DataFrame([means, std_devs], index=["Mean", "Standard Deviation"])
    return result_df

def domain_correlation(df, domain_prefix):
    """
    Compute the correlation matrix for columns that start with the specified domain prefix.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - domain_prefix (str): The prefix of the columns for which to compute the correlation matrix.
    
    Returns:
    - pd.DataFrame: A 3x3 DataFrame containing the correlation matrix for the first three columns that start with the specified prefix.
    """
    # Filter columns that start with the domain prefix
    filtered_columns = [col for col in df.columns if col.startswith(domain_prefix)]
    # Select only the first three columns that match the prefix, if available
    filtered_columns = filtered_columns[:3]
    domain_df = df[filtered_columns]
    
    # Compute the correlation matrix
    correlation_matrix = domain_df.corr()
    
    # Return only the top left 3x3 section of the correlation matrix
    return correlation_matrix.iloc[:3, :3]

def calculate_intra_domain_correlations(data):
    """Calculate average intra-domain correlations for all domains"""
    print("Calculating intra-domain correlations...")
    
    # Domains with their facets defined
    domains = {
        "Extraversion": [['reversed_bfi1', 'reversed_bfi16', 'reversed_bfi31', 'reversed_bfi46'], 
                          ['reversed_bfi6', 'reversed_bfi21', 'reversed_bfi36', 'reversed_bfi51'], 
                          ['reversed_bfi11', 'reversed_bfi26', 'reversed_bfi41', 'reversed_bfi56']],
        "Agreeableness": [['reversed_bfi2', 'reversed_bfi17', 'reversed_bfi32', 'reversed_bfi47'], 
                           ['reversed_bfi7', 'reversed_bfi22', 'reversed_bfi37', 'reversed_bfi52'], 
                           ['reversed_bfi12', 'reversed_bfi27', 'reversed_bfi42', 'reversed_bfi57']],
        "Conscientiousness": [['reversed_bfi3', 'reversed_bfi18', 'reversed_bfi33', 'reversed_bfi48'], 
                               ['reversed_bfi8', 'reversed_bfi23', 'reversed_bfi38', 'reversed_bfi53'], 
                               ['reversed_bfi13', 'reversed_bfi28', 'reversed_bfi43', 'reversed_bfi58']],
        "Neuroticism": [['reversed_bfi4', 'reversed_bfi19', 'reversed_bfi34', 'reversed_bfi49'], 
                         ['reversed_bfi9', 'reversed_bfi24', 'reversed_bfi39', 'reversed_bfi54'], 
                         ['reversed_bfi14', 'reversed_bfi29', 'reversed_bfi44', 'reversed_bfi59']],
        "Openness": [['reversed_bfi10', 'reversed_bfi25', 'reversed_bfi40', 'reversed_bfi55'], 
                      ['reversed_bfi5', 'reversed_bfi20', 'reversed_bfi35', 'reversed_bfi50'], 
                      ['reversed_bfi15', 'reversed_bfi30', 'reversed_bfi45', 'reversed_bfi60']]
    }

    # Function to calculate average correlation excluding the diagonal
    def average_correlation(items):
        # Subset the data for the items
        subset = data[items]
        # Calculate the correlation matrix
        corr_matrix = subset.corr()
        # Flatten the matrix and exclude diagonal (self-correlation)
        correlations = corr_matrix.values[np.triu_indices_from(corr_matrix, k=1)]
        # Calculate the mean of the correlations
        return np.mean(correlations)

    # Dictionary to hold the average correlations for each domain
    domain_avg_correlations = {}

    # Calculate the average of the average correlations for each domain
    for domain, facets in domains.items():
        avg_corrs = [average_correlation(facet) for facet in facets]
        domain_avg_correlations[domain] = np.mean(avg_corrs)

    # Get the overall average
    average_domain_avg_correlations = np.mean(list(domain_avg_correlations.values()))
    
    print(f"Domain average correlations: {domain_avg_correlations}")
    print(f"Overall average intra-domain correlation: {average_domain_avg_correlations:.3f}")
    
    return average_domain_avg_correlations

def simulate_item_responses(means, std_devs, corr_matrix, intra_group_corr, n_simulations):
    """
    Simulate item responses based on group characteristics and correlations.

    Parameters:
    - means (np.array): Array of means for each group.
    - std_devs (np.array): Array of standard deviations for each group.
    - corr_matrix (np.array): Correlation matrix between the groups.
    - intra_group_corr (float): Correlation coefficient for items within the same group.
    - n_simulations (int): Number of simulations to generate.

    Returns:
    - np.array: Matrix of simulated responses (n_simulations x num_items).
    """
    num_groups = len(means)
    num_items_per_group = 4  # Assuming 4 items per group
    num_items = num_groups * num_items_per_group

    # Construct the covariance matrix from correlations and standard deviations
    cov_matrix = np.outer(std_devs, std_devs) * corr_matrix

    # Generate group-level scores
    group_scores = np.random.multivariate_normal(means, cov_matrix, size=n_simulations)

    # Placeholder for item scores
    item_scores = np.zeros((n_simulations, num_items))

    # Calculate item-level standard deviation within groups
    item_std_dev_within_group = np.sqrt((1 - intra_group_corr) * std_devs**2)

    # Generate item scores
    for group_index in range(num_groups):
        start_idx = group_index * num_items_per_group
        end_idx = start_idx + num_items_per_group
        
        for i in range(num_items_per_group):
            item_errors = np.random.normal(0, item_std_dev_within_group[group_index], n_simulations)
            item_scores[:, start_idx + i] = group_scores[:, group_index] + item_errors

    # Convert scores to integers and ensure they are within the range [1, 5]
    bounded_item_scores = np.clip(np.round(item_scores), 1, 5)

    return bounded_item_scores

def simulate_domain_data(data, domain_prefix, domain_name, intra_group_corr, n_simulations=200):
    """Simulate data for a specific domain"""
    print(f"Simulating data for {domain_name}...")
    
    # Extract parameters
    means = np.array(domain_stats(data, domain_prefix).loc['Mean'].values[:3])
    std_devs = np.array(domain_stats(data, domain_prefix).loc['Standard Deviation'].values[:3])
    corr_matrix = domain_correlation(data, domain_prefix)
    
    # Simulate item responses
    simulated_data = simulate_item_responses(means, std_devs, corr_matrix, intra_group_corr, n_simulations)
    
    return simulated_data

def create_final_dataset(simulated_data_dict):
    """Create the final dataset with proper column names and domain scores"""
    print("Creating final dataset...")
    
    # Define column names for each domain
    column_names = {
        'extraversion': ['simulated_bfi1', 'simulated_bfi16', 'simulated_bfi31', 'simulated_bfi46', 
                         'simulated_bfi6', 'simulated_bfi21', 'simulated_bfi36', 'simulated_bfi51', 
                         'simulated_bfi11', 'simulated_bfi26', 'simulated_bfi41', 'simulated_bfi56'],
        'agreeableness': ['simulated_bfi2', 'simulated_bfi17', 'simulated_bfi32', 'simulated_bfi47', 
                          'simulated_bfi7', 'simulated_bfi22', 'simulated_bfi37', 'simulated_bfi52', 
                          'simulated_bfi12', 'simulated_bfi27', 'simulated_bfi42', 'simulated_bfi57'],
        'conscientiousness': ['simulated_bfi3', 'simulated_bfi18', 'simulated_bfi33', 'simulated_bfi48', 
                              'simulated_bfi8', 'simulated_bfi23', 'simulated_bfi38', 'simulated_bfi53', 
                              'simulated_bfi13', 'simulated_bfi28', 'simulated_bfi43', 'simulated_bfi58'],
        'neuroticism': ['simulated_bfi4', 'simulated_bfi19', 'simulated_bfi34', 'simulated_bfi49', 
                        'simulated_bfi9', 'simulated_bfi24', 'simulated_bfi39', 'simulated_bfi54', 
                        'simulated_bfi14', 'simulated_bfi29', 'simulated_bfi44', 'simulated_bfi59'],
        'openness': ['simulated_bfi10', 'simulated_bfi25', 'simulated_bfi40', 'simulated_bfi55', 
                     'simulated_bfi5', 'simulated_bfi20', 'simulated_bfi35', 'simulated_bfi50', 
                     'simulated_bfi15', 'simulated_bfi30', 'simulated_bfi45', 'simulated_bfi60']
    }
    
    # Create DataFrames for each domain
    domain_dfs = {}
    for domain, data in simulated_data_dict.items():
        domain_dfs[domain] = pd.DataFrame(data, columns=column_names[domain])
    
    # Combine all domains
    simulated_data = pd.concat([domain_dfs['extraversion'], domain_dfs['agreeableness'], 
                               domain_dfs['conscientiousness'], domain_dfs['neuroticism'], 
                               domain_dfs['openness']], axis=1)
    
    # Calculate facet scores
    print("Calculating facet scores...")
    simulated_data['bfi_e_sociability'] = (simulated_data['simulated_bfi1'] + simulated_data['simulated_bfi16'] + 
                                          simulated_data['simulated_bfi31'] + simulated_data['simulated_bfi46'])/4
    simulated_data['bfi_e_assertiveness'] = (simulated_data['simulated_bfi6'] + simulated_data['simulated_bfi21'] + 
                                            simulated_data['simulated_bfi36'] + simulated_data['simulated_bfi51'])/4
    simulated_data['bfi_e_energy_level'] = (simulated_data['simulated_bfi11'] + simulated_data['simulated_bfi26'] + 
                                           simulated_data['simulated_bfi41'] + simulated_data['simulated_bfi56'])/4
    
    simulated_data['bfi_a_compassion'] = (simulated_data['simulated_bfi2'] + simulated_data['simulated_bfi17'] + 
                                         simulated_data['simulated_bfi32'] + simulated_data['simulated_bfi47'])/4
    simulated_data['bfi_a_respectfulness'] = (simulated_data['simulated_bfi7'] + simulated_data['simulated_bfi22'] + 
                                             simulated_data['simulated_bfi37'] + simulated_data['simulated_bfi52'])/4
    simulated_data['bfi_a_trust'] = (simulated_data['simulated_bfi12'] + simulated_data['simulated_bfi27'] + 
                                    simulated_data['simulated_bfi42'] + simulated_data['simulated_bfi57'])/4
    
    simulated_data['bfi_c_organization'] = (simulated_data['simulated_bfi3'] + simulated_data['simulated_bfi18'] + 
                                           simulated_data['simulated_bfi33'] + simulated_data['simulated_bfi48'])/4
    simulated_data['bfi_c_productiveness'] = (simulated_data['simulated_bfi8'] + simulated_data['simulated_bfi23'] + 
                                             simulated_data['simulated_bfi38'] + simulated_data['simulated_bfi53'])/4
    simulated_data['bfi_c_responsibility'] = (simulated_data['simulated_bfi13'] + simulated_data['simulated_bfi28'] + 
                                             simulated_data['simulated_bfi43'] + simulated_data['simulated_bfi58'])/4
    
    simulated_data['bfi_n_anxiety'] = (simulated_data['simulated_bfi4'] + simulated_data['simulated_bfi19'] + 
                                      simulated_data['simulated_bfi34'] + simulated_data['simulated_bfi49'])/4
    simulated_data['bfi_n_depression'] = (simulated_data['simulated_bfi9'] + simulated_data['simulated_bfi24'] + 
                                         simulated_data['simulated_bfi39'] + simulated_data['simulated_bfi54'])/4
    simulated_data['bfi_n_emotional_volatility'] = (simulated_data['simulated_bfi14'] + simulated_data['simulated_bfi29'] + 
                                                   simulated_data['simulated_bfi44'] + simulated_data['simulated_bfi59'])/4
    
    simulated_data['bfi_o_intellectual_curiosity'] = (simulated_data['simulated_bfi10'] + simulated_data['simulated_bfi25'] + 
                                                     simulated_data['simulated_bfi40'] + simulated_data['simulated_bfi55'])/4
    simulated_data['bfi_o_aesthetic_sensitivity'] = (simulated_data['simulated_bfi5'] + simulated_data['simulated_bfi20'] + 
                                                    simulated_data['simulated_bfi35'] + simulated_data['simulated_bfi50'])/4
    simulated_data['bfi_o_creative_imagination'] = (simulated_data['simulated_bfi15'] + simulated_data['simulated_bfi30'] + 
                                                   simulated_data['simulated_bfi45'] + simulated_data['simulated_bfi60'])/4
    
    # Calculate domain scores
    print("Calculating domain scores...")
    simulated_data['bfi_e'] = (simulated_data['bfi_e_sociability'] + simulated_data['bfi_e_assertiveness'] + 
                              simulated_data['bfi_e_energy_level'])/3
    simulated_data['bfi_a'] = (simulated_data['bfi_a_compassion'] + simulated_data['bfi_a_respectfulness'] + 
                              simulated_data['bfi_a_trust'])/3
    simulated_data['bfi_c'] = (simulated_data['bfi_c_organization'] + simulated_data['bfi_c_productiveness'] + 
                              simulated_data['bfi_c_responsibility'])/3
    simulated_data['bfi_n'] = (simulated_data['bfi_n_anxiety'] + simulated_data['bfi_n_depression'] + 
                              simulated_data['bfi_n_emotional_volatility'])/3
    simulated_data['bfi_o'] = (simulated_data['bfi_o_intellectual_curiosity'] + simulated_data['bfi_o_aesthetic_sensitivity'] + 
                              simulated_data['bfi_o_creative_imagination'])/3
    
    return simulated_data

def apply_final_reverse_coding(simulated_data):
    """Apply final reverse coding to match original BFI-2 structure"""
    print("Applying final reverse coding...")
    
    reverse_simulated_data = simulated_data.copy()
    
    reverse_coding_map = {
        'simulated_bfi1': 'bfi1', 'simulated_bfi2': 'bfi2', 'simulated_bfi3': 'bfi3R', 'simulated_bfi4': 'bfi4R',
        'simulated_bfi5': 'bfi5R', 'simulated_bfi6': 'bfi6', 'simulated_bfi7': 'bfi7', 'simulated_bfi8': 'bfi8R',
        'simulated_bfi9': 'bfi9R', 'simulated_bfi10': 'bfi10', 'simulated_bfi11': 'bfi11R', 'simulated_bfi12': 'bfi12R',
        'simulated_bfi13': 'bfi13', 'simulated_bfi14': 'bfi14', 'simulated_bfi15': 'bfi15', 'simulated_bfi16': 'bfi16R',
        'simulated_bfi17': 'bfi17R', 'simulated_bfi18': 'bfi18', 'simulated_bfi19': 'bfi19', 'simulated_bfi20': 'bfi20',
        'simulated_bfi21': 'bfi21', 'simulated_bfi22': 'bfi22R', 'simulated_bfi23': 'bfi23R', 'simulated_bfi24': 'bfi24R',
        'simulated_bfi25': 'bfi25R', 'simulated_bfi26': 'bfi26R', 'simulated_bfi27': 'bfi27', 'simulated_bfi28': 'bfi28R',
        'simulated_bfi29': 'bfi29R', 'simulated_bfi30': 'bfi30R', 'simulated_bfi31': 'bfi31R', 'simulated_bfi32': 'bfi32',
        'simulated_bfi33': 'bfi33', 'simulated_bfi34': 'bfi34', 'simulated_bfi35': 'bfi35', 'simulated_bfi36': 'bfi36R',
        'simulated_bfi37': 'bfi37R', 'simulated_bfi38': 'bfi38', 'simulated_bfi39': 'bfi39', 'simulated_bfi40': 'bfi40',
        'simulated_bfi41': 'bfi41', 'simulated_bfi42': 'bfi42R', 'simulated_bfi43': 'bfi43', 'simulated_bfi44': 'bfi44R',
        'simulated_bfi45': 'bfi45R', 'simulated_bfi46': 'bfi46', 'simulated_bfi47': 'bfi47R', 'simulated_bfi48': 'bfi48R',
        'simulated_bfi49': 'bfi49R', 'simulated_bfi50': 'bfi50R', 'simulated_bfi51': 'bfi51R', 'simulated_bfi52': 'bfi52',
        'simulated_bfi53': 'bfi53', 'simulated_bfi54': 'bfi54', 'simulated_bfi55': 'bfi55R', 'simulated_bfi56': 'bfi56',
        'simulated_bfi57': 'bfi57', 'simulated_bfi58': 'bfi58R', 'simulated_bfi59': 'bfi59', 'simulated_bfi60': 'bfi60'
    }

    # Perform reverse coding
    for key, value in reverse_coding_map.items():
        if value.endswith('R'):  # Reverse coded
            reverse_simulated_data[key] = 6 - reverse_simulated_data[key]
        else:  # Not reverse coded
            reverse_simulated_data[key] = reverse_simulated_data[key]
    
    # Remove the 'simulated_' prefix from the variable names
    reverse_simulated_data.columns = reverse_simulated_data.columns.str.replace('simulated_', '')
    
    return reverse_simulated_data

def main():
    """Main function to run the complete data generation process"""
    print("Starting BFI-2 Facet-Level Parameter Extraction and Simulation")
    print("=" * 60)
    
    # Step 1: Load original data
    data = load_original_data()
    print(f"Loaded original data: {data.shape}")
    
    # Step 2: Apply reverse coding
    data = apply_reverse_coding(data)
    
    # Step 3: Calculate intra-domain correlations
    intra_group_corr = calculate_intra_domain_correlations(data)
    
    # Step 4: Simulate data for each domain
    n_simulations = 200
    print(f"Simulating {n_simulations} participants...")
    
    simulated_data_dict = {}
    domains = [
        ('bfi2_e', 'extraversion'),
        ('bfi2_a', 'agreeableness'), 
        ('bfi2_c', 'conscientiousness'),
        ('bfi2_n', 'neuroticism'),
        ('bfi2_o', 'openness')
    ]
    
    for domain_prefix, domain_name in domains:
        simulated_data_dict[domain_name] = simulate_domain_data(
            data, domain_prefix, domain_name, intra_group_corr, n_simulations
        )
    
    # Step 5: Create final dataset
    simulated_data = create_final_dataset(simulated_data_dict)
    
    # Step 6: Apply final reverse coding
    final_data = apply_final_reverse_coding(simulated_data)
    
    # Step 7: Save results
    output_file = 'facet_lvl_simulated_data.csv'
    final_data.to_csv(output_file, index=False)
    
    print(f"\nData generation complete!")
    print(f"Final dataset shape: {final_data.shape}")
    print(f"Saved to: {output_file}")
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(f"- Number of participants: {len(final_data)}")
    print(f"- Number of BFI-2 items: 60")
    print(f"- Number of facet scores: 15")
    print(f"- Number of domain scores: 5")
    print(f"- Total columns: {len(final_data.columns)}")
    
    # Show sample correlations
    domain_cols = ['bfi_e', 'bfi_a', 'bfi_c', 'bfi_n', 'bfi_o']
    if all(col in final_data.columns for col in domain_cols):
        print("\nDomain Score Correlations:")
        corr_matrix = final_data[domain_cols].corr()
        print(corr_matrix.round(3))
    
    print("\nFirst few rows of generated data:")
    print(final_data.head())

if __name__ == "__main__":
    main() 