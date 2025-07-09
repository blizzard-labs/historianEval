import pandas as pd
import numpy as np
import re
import math

def clean_protein_evolution_data(input_file, output_file=None):
    """
    Clean protein evolution CSV data and extract specified columns.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file (optional)
    
    Returns:
    pandas.DataFrame: Cleaned dataframe with specified columns
    """
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Initialize the output dataframe
    cleaned_data = {}
    
    # Basic information columns
    cleaned_data['Filename'] = df['filename'].str.replace('.fasta', '', regex=False)
    cleaned_data['n_sequences_tips'] = df['n_sequences']  # Same as n_tips based on your data
    cleaned_data['alignment_length'] = df['alignment_length']
    
    # Gamma and invariant site parameters
    cleaned_data['gamma_shape'] = df['gamma_shape']
    cleaned_data['prop_invariant'] = df['prop_invariant']
    
    # Insertion/deletion rates and lengths
    cleaned_data['insertion_rate'] = df['insertion_rate']
    cleaned_data['deletion_rate'] = df['deletion_rate']
    cleaned_data['mean_insertion_length'] = df['mean_insertion_length']
    cleaned_data['mean_deletion_length'] = df['mean_deletion_length']
    
    # Best Birth-Death model parameters
    cleaned_data['best_BD_speciation_rate'] = df['bd_speciation_rate']
    cleaned_data['best_BD_extinction_rate'] = df['bd_extinction_rate']
    
    # Best Coalescent model parameters
    cleaned_data['best_coal_growth_rate'] = df['coal_growth_rate']
    cleaned_data['best_coal_eff_pop_size'] = df['coal_effective_pop_size']
    
    # AIC scores for best models
    
    bd_weights = []
    for idx, row in df.iterrows():
        bd_weight, coal_weight = compute_aic_weights(row['bd_aic'], row['coal_aic'])
        bd_weights.append(bd_weight)
    
    cleaned_data['bd_weight'] = bd_weights
    
    # Define all possible model types
    bd_models = ['BCSTDCST', 'BEXPDCST', 'BLINDCST', 'BCSTDEXP', 'BEXPDEXP', 'BLINDEXP', 
                 'BCSTDLIN', 'BEXPDLIN', 'BLINDLIN']
    coal_models = ['COALCST', 'COALEXP', 'COALLIN', 'COALSTEP', 'COALLOG']
    
    # Create binary columns for each BD model
    for model in bd_models:
        model_name = f'BD_{model}'
        cleaned_data[f'best_{model}'] = np.where(df['best_bd_model'] == model_name, 1, 0)
    
    # Create binary columns for each Coalescent model
    for model in coal_models:
        model_name = f'COAL_{model}'
        cleaned_data[f'best_{model}'] = np.where(df['best_coal_model'] == model_name, 1, 0)
    
    # Tree statistics
    cleaned_data['tree_length'] = df['tree_length']
    cleaned_data['crown_age'] = df['crown_age']
    
    # Create the cleaned dataframe
    cleaned_df = pd.DataFrame(cleaned_data)
    
    # Handle missing values (replace with NaN for numeric columns where appropriate)
    numeric_columns = [
        'gamma_shape', 'prop_invariant', 'insertion_rate', 'deletion_rate',
        'mean_insertion_length', 'mean_deletion_length', 'best_BD_speciation_rate',
        'best_BD_extinction_rate', 'best_coal_growth_rate', 'best_coal_eff_pop_size',
        'bd_weight', 'tree_length', 'crown_age'
    ]
    
    # Add model indicator columns (these should remain as integers)
    model_columns = [col for col in cleaned_df.columns if col.startswith('best_') and 
                    any(model in col for model in bd_models + coal_models)]
    
    for col in numeric_columns:
        if col in cleaned_df.columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
    
    # Round numeric values to reasonable precision
    for col in numeric_columns:
        if col in cleaned_df.columns:
            if col in ['insertion_rate', 'deletion_rate']:
                cleaned_df[col] = cleaned_df[col].round(8)  # Keep high precision for rates
            elif col in ['best_BD_speciation_rate', 'best_BD_extinction_rate']:
                cleaned_df[col] = cleaned_df[col].round(8)
            elif col in ['gamma_shape', 'prop_invariant']:
                cleaned_df[col] = cleaned_df[col].round(4)
            else:
                cleaned_df[col] = cleaned_df[col].round(3)
    
    # Save to file if output path is provided
    if output_file:
        cleaned_df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")
    
    return cleaned_df

def display_summary_stats(df):
    """Display summary statistics for the cleaned data"""
    print("\nSummary Statistics:")
    print("=" * 50)
    print(f"Total number of sequences: {len(df)}")
    
    # Count best models
    bd_models = ['BCSTDCST', 'BEXPDCST', 'BLINDCST', 'BCSTDEXP', 'BEXPDEXP', 'BLINDEXP', 
                 'BCSTDLIN', 'BEXPDLIN', 'BLINDLIN']
    coal_models = ['COALCST', 'COALEXP', 'COALLIN', 'COALSTEP', 'COALLOG']
    
    print("\nBest BD Models:")
    for model in bd_models:
        col_name = f'best_{model}'
        if col_name in df.columns:
            count = df[col_name].sum()
            if count > 0:
                print(f"  {model}: {count}")
    
    print("\nBest Coalescent Models:")
    for model in coal_models:
        col_name = f'best_{model}'
        if col_name in df.columns:
            count = df[col_name].sum()
            if count > 0:
                print(f"  {model}: {count}")
    
    print(f"\nAverage alignment length: {df['alignment_length'].mean():.1f}")
    print(f"Average number of sequences: {df['n_sequences_tips'].mean():.1f}")
    print(f"Average tree length: {df['tree_length'].mean():.2f}")
    print(f"Average crown age: {df['crown_age'].mean():.2f}")

def compute_aic_weights(AIC_bd, AIC_coal):
    delta_bd = AIC_bd - min(AIC_bd, AIC_coal)
    delta_coal = AIC_coal - min(AIC_bd, AIC_coal)

    w_bd = np.exp(-0.5 * delta_bd)
    w_coal = np.exp(-0.5 * delta_coal)
    total = w_bd + w_coal

    return w_bd / total, w_coal / total

# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    input_file = "data/model_gen/V1_sample_aa/protein_evolution_parameters_with_rates.csv"
    output_file = "data/model_gen/V1_sample_aa/cleaned_protein_evolution_data.csv"
    
    try:
        # Clean the data
        cleaned_df = clean_protein_evolution_data(input_file, output_file)
        
        # Display the first few rows
        print("Cleaned Data Preview:")
        print("=" * 50)
        print(cleaned_df.head())
        
        # Display summary statistics
        display_summary_stats(cleaned_df)
        
        # Show column names and data types
        print("\nColumn Information:")
        print("=" * 50)
        print(cleaned_df.dtypes)
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")