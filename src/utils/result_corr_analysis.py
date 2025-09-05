import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

def analyze_csv_correlations(csv_file, output_folder='output', correlation_threshold=0.7):
    """
    Analyze CSV data and create correlation plots and matrix.
    
    Parameters:
    csv_file (str): Path to the CSV file
    output_folder (str): Output directory for plots
    correlation_threshold (float): Minimum correlation coefficient to plot relationships
    """
    
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Read the CSV file
    print(f"Reading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Select only numeric columns for correlation analysis
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Numeric columns found: {len(numeric_columns)}")
    
    if len(numeric_columns) < 2:
        print("Not enough numeric columns for correlation analysis!")
        return
    
    # Calculate correlation matrix
    print("Calculating correlation matrix...")
    correlation_matrix = df[numeric_columns].corr()
    
    # Create correlation matrix heatmap
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix Heatmap', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(output_path / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Correlation matrix saved to {output_path / 'correlation_matrix.png'}")
    
    # Find high correlation pairs
    print(f"\nFinding variable pairs with correlation >= {correlation_threshold}...")
    
    target_vars = ['alignment_length', 'gamma_shape', 'prop_invariant', 'mean_insertion_length', 'mean_deletion_length',
                   'normalized_colless_index', 'gamma', 'best_BD_speciation_rate', 'best_BD_extinction_rate',
                   'best_BLINDLIN', 'best_BCSTDLIN', 'best_BLINDEXP', 'best_BEXPDLIN', 'best_BEXPDCST', 'indel_rate']  # your target variables
    params_of_interest = ['likelihood_ess', 'n_samples', 'sp_score', 'spfn', 'spfp', 'tc', 'rf_distance', 'rfl_distance', 'wall_clock_s']  # your other parameters
    
    high_corr_pairs = []
    
    
    high_corr_pairs = []
    for target_var in target_vars:
        for param in params_of_interest:
            if param in correlation_matrix.columns and target_var in correlation_matrix.columns:
                corr_value = correlation_matrix.loc[target_var, param]
                if abs(corr_value) >= correlation_threshold:
                    high_corr_pairs.append((target_var, param, corr_value))
    '''
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) >= correlation_threshold:
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                high_corr_pairs.append((var1, var2, corr_value))
    
    '''
    
    
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print(f"Found {len(high_corr_pairs)} high correlation pairs:")
    for var1, var2, corr in high_corr_pairs:  
        print(f"  {var1} vs {var2}: {corr:.3f}")
    
     # Create scatter plots for high correlation pairs
    if high_corr_pairs:
        print(f"\nCreating scatter plots for high correlation pairs...")
        for idx, (var1, var2, corr) in enumerate(high_corr_pairs):
            data = df[[var1, var2]].dropna()
            if len(data) < 2 or data[var1].nunique() < 2 or data[var2].nunique() < 2:
                print(f"Skipping plot for {var1} vs {var2}: not enough valid or non-constant data.")
                continue
            plt.figure(figsize=(8, 6))
            plt.scatter(data[var1], data[var2], alpha=0.6, s=60, color='steelblue')
            try:
                z = np.polyfit(data[var1], data[var2], 1)
                p = np.poly1d(z)
                plt.plot(data[var1], p(data[var1]), "r--", alpha=0.8, linewidth=2)
            except Exception as e:
                print(f"Could not fit line for {var1} vs {var2}: {e}")
            plt.xlabel(var1, fontsize=12)
            plt.ylabel(var2, fontsize=12)
            plt.title(f'{var1} vs {var2} | Correlation: {corr:.3f}', fontsize=14)
            plt.grid(True, alpha=0.3)
            safe_var1 = var1.replace('/', '_').replace(' ', '_')
            safe_var2 = var2.replace('/', '_').replace(' ', '_')
            filename = f'correlation_{idx+1}_{safe_var1}_vs_{safe_var2}.png'
            plt.tight_layout()
            plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
            plt.close()
        print(f"All {len(high_corr_pairs)} scatter plots saved to '{output_folder}' directory.")
    
    
    # Save correlation data to CSV
    correlation_summary = pd.DataFrame(high_corr_pairs, 
                                     columns=['Variable_1', 'Variable_2', 'Correlation'])
    correlation_summary.to_csv(output_path / 'high_correlations_summary.csv', index=False)
    print(f"Correlation summary saved to {output_path / 'high_correlations_summary.csv'}")
    
    # Generate summary statistics
    print("\nDataset Summary:")
    print(f"Total variables: {len(df.columns)}")
    print(f"Numeric variables: {len(numeric_columns)}")
    print(f"Total observations: {len(df)}")
    print(f"High correlation pairs (>= {correlation_threshold}): {len(high_corr_pairs)}")
    
    if high_corr_pairs:
        print(f"Strongest positive correlation: {max(high_corr_pairs, key=lambda x: x[2])}")
        print(f"Strongest negative correlation: {min(high_corr_pairs, key=lambda x: x[2])}")
    
    # Save full correlation matrix to CSV
    correlation_matrix.to_csv(output_path / 'full_correlation_matrix.csv')
    print(f"Full correlation matrix saved to {output_path / 'full_correlation_matrix.csv'}")
    
    print(f"\nAnalysis complete! All outputs saved to '{output_folder}' directory.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python src/utils/result_corr_analysis.py <csv_file_path> <output_folder> <correlation_threshold>")
        print("Example: python src/utils/result_corr_analysis.py data/results.csv output 0.7")
        sys.exit(1)
    
    CSV_FILE = sys.argv[1]
    OUTPUT_FOLDER = sys.argv[2]
    CORRELATION_THRESHOLD = float(sys.argv[3])
    
    try:
        analyze_csv_correlations(CSV_FILE, OUTPUT_FOLDER, CORRELATION_THRESHOLD)
    except FileNotFoundError:
        print(f"Error: Could not find the file '{CSV_FILE}'")
        print("Please make sure the file exists and update the CSV_FILE variable with the correct path.")
    except Exception as e:
        print(f"An error occurred: {e}")