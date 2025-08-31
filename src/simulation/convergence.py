#!/usr/bin/env python3
"""
MCMC Phylogenetic Trace Analysis Script

This script analyzes MCMC trace logs from phylogenetic analyses, computing:
- RF distances between successive topologies
- Trace plots, autocorrelation plots, and density plots for all parameters
- ESS, Mean, and other convergence statistics
- Summary statistics exported to CSV

Author: Generated for phylogenetic MCMC analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import os
import warnings
from typing import Dict, List, Tuple, Any
import re
import dendropy
import sys
import ete3

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class PhylogeneticTraceAnalyzer:
    """
    A comprehensive analyzer for MCMC phylogenetic trace logs
    """
    
    def __init__(self, log_file: str, output_folder: str):
        """
        Initialize the analyzer with a log file
        
        Args:
            log_file: Path to the trace log file
        """
        self.log_file = log_file
        
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
        self.df = None
        self.numeric_columns = []
        self.topology_column = 'Tree Topology'
        self.rf_column = 'RF_distance_lag1'
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and parse the trace log file
        
        Returns:
            DataFrame with parsed trace data
        """
        print("Loading trace log data...")
        
        # Read the tab-separated file
        self.df = pd.read_csv(self.log_file, sep='\t')
        
        # Identify numeric columns (exclude topology)
        self.numeric_columns = [col for col in self.df.columns 
                               if col != self.topology_column and 
                               pd.api.types.is_numeric_dtype(self.df[col])]
        
        print(f"Loaded {len(self.df)} samples with {len(self.numeric_columns)} numeric parameters")
        print(f"Numeric columns: {self.numeric_columns}")
        
        return self.df
    
    def compute_rf_distances(self) -> pd.DataFrame:
        print("Computing RF distances between successive topologies using dendropy...")
        if self.topology_column not in self.df.columns:
            raise ValueError(f"Topology column '{self.topology_column}' not found in data")
        topologies = self.df[self.topology_column].tolist()
        rf_distances = [np.nan]
        for i in range(1, len(topologies)):
            try:
                '''
                tree1 = dendropy.Tree.get_from_string(topologies[i-1] + ";", schema="newick")
                tree2 = dendropy.Tree.get_from_string(topologies[i] + ";", schema="newick")
                rf_dist = tree1.robinson_foulds_distance(tree2, ignore_missing_taxa=True)
                '''
                tree1 = ete3.Tree(topologies[i-1] + ";")
                tree2 = ete3.Tree(topologies[i] + ";")
                rf_dist = tree1.robinson_foulds(tree2)[0]
            except Exception as e:
                print(f"Error computing RF distance for trees at index {i-1} and {i}: {e}")
                rf_dist = np.nan
            rf_distances.append(rf_dist)
        self.df[self.rf_column] = rf_distances
        if self.rf_column not in self.numeric_columns:
            self.numeric_columns.append(self.rf_column)
        print(f"Added RF distance column: {self.rf_column}")
        print(rf_distances[:20])
        return self.df
    
    def compute_effective_sample_size(self, series: pd.Series, max_lag: int = None) -> float:
        """
        Compute Effective Sample Size (ESS) using autocorrelation
        
        Args:
            series: Time series data
            max_lag: Maximum lag for autocorrelation calculation
            
        Returns:
            Effective Sample Size
        """
        if max_lag is None:
            max_lag = min(len(series) // 4, 200)
        
        # Remove NaN values
        clean_series = series.dropna()
        if len(clean_series) < 10:
            return np.nan
        
        # Compute autocorrelation
        autocorrs = []
        mean_val = clean_series.mean()
        var_val = clean_series.var()
        
        if var_val == 0:
            return len(clean_series)
        
        for lag in range(1, min(max_lag + 1, len(clean_series) - 1)):
            if lag >= len(clean_series):
                break
                
            # Compute lag-k autocorrelation
            lagged_series = clean_series.iloc[:-lag]
            original_series = clean_series.iloc[lag:]
            
            if len(lagged_series) == 0 or len(original_series) == 0:
                break
                
            autocorr = np.corrcoef(lagged_series, original_series)[0, 1]
            if np.isnan(autocorr):
                break
                
            autocorrs.append(autocorr)
            
            # Stop when autocorrelation becomes negligible
            if autocorr < 0.1:
                break
        
        if not autocorrs:
            return len(clean_series)
        
        # Integrated autocorrelation time
        tau_int = 1 + 2 * sum(autocorrs)
        
        # ESS = N / (2 * tau_int + 1)
        ess = len(clean_series) / (2 * tau_int + 1)
        
        return max(1.0, ess)
    
    def create_trace_plot(self, column: str, save_path: str):
        """Create trace plot for a parameter"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        
        data = self.df[column].dropna()
        
        #used to be data.index
        ax.plot(data.index, data.values, alpha=0.7, linewidth=0.8)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(column)
        ax.set_title(f'Trace Plot: {column}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_autocorr_plot(self, column: str, save_path: str, max_lag: int = 50):
        """Create autocorrelation plot for a parameter"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        
        data = self.df[column].dropna()
        if len(data) < 10:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, 
                   ha='center', va='center')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return
        
        # Compute autocorrelations
        lags = range(1, min(max_lag + 1, len(data) - 1))
        autocorrs = []
        
        for lag in lags:
            if lag >= len(data):
                break
            lagged_data = data.iloc[:-lag]
            original_data = data.iloc[lag:]
            
            if len(lagged_data) > 0 and len(original_data) > 0:
                autocorr = np.corrcoef(lagged_data, original_data)[0, 1]
                if not np.isnan(autocorr):
                    autocorrs.append(autocorr)
                else:
                    break
            else:
                break
        
        if autocorrs:
            lags_plot = list(range(1, len(autocorrs) + 1))
            ax.plot(lags_plot, autocorrs, 'o-', markersize=3, alpha=0.7)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, 
                      label='0.1 threshold')
        
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title(f'Autocorrelation Plot: {column}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_density_plot(self, column: str, save_path: str):
        """Create density plot for a parameter"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        
        data = self.df[column].dropna()
        data = data[np.isfinite(data)]
        
        if len(data) < 3:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes, 
                   ha='center', va='center')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return
        
        # Create histogram
        
        ax.hist(data, bins=min(30, len(data)//3), density=True, alpha=0.7, 
               color='skyblue', edgecolor='black', linewidth=0.5)
        
        '''
        # Add KDE if enough data points
        if len(data) > 5:
            try:
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 200)
                ax.plot(x_range, kde(x_range), 'red', linewidth=2, label='KDE')
                ax.legend()
            except Exception:
                pass  # Skip KDE if it fails
        '''
        
        ax.set_xlabel(column)
        ax.set_ylabel('Density')
        ax.set_title(f'Density Plot: {column}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_convergence_plots(self) -> str:
        """
        Create all convergence diagnostic plots
        
        Returns:
            Path to the plots directory
        """
        print("Creating convergence diagnostic plots...")
        
        # Create plots directory
        plots_dir = os.path.join(self.output_folder, 'convergence_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create plots for each numeric parameter
        for column in self.numeric_columns:
            print(f"  Creating plots for {column}...")
            
            # Safe filename
            safe_name = re.sub(r'[^\w\-_.]', '_', column)
            
            # Trace plot
            trace_path = os.path.join(plots_dir, f'{safe_name}_trace.png')
            self.create_trace_plot(column, trace_path)
            
            # Autocorrelation plot
            autocorr_path = os.path.join(plots_dir, f'{safe_name}_autocorr.png')
            self.create_autocorr_plot(column, autocorr_path)
            
            # Density plot
            density_path = os.path.join(plots_dir, f'{safe_name}_density.png')
            self.create_density_plot(column, density_path)
        
        print(f"All plots saved to '{plots_dir}' directory")
        return plots_dir
    
    def compute_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute comprehensive statistics for all parameters
        
        Returns:
            Dictionary of statistics for each parameter
        """
        print("Computing convergence statistics...")
        
        statistics = {}
        
        # Statistics for numeric parameters
        for column in self.numeric_columns:
            data = self.df[column].dropna()
            
            if len(data) == 0:
                stats_dict = {
                    'ESS': np.nan,
                    'Mean': np.nan,
                    'Std': np.nan,
                    'Min': np.nan,
                    'Max': np.nan,
                    'Median': np.nan,
                    'N_samples': 0,
                    'N_unique': 0
                }
            else:
                ess = self.compute_effective_sample_size(data)
                
                stats_dict = {
                    'ESS': ess,
                    'Mean': data.mean(),
                    'Std': data.std(),
                    'Min': data.min(),
                    'Max': data.max(),
                    'Median': data.median(),
                    'N_samples': len(data),
                    'N_unique': data.nunique()
                }
            
            statistics[column] = stats_dict
        
        # Special statistics for topology
        if self.topology_column in self.df.columns:
            topology_data = self.df[self.topology_column].dropna()
            unique_topologies = topology_data.nunique()
            total_topologies = len(topology_data)
            percent_unique = (unique_topologies / total_topologies) * 100 if total_topologies > 0 else 0
            
            statistics[self.topology_column] = {
                'ESS': np.nan,  # Not applicable for categorical data
                'Mean': np.nan,
                'Std': np.nan,
                'Min': np.nan,
                'Max': np.nan,
                'Median': np.nan,
                'N_samples': total_topologies,
                'N_unique': unique_topologies,
                'Percent_unique_topologies': percent_unique
            }
        
        return statistics
    
    def export_statistics_to_csv(self, statistics: Dict[str, Dict[str, Any]], 
                                filename: str = 'mcmc_statistics.csv'):
        """
        Export statistics to CSV file
        
        Args:
            statistics: Statistics dictionary
            filename: Output CSV filename
        """
        filename = os.path.join(self.output_folder, filename)
        print(f"Exporting statistics to {filename}...")
        
        # Convert to DataFrame
        stats_df = pd.DataFrame.from_dict(statistics, orient='index')
        
        # Add parameter name as a column
        stats_df.reset_index(inplace=True)
        stats_df.rename(columns={'index': 'Parameter'}, inplace=True)
        
        # Round numeric columns
        numeric_cols = stats_df.select_dtypes(include=[np.number]).columns
        stats_df[numeric_cols] = stats_df[numeric_cols].round(6)
        
        # Save to CSV
        stats_df.to_csv(filename, index=False)
        
        print(f"Statistics exported to {filename}")
        return stats_df
    
    def save_updated_log(self, filename: str = None):
        """
        Save the updated log file with RF distances
        
        Args:
            filename: Output filename (defaults to original filename with _updated suffix)
        """
        if filename is None:
            filename = os.path.join(self.output_folder, 'trace_log_updated.log')
        
        print(f"Saving updated log file to {filename}...")
        self.df.to_csv(filename, sep='\t', index=False)
        print(f"Updated log file saved to {filename}")
        
        return filename
    
    def run_complete_analysis(self) -> Tuple[str, str, str]:
        """
        Run the complete analysis pipeline
        
        Returns:
            Tuple of (plots_directory, statistics_file, updated_log_file)
        """
        print("="*60)
        print("MCMC Phylogenetic Trace Analysis")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Compute RF distances
        self.compute_rf_distances()
        
        # Create plots
        plots_dir = self.create_convergence_plots()
        
        # Compute statistics
        statistics = self.compute_statistics()
        
        # Export statistics
        stats_file = self.export_statistics_to_csv(statistics)
        
        # Save updated log
        updated_log = self.save_updated_log()
        
        # Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"📊 Plots directory: {plots_dir}")
        print(f"📈 Statistics file: {stats_file}")
        print(f"📋 Updated log file: {updated_log}")
        print(f"🔢 Analyzed {len(self.numeric_columns)} numeric parameters")
        print(f"🌳 Total samples: {len(self.df)}")
        
        if self.topology_column in statistics:
            topo_stats = statistics[self.topology_column]
            print(f"🌲 Unique topologies: {topo_stats['N_unique']} / {topo_stats['N_samples']} "
                  f"({topo_stats.get('Percent_unique_topologies', 0):.1f}%)")
        
        print("="*60)
        
        return plots_dir, stats_file, updated_log

def main():
    """
    Main function to run the analysis
    """
    if len(sys.argv) < 3:
        print("Usage: python convergence.py <trace_log_file> <output_folder>")
        print("Example: python src/simulation/convergence.py data/simulation/SCOPt4e1/seq_1/historian/combined_trace.log data/simulation/SCOPt4e1/seq_1/historian/mcmcStats")
        print("Example: python src/simulation/convergence.py data/simulation/SCOPt4e1/seq_1/baliphy-1/combined_trace.log data/simulation/SCOPt4e1/seq_1/baliphy-1/mcmcStats")
        sys.exit(1)
    
    LOG_FILE = sys.argv[1]
    output_folder = sys.argv[2]
    
    # Initialize analyzer
    analyzer = PhylogeneticTraceAnalyzer(LOG_FILE, output_folder)
    
    try:
        # Run complete analysis
        plots_dir, stats_file, updated_log = analyzer.run_complete_analysis()
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"   Check the '{plots_dir}' folder for diagnostic plots")
        print(f"   Open '{stats_file}' for summary statistics")
        print(f"   Use '{updated_log}' for the updated trace log with RF distances")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        print(f"   Please check your log file format and try again.")
        raise

if __name__ == "__main__":
    main()