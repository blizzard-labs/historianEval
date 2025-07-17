#!/usr/bin/env python3
"""
Phylogenetic Parameter Distribution Fitting Script

This script fits joint distributions to phylogenetic parameters extracted from TreeFam alignments
for use in realistic sequence simulation with indel-seq-gen.
"""


import os
import sys
import pickle
import pandas as pd
import numpy as np
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, anderson
import warnings
warnings.filterwarnings('ignore')

# Try to import copulas - install with: pip install copulas
try:
    from copulas.multivariate import GaussianMultivariate
    from copulas.univariate import Univariate
    COPULAS_AVAILABLE = True
except ImportError:
    print("Warning: copulas library not available. Install with: pip install copulas")
    COPULAS_AVAILABLE = False

# For multivariate normal as fallback
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class PhylogeneticParameterFitter:
    """
    Class for fitting distributions to phylogenetic parameters
    """
    
    def __init__(self, csv_file):
        """Initialize with CSV file containing TreeFam parameters"""
        self.data = pd.read_csv(csv_file)
        self.fitted_distributions = {}
        self.joint_model = None
        self.parameter_groups = self._define_parameter_groups()
        
    def _define_parameter_groups(self):
        """Define logical groups of parameters for joint modeling"""
        
        #n_sequences, alignment_length, gamma_shape, prop_invariant, insertion_rate, deletion_rate, insertion_events, deletion_events
        #mean_insertion_length, mean_deletion_length, total_gaps, indel_to_substitution_ration, rf_length_distance
        
        # n_sequences, gamma_shape, prop_invariant, insertion_rate, deletion_rate, mean_insertion_length, mean_deletion_length, rf_length_distance
        '''
        return {
            'amino_acid_frequencies': [col for col in self.data.columns if col.startswith('freq_')],
            'substitution_model': ['gamma_shape', 'prop_invariant'],
            'indel_parameters': ['indel_rate', 'mean_indel_length'],
            'tree_parameters': ['tree_length', 'crown_age', 'n_tips'],
            'diversification': ['speciation_rate', 'extinction_rate', 'net_diversification'],
            'sequence_properties': ['n_sequences', 'alignment_length', 'total_gaps'],
            'key_parameters' : ['n_sequences', 'gamma_shape', 'prop_invariant',
                                'insertion_rate', 'deletion_rate',
                                'mean_insertion_length', 'mean_deletion_length',
                                'rf_length_distance']
        }
        '''
        
        return {
            'key_parameters' : ['n_sequences', 'alignment_length', 'gamma_shape', 'prop_invariant',
                                'insertion_rate', 'deletion_rate',
                                'mean_insertion_length', 'mean_deletion_length',
                                'rf_length_distance']
        }

    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        # Remove non-numeric columns and handle missing values
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.numeric_data = self.data[numeric_cols].copy()
        
        # Handle missing values
        self.numeric_data = self.numeric_data.dropna()
        
        # Log-transform rate parameters (they're often log-normal)
        rate_params = ['gamma_shape', 'insertion_rate', 'deletion_rate', 'mean_insertion_length', 'mean_deletion_length']
        
        for param in rate_params:
            if param in self.numeric_data.columns:
                # Add small constant to avoid log(0)
                self.numeric_data[param + '_log'] = np.log(self.numeric_data[param] + 1e-10)
        
        # Logit-transform proportions
        prop_params = ['prop_invariant'] + [col for col in self.numeric_data.columns if col.startswith('freq_')]
        
        for param in prop_params:
            if param in self.numeric_data.columns:
                # Logit transform: log(p/(1-p)), handling edge cases
                p = self.numeric_data[param].clip(1e-10, 1-1e-10)
                self.numeric_data[param + '_logit'] = np.log(p / (1 - p))
        
        print(f"Preprocessed data shape: {self.numeric_data.shape}")
        return self.numeric_data
    
    def fit_marginal_distributions(self, param_subset=None):
        """Fit marginal distributions to individual parameters"""
        if param_subset is None:
            param_subset = self.numeric_data.columns
        
        # Common distributions to test
        distributions = [
            stats.norm, stats.lognorm, stats.gamma, stats.expon, 
            stats.beta, stats.weibull_min, stats.chi2
        ]
        
        self.fitted_distributions = {}
        
        for param in param_subset:
            if param not in self.numeric_data.columns:
                continue
                
            data = self.numeric_data[param].dropna()
            if len(data) < 10:  # Skip if too few data points
                continue
                
            best_dist = None
            best_params = None
            best_aic = np.inf
            
            print(f"Fitting distributions for {param}...")
            
            for dist in distributions:
                try:
                    # Fit distribution
                    if dist == stats.beta:
                        # Beta distribution needs data in [0,1]
                        if data.min() < 0 or data.max() > 1:
                            continue
                    
                    params = dist.fit(data)
                    
                    # Calculate AIC
                    loglik = np.sum(dist.logpdf(data, *params))
                    aic = 2 * len(params) - 2 * loglik
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_dist = dist
                        best_params = params
                        
                except Exception as e:
                    continue
            
            if best_dist is not None:
                self.fitted_distributions[param] = {
                    'distribution': best_dist,
                    'params': best_params,
                    'aic': best_aic
                }
                print(f"  Best fit: {best_dist.name} (AIC: {best_aic:.2f})")
            else:
                print(f"  No suitable distribution found for {param}")
    
    def fit_joint_distribution_copula(self, param_group='key_parameters'):
        """Fit joint distribution using copulas"""
        if not COPULAS_AVAILABLE:
            print("Copulas not available, using multivariate normal instead")
            return self.fit_joint_distribution_mvn(param_group)
        
        params = self.parameter_groups[param_group]
        available_params = [p for p in params if p in self.numeric_data.columns]
        
        if len(available_params) < 2:
            print(f"Not enough parameters available for {param_group}")
            return None
        
        # Get data for joint modeling
        joint_data = self.numeric_data[available_params].dropna()
        
        if len(joint_data) < 10:
            print(f"Not enough samples for joint modeling of {param_group}")
            return None
        
        print(f"Fitting joint distribution for {param_group} with {len(available_params)} parameters...")
        
        # Fit copula
        self.joint_model = GaussianMultivariate()
        self.joint_model.fit(joint_data)
        
        print(f"Joint model fitted successfully for {param_group}")
        return self.joint_model
    
    def fit_joint_distribution_mvn(self, param_group='key_parameters'):
        """Fit multivariate normal distribution as fallback"""
        params = self.parameter_groups[param_group]
        available_params = [p for p in params if p in self.numeric_data.columns]
        
        if len(available_params) < 2:
            print(f"Not enough parameters available for {param_group}")
            return None
        
        # Get data for joint modeling
        joint_data = self.numeric_data[available_params].dropna()
        
        if len(joint_data) < 10:
            print(f"Not enough samples for joint modeling of {param_group}")
            return None
        
        print(f"Fitting multivariate normal for {param_group} with {len(available_params)} parameters...")
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(joint_data)
        
        # Fit multivariate normal
        mean = np.mean(scaled_data, axis=0)
        cov = np.cov(scaled_data.T)
        
        self.joint_model = {
            'type': 'multivariate_normal',
            'mean': mean,
            'cov': cov,
            'scaler': scaler,
            'param_names': available_params
        }
        
        print(f"Multivariate normal fitted successfully for {param_group}")
        return self.joint_model
    
    def sample_parameters(self, n_samples=100, param_group='key_parameters'):
        """Sample new parameter sets from fitted distributions"""
        if self.joint_model is None:
            print("No joint model fitted. Please fit a joint distribution first.")
            return None
        
        if COPULAS_AVAILABLE and hasattr(self.joint_model, 'sample'):
            # Sample from copula
            samples = self.joint_model.sample(n_samples)
        else:
            # Sample from multivariate normal
            if self.joint_model['type'] == 'multivariate_normal':
                samples_scaled = np.random.multivariate_normal(
                    self.joint_model['mean'], 
                    self.joint_model['cov'], 
                    n_samples
                )
                # Inverse transform
                samples = self.joint_model['scaler'].inverse_transform(samples_scaled)
                samples = pd.DataFrame(samples, columns=self.joint_model['param_names'])
        
        return samples
    
    def validate_fit(self, output_folder, param_group='key_parameters'):
        """Validate the fitted joint distribution"""
        if self.joint_model is None:
            print("No joint model to validate")
            return
        
        # Generate samples
        samples = self.sample_parameters(n_samples=1000, param_group=param_group)
        
        if samples is None:
            return
        
        # Compare distributions
        params = self.parameter_groups[param_group]
        available_params = [p for p in params if p in self.numeric_data.columns]
        
        n_params = len(available_params)
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(15, 8))
        if n_params == 1:
            axes = [axes]
        axes = axes.flatten()
        
        for i, param in enumerate(available_params):
            if param in samples.columns:
                axes[i].hist(self.numeric_data[param], bins=20, alpha=0.7, 
                           label='Original', density=True)
                axes[i].hist(samples[param], bins=20, alpha=0.7, 
                           label='Sampled', density=True)
                axes[i].set_title(param)
                axes[i].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'param_fits.png'), dpi=300)
        plt.show()
    
    def export_for_simulation(self, param_group='key_parameters', n_samples=100):
        """Export parameters in format suitable for indel-seq-gen"""
        samples = self.sample_parameters(n_samples, param_group)
        
        if samples is None:
            return None
        
        # Convert back from transformed parameters if needed
        export_data = samples.copy()
        
        # Inverse logit transform for frequencies
        for col in export_data.columns:
            if col.endswith('_logit'):
                original_col = col.replace('_logit', '')
                if original_col.startswith('freq_'):
                    # Inverse logit: exp(x) / (1 + exp(x))
                    export_data[original_col] = np.exp(export_data[col]) / (1 + np.exp(export_data[col]))
                    export_data = export_data.drop(columns=[col])
        
        # Inverse log transform for rates
        for col in export_data.columns:
            if col.endswith('_log'):
                original_col = col.replace('_log', '')
                export_data[original_col] = np.exp(export_data[col])
                export_data = export_data.drop(columns=[col])
        
        return export_data
    
    def plot_parameter_correlations(self, output_folder):
        """Plot correlation matrix of parameters"""
        # Focus on most relevant parameters
        key_params = []
        for group in self.parameter_groups.values():
            key_params.extend([p for p in group if p in self.numeric_data.columns])
        
        if len(key_params) > 20:  # Limit to avoid overcrowding
            key_params = key_params[:20]
        
        corr_data = self.numeric_data[key_params].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'parameter_correlations.png'), dpi = 300)
        plt.close()

def main():
    print("Starting parameter fitting workflow...")
    
    if len(sys.argv) < 2:
        print("Usage: python src/model_gen_aa/modelfit.py <output_folder> [parameter_file] [model_path] [n_samples]")
        sys.exit(1)
    
    output_folder = sys.argv[1]
    parameter_file = sys.argv[2] if len(sys.argv) > 2 else 'none'
    model_path = sys.argv[3] if len(sys.argv) > 3 else 'none'
    n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    if (parameter_file == model_path) or (parameter_file != 'none' and model_path != 'none'):
        print("Error: Please specify either a parameter file or a model path, not both or none.")
        sys.exit(1)
    
    if not os.path.exists(output_folder):
        print(f"Creating output folder: {output_folder}")
        os.makedirs(output_folder)
    
    parameter_group = 'key_parameters'
    print(model_path)
    if parameter_file != 'none':
        if not os.path.exists(parameter_file):
            print(f"Error: Parameter file '{parameter_file}' does not exist.")
            sys.exit(1)
            
        #Initialize the fitter with the parameter file
        fitter = PhylogeneticParameterFitter(parameter_file)
        
        # Preprocess data
        print("Preprocessing data...")
        fitter.preprocess_data()
        
        # Plot correlations
        print("Plotting parameter correlations...")
        fitter.plot_parameter_correlations(output_folder)
        
        # Fit marginal distributions
        print("Fitting marginal distributions...")
        fitter.fit_marginal_distributions()
        
        # Fit joint distribution for amino acid frequencies
        print("Fitting joint distribution for amino acid frequencies...")
        fitter.fit_joint_distribution_copula(parameter_group)
        
        # Validate fit
        print("Validating fit...")
        fitter.validate_fit(output_folder, parameter_group)
        
        with open(os.path.join(output_folder, 'model.pkl'), 'wb') as f:
            pickle.dump(fitter, f, pickle.HIGHEST_PROTOCOL)
        print(f"Joint distributions saved to {output_folder}/model.pkl")
        
    elif model_path != 'none':
        with open(model_path, 'rb') as f:
            fitter = pickle.load(f)
    
        # Export parameters for simulation
        print("Exporting parameters for simulation...")
        simulation_params = fitter.export_for_simulation(parameter_group, n_samples)
    
        if simulation_params is not None:
            print(f"Generated {len(simulation_params)} parameter sets for simulation")
            print("First few parameter sets:")
            print(simulation_params.head())
            
            # Save to CSV
            simulation_params.to_csv(os.path.join(output_folder, 'simulated_phylo_parameters.csv'), index=False)
            print(f"Parameters saved to '{os.path.join(output_folder, 'simulated_phylo_parameters.csv')}'")
    

if __name__ == "__main__":
    main()
    
    '''
    # Example of how to use the results
    print("\n" + "="*50)
    print("USAGE EXAMPLE:")
    print("="*50)
    print("# Generate new parameter sets:")
    print("new_params = fitter.sample_parameters(n_samples=10)")
    print("print(new_params)")
    print("\n# Export for indel-seq-gen:")
    print("indel_params = fitter.export_for_simulation('indel_parameters', n_samples=10)")
    print("print(indel_params)")
    '''