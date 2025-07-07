#!/usr/bin/env python3
"""
Phylogenetic Parameter Distribution Fitting Script

This script fits joint distributions to phylogenetic parameters extracted from TreeFam alignments
for use in realistic sequence simulation with indel-seq-gen.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, anderson
import json
import pickle
import base64
from pathlib import Path
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
    
    def __init__(self, csv_file=None):
        """Initialize with CSV file containing TreeFam parameters"""
        if csv_file is not None:
            self.data = pd.read_csv(csv_file)
        else:
            self.data = None
        self.fitted_distributions = {}
        self.joint_models = {}  # Store multiple joint models
        self.parameter_groups = self._define_parameter_groups()
        
    def _serialize_numpy_arrays(self, obj):
        """Convert numpy arrays to base64 strings for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return {
                '__numpy_array__': True,
                'data': base64.b64encode(obj.tobytes()).decode('utf-8'),
                'shape': obj.shape,
                'dtype': str(obj.dtype)
            }
        elif isinstance(obj, dict):
            return {key: self._serialize_numpy_arrays(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_numpy_arrays(item) for item in obj]
        else:
            return obj
    
    def _deserialize_numpy_arrays(self, obj):
        """Convert base64 strings back to numpy arrays"""
        if isinstance(obj, dict):
            if '__numpy_array__' in obj:
                data = base64.b64decode(obj['data'].encode('utf-8'))
                return np.frombuffer(data, dtype=obj['dtype']).reshape(obj['shape'])
            else:
                return {key: self._deserialize_numpy_arrays(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._deserialize_numpy_arrays(item) for item in obj]
        else:
            return obj
        """Define logical groups of parameters for joint modeling"""
    def _define_parameter_groups(self):
        """Define logical groups of parameters for joint modeling"""
        if self.data is None:
            return {}
        return {
            'amino_acid_frequencies': [col for col in self.data.columns if col.startswith('freq_')],
            'substitution_model': ['gamma_shape', 'prop_invariant'],
            'indel_parameters': ['indel_rate', 'mean_indel_length'],
            'tree_parameters': ['tree_length', 'crown_age', 'n_tips'],
            'diversification': ['speciation_rate', 'extinction_rate', 'net_diversification'],
            'sequence_properties': ['n_sequences', 'alignment_length', 'total_gaps']
        }
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        # Remove non-numeric columns and handle missing values
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.numeric_data = self.data[numeric_cols].copy()
        
        # Handle missing values
        self.numeric_data = self.numeric_data.dropna()
        
        # Log-transform rate parameters (they're often log-normal)
        rate_params = ['gamma_shape', 'indel_rate', 'speciation_rate', 'extinction_rate', 
                      'net_diversification', 'tree_length']
        
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
    
    def fit_joint_distribution_copula(self, param_group='amino_acid_frequencies'):
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
        joint_model = GaussianMultivariate()
        joint_model.fit(joint_data)
        
        # Store the model
        self.joint_models[param_group] = joint_model
        
        print(f"Joint model fitted successfully for {param_group}")
        return joint_model
    
    def fit_joint_distribution_mvn(self, param_group='amino_acid_frequencies'):
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
        
        joint_model = {
            'type': 'multivariate_normal',
            'mean': mean,
            'cov': cov,
            'scaler': scaler,
            'param_names': available_params
        }
        
        # Store the model
        self.joint_models[param_group] = joint_model
        
        print(f"Multivariate normal fitted successfully for {param_group}")
        return joint_model
    
    def save_models(self, filename='phylo_models.json'):
        """Save fitted distributions and joint models to JSON file"""
        save_data = {
            'fitted_distributions': {},
            'joint_models': {},
            'parameter_groups': self.parameter_groups
        }
        
        # Save marginal distributions
        for param, dist_info in self.fitted_distributions.items():
            save_data['fitted_distributions'][param] = {
                'distribution_name': dist_info['distribution'].name,
                'params': list(dist_info['params']),
                'aic': dist_info['aic']
            }
        
        # Save joint models
        for group_name, model in self.joint_models.items():
            if COPULAS_AVAILABLE and hasattr(model, 'to_dict'):
                # For copula models
                try:
                    save_data['joint_models'][group_name] = {
                        'type': 'copula',
                        'model_dict': model.to_dict()
                    }
                except:
                    # If copula serialization fails, use pickle fallback
                    model_bytes = pickle.dumps(model)
                    save_data['joint_models'][group_name] = {
                        'type': 'copula_pickle',
                        'model_data': base64.b64encode(model_bytes).decode('utf-8')
                    }
            else:
                # For multivariate normal models
                model_serialized = self._serialize_numpy_arrays(model)
                save_data['joint_models'][group_name] = {
                    'type': 'multivariate_normal',
                    'model_data': model_serialized
                }
        
        # Save to JSON
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Models saved to {filename}")
        return filename
    
    def load_models(self, filename='phylo_models.json'):
        """Load fitted distributions and joint models from JSON file"""
        try:
            with open(filename, 'r') as f:
                save_data = json.load(f)
        except FileNotFoundError:
            print(f"File {filename} not found")
            return False
        
        # Load parameter groups
        self.parameter_groups = save_data.get('parameter_groups', {})
        
        # Load marginal distributions
        self.fitted_distributions = {}
        for param, dist_info in save_data.get('fitted_distributions', {}).items():
            dist_name = dist_info['distribution_name']
            # Get distribution object from scipy.stats
            dist_obj = getattr(stats, dist_name)
            self.fitted_distributions[param] = {
                'distribution': dist_obj,
                'params': tuple(dist_info['params']),
                'aic': dist_info['aic']
            }
        
        # Load joint models
        self.joint_models = {}
        for group_name, model_info in save_data.get('joint_models', {}).items():
            model_type = model_info['type']
            
            if model_type == 'copula' and COPULAS_AVAILABLE:
                try:
                    model = GaussianMultivariate()
                    model = model.from_dict(model_info['model_dict'])
                    self.joint_models[group_name] = model
                except:
                    print(f"Failed to load copula model for {group_name}")
                    
            elif model_type == 'copula_pickle' and COPULAS_AVAILABLE:
                try:
                    model_bytes = base64.b64decode(model_info['model_data'].encode('utf-8'))
                    model = pickle.loads(model_bytes)
                    self.joint_models[group_name] = model
                except:
                    print(f"Failed to load pickled copula model for {group_name}")
                    
            elif model_type == 'multivariate_normal':
                model_data = self._deserialize_numpy_arrays(model_info['model_data'])
                self.joint_models[group_name] = model_data
        
        print(f"Models loaded from {filename}")
        print(f"Loaded {len(self.fitted_distributions)} marginal distributions")
        print(f"Loaded {len(self.joint_models)} joint models")
        return True
    
    def sample_parameters(self, n_samples=100, param_group='amino_acid_frequencies'):
        """Sample new parameter sets from fitted distributions"""
        if param_group not in self.joint_models:
            print(f"No joint model fitted for {param_group}. Available groups: {list(self.joint_models.keys())}")
            return None
        
        joint_model = self.joint_models[param_group]
        
        if COPULAS_AVAILABLE and hasattr(joint_model, 'sample'):
            # Sample from copula
            samples = joint_model.sample(n_samples)
        else:
            # Sample from multivariate normal
            if isinstance(joint_model, dict) and joint_model.get('type') == 'multivariate_normal':
                samples_scaled = np.random.multivariate_normal(
                    joint_model['mean'], 
                    joint_model['cov'], 
                    n_samples
                )
                # Inverse transform
                samples = joint_model['scaler'].inverse_transform(samples_scaled)
                samples = pd.DataFrame(samples, columns=joint_model['param_names'])
            else:
                print(f"Unknown joint model type for {param_group}")
                return None
        
        return samples
    
    def validate_fit(self, param_group='amino_acid_frequencies'):
        """Validate the fitted joint distribution"""
        if param_group not in self.joint_models:
            print(f"No joint model fitted for {param_group}")
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
        plt.show()
    
    def export_for_simulation(self, param_group='amino_acid_frequencies', n_samples=100):
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
    
    def list_available_models(self):
        """List all available fitted models"""
        print("Available marginal distributions:")
        for param in self.fitted_distributions:
            dist_info = self.fitted_distributions[param]
            print(f"  {param}: {dist_info['distribution'].name} (AIC: {dist_info['aic']:.2f})")
        
        print("\nAvailable joint models:")
        for group in self.joint_models:
            model = self.joint_models[group]
            if isinstance(model, dict) and model.get('type') == 'multivariate_normal':
                print(f"  {group}: Multivariate Normal ({len(model['param_names'])} parameters)")
            else:
                print(f"  {group}: Copula model")

def load_and_sample(model_file='phylo_models.json', param_group='amino_acid_frequencies', n_samples=100):
    """
    Convenience function to load models and sample parameters
    Usage: samples = load_and_sample('my_models.json', 'amino_acid_frequencies', 50)
    """
    fitter = PhylogeneticParameterFitter()
    if fitter.load_models(model_file):
        return fitter.sample_parameters(n_samples, param_group)
    else:
        print("Failed to load models")
        return None
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
        plt.show()

#

'''
def main():
    """Main function to demonstrate the workflow"""
    # Initialize the fitter
    fitter = PhylogeneticParameterFitter('protein_evolution_parameters_with_rates-1.csv')
    
    # Preprocess data
    print("Preprocessing data...")
    fitter.preprocess_data()
    
    # Plot correlations
    print("Plotting parameter correlations...")
    fitter.plot_parameter_correlations()
    
    # Fit marginal distributions
    print("Fitting marginal distributions...")
    fitter.fit_marginal_distributions()
    
    # Fit joint distribution for amino acid frequencies
    print("Fitting joint distribution for amino acid frequencies...")
    fitter.fit_joint_distribution_copula('amino_acid_frequencies')
    
    # Fit additional parameter groups
    print("Fitting joint distribution for substitution model parameters...")
    fitter.fit_joint_distribution_mvn('substitution_model')
    
    print("Fitting joint distribution for indel parameters...")
    fitter.fit_joint_distribution_mvn('indel_parameters')
    
    # Save all models
    print("Saving models...")
    fitter.save_models('phylo_models.json')
    
    # List available models
    print("Available fitted models:")
    fitter.list_available_models()
    
    # Validate fit
    print("Validating fit...")
    fitter.validate_fit('amino_acid_frequencies')
    
    # Export parameters for simulation
    print("Exporting parameters for simulation...")
    simulation_params = fitter.export_for_simulation('amino_acid_frequencies', n_samples=50)
    
    if simulation_params is not None:
        print(f"Generated {len(simulation_params)} parameter sets for simulation")
        print("First few parameter sets:")
        print(simulation_params.head())
        
        # Save to CSV
        simulation_params.to_csv('simulated_phylo_parameters.csv', index=False)
        print("Parameters saved to 'simulated_phylo_parameters.csv'")
    
    return fitter
'''


def main():
    print('Started model fitting/sampling script')
    if len(sys.argv) < 2:
        print("Usage: python modelfit.py <mode> <input_file> [output_file]")
        print("Example: python modelfit.py fit parameter_file.csv model.json")
        print("Example: python modelfit.py sample model.json output.csv")
        sys.exit(1)
    
    mode = sys.argv[1].strip().lower()
    input_file = sys.argv[2]
    
    if (mode == "fit"):
        output_file = sys.argv[3] if len(sys.argv) > 3 else "model.json"
    else:
        output_file = sys.argv[3] if len(sys.argv) > 3 else "output.csv"
    
    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    if (mode == 'fit'):
        fitter = PhylogeneticParameterFitter(input_file)
        
        print('Preprocessing data ...')
        fitter.preprocess_data()
        
        # Plot correlations
        print("Plotting parameter correlations...")
        fitter.plot_parameter_correlations()
        
        # Fit marginal distributions
        print("Fitting marginal distributions...")
        fitter.fit_marginal_distributions()
        
        # Fit joint distribution for amino acid frequencies
        print("Fitting joint distribution for amino acid frequencies...")
        fitter.fit_joint_distribution_copula('amino_acid_frequencies')
        
        # Fit additional parameter groups
        print("Fitting joint distribution for substitution model parameters...")
        fitter.fit_joint_distribution_mvn('substitution_model')
        
        print("Fitting joint distribution for indel parameters...")
        fitter.fit_joint_distribution_mvn('indel_parameters')        
        
        print("Fitting joint distribution for tree parameters...")
        fitter.fit_joint_distribution_mvn('')
        
        # Validate fit
        print("Validating fit...")
        fitter.validate_fit('amino_acid_frequencies')
                
    elif (mode == 'sample'):
        pass


if __name__ == "__main__":
    # Run the main workflow
    fitter = main()
    
    # Example of how to use the results
    print("\n" + "="*50)
    print("USAGE EXAMPLE:")
    print("="*50)
    print("# Generate new parameter sets:")
    print("new_params = fitter.sample_parameters(n_samples=10, param_group='amino_acid_frequencies')")
    print("print(new_params)")
    print("\n# Export for indel-seq-gen:")
    print("indel_params = fitter.export_for_simulation('indel_parameters', n_samples=10)")
    print("print(indel_params)")
    print("\n# Load models later:")
    print("from phylo_param_fitting import load_and_sample")
    print("samples = load_and_sample('phylo_models.json', 'amino_acid_frequencies', 50)")
    print("print(samples)")
    print("\n# Or create new fitter and load:")
    print("new_fitter = PhylogeneticParameterFitter()")
    print("new_fitter.load_models('phylo_models.json')")
    print("samples = new_fitter.sample_parameters(100, 'substitution_model')")
    print("print(samples)")