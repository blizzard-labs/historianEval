#!/usr/bin/env python3
"""
Phylogenetic Parameter Distribution Fitting Script

This script fits individual joint distributions for each birth-death model configuration
to phylogenetic parameters extracted from TreeFam alignments for use in realistic 
sequence simulation with indel-seq-gen.
"""


import os
import sys
import pickle
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import kstest, anderson
import warnings
import dendropy
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
    Class for fitting distributions to phylogenetic parameters with individual BD models
    """
    
    def __init__(self, csv_file):
        """Initialize with CSV file containing TreeFam parameters"""
        self.data = pd.read_csv(csv_file)
        self.fitted_distributions = {}
        self.bd_models = {}  # Store individual models for each BD configuration
        self.bd_model_probs = {}  # Store probability of each BD model
        self.parameter_groups = self._define_parameter_groups()
        
    def _define_parameter_groups(self):
        """Define logical groups of parameters for joint modeling"""
        
        # Core parameters that will be included in each BD-specific model
        return {
            'core_parameters': ['n_sequences_tips', 'alignment_length', 'crown_age',
                               'gamma_shape', 'prop_invariant',
                               'insertion_rate', 'deletion_rate',
                               'mean_insertion_length', 'mean_deletion_length',
                               'normalized_colless_index', 'gamma',
                               'best_BD_speciation_rate', 'best_BD_extinction_rate',
                               'best_BD_speciation_alpha', 'best_BD_extinction_alpha']
        }
    
    def _get_bd_model_columns(self):
        """Get birth-death model indicator columns"""
        bd_cols = [col for col in self.data.columns if col.startswith('best_B') and not col.startswith('best_BD')]
        return bd_cols
    
    def _split_data_by_bd_model(self):
        """Split data into separate datasets for each BD model"""
        bd_cols = self._get_bd_model_columns()
        
        if not bd_cols:
            print("Warning: No BD model columns found!")
            return {'default': self.data}
        
        bd_datasets = {}
        
        for bd_col in bd_cols:
            # Get rows where this BD model is selected (value = 1)
            bd_data = self.data[self.data[bd_col] == 1].copy()
            
            if len(bd_data) > 0:
                # Remove all BD indicator columns from this dataset
                bd_data = bd_data.drop(columns=bd_cols)
                bd_datasets[bd_col] = bd_data
                print(f"BD model '{bd_col}': {len(bd_data)} samples")
            else:
                print(f"BD model '{bd_col}': 0 samples (skipping)")
        
        # Calculate model probabilities
        total_samples = sum(len(dataset) for dataset in bd_datasets.values())
        self.bd_model_probs = {model: len(dataset)/total_samples 
                              for model, dataset in bd_datasets.items()}
        
        print(f"\nBD Model Probabilities:")
        for model, prob in self.bd_model_probs.items():
            print(f"  {model}: {prob:.3f}")
        
        return bd_datasets
    
    def preprocess_data(self):
        """Clean and preprocess the data for each BD model"""
        # Split data by BD model first
        self.bd_datasets = self._split_data_by_bd_model()
        
        # Preprocess each BD model dataset separately
        self.bd_numeric_data = {}
        
        for bd_model, bd_data in self.bd_datasets.items():
            print(f"\nPreprocessing data for BD model: {bd_model}")
            
            # Remove non-numeric columns and handle missing values
            numeric_cols = bd_data.select_dtypes(include=[np.number]).columns
            numeric_data = bd_data[numeric_cols].copy()
            
            # Handle missing values
            numeric_data = numeric_data.dropna()
            
            # Log-transform rate parameters (they're often log-normal)
            rate_params = ['gamma_shape', 'insertion_rate', 'deletion_rate', 'mean_insertion_length', 
                          'mean_deletion_length', 'best_BD_speciation_rate', 'best_BD_extinction_rate']
            
            for param in rate_params:
                if param in numeric_data.columns:
                    # Add small constant to avoid log(0)
                    numeric_data[param + '_log'] = np.log(numeric_data[param] + 1e-10)
            
            # Logit-transform proportions
            prop_params = ['prop_invariant'] + [col for col in numeric_data.columns if col.startswith('freq_')]
            
            for param in prop_params:
                if param in numeric_data.columns:
                    # Logit transform: log(p/(1-p)), handling edge cases
                    p = numeric_data[param].clip(1e-10, 1-1e-10)
                    numeric_data[param + '_logit'] = np.log(p / (1 - p))
            
            self.bd_numeric_data[bd_model] = numeric_data
            print(f"  Preprocessed shape: {numeric_data.shape}")
        
        # Store transformation info for bias correction
        self.transformations = {
            'log_params': [p for p in rate_params if any(p in data.columns for data in self.bd_numeric_data.values())],
            'logit_params': [p for p in prop_params if any(p in data.columns for data in self.bd_numeric_data.values())]
        }
        
        return self.bd_numeric_data
    
    def fit_marginal_distributions(self, param_subset=None):
        """Fit marginal distributions to individual parameters for each BD model"""
        # Common distributions to test
        distributions = [
            stats.norm, stats.lognorm, stats.gamma, stats.beta, stats.expon,
            stats.weibull_min, stats.uniform, stats.chi2, stats.invgamma,
            stats.pareto, stats.genextreme, stats.gumbel_r, stats.gumbel_l,
            stats.logistic, stats.laplace, stats.t, stats.genpareto
        ]
        
        self.fitted_distributions = {}
        
        for bd_model, numeric_data in self.bd_numeric_data.items():
            print(f"\nFitting marginal distributions for BD model: {bd_model}")
            self.fitted_distributions[bd_model] = {}
            
            param_cols = param_subset if param_subset else numeric_data.columns
            
            for param in param_cols:
                if param not in numeric_data.columns:
                    continue
                    
                data = numeric_data[param].dropna()
                if len(data) < 10:  # Skip if too few data points
                    continue
                    
                best_dist = None
                best_params = None
                best_aic = np.inf
                
                print(f"  Fitting distributions for {param}...")
                
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
                    self.fitted_distributions[bd_model][param] = {
                        'distribution': best_dist,
                        'params': best_params,
                        'aic': best_aic
                    }
                    print(f"    Best fit: {best_dist.name} (AIC: {best_aic:.2f})")
                else:
                    print(f"    No suitable distribution found for {param}")
    
    def fit_joint_distribution_copula(self, param_group='core_parameters'):
        """Fit joint distribution using copulas for each BD model"""
        if not COPULAS_AVAILABLE:
            print("Copulas not available, using multivariate normal instead")
            return self.fit_joint_distribution_mvn(param_group)
        
        params = self.parameter_groups[param_group]
        
        for bd_model, numeric_data in self.bd_numeric_data.items():
            print(f"\nFitting joint distribution for BD model: {bd_model}")
            
            available_params = [p for p in params if p in numeric_data.columns]
            
            if len(available_params) < 2:
                print(f"  Not enough parameters available for {bd_model}")
                continue
            
            # Get data for joint modeling
            joint_data = numeric_data[available_params].dropna()
            
            if len(joint_data) < 10:
                print(f"  Not enough samples for joint modeling of {bd_model}")
                continue
            
            print(f"  Fitting with {len(available_params)} parameters and {len(joint_data)} samples...")
            
            # Fit copula
            joint_model = GaussianMultivariate()
            joint_model.fit(joint_data)
            
            self.bd_models[bd_model] = {
                'model': joint_model,
                'param_names': available_params,
                'n_samples': len(joint_data)
            }
            
            print(f"  Joint model fitted successfully for {bd_model}")
        
        return self.bd_models
    
    def fit_joint_distribution_mvn(self, param_group='core_parameters'):
        """Fit multivariate normal distribution as fallback for each BD model"""
        params = self.parameter_groups[param_group]
        
        for bd_model, numeric_data in self.bd_numeric_data.items():
            print(f"\nFitting multivariate normal for BD model: {bd_model}")
            
            available_params = [p for p in params if p in numeric_data.columns]
            
            if len(available_params) < 2:
                print(f"  Not enough parameters available for {bd_model}")
                continue
            
            # Get data for joint modeling
            joint_data = numeric_data[available_params].dropna()
            
            if len(joint_data) < 10:
                print(f"  Not enough samples for joint modeling of {bd_model}")
                continue
            
            print(f"  Fitting with {len(available_params)} parameters and {len(joint_data)} samples...")
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(joint_data)
            
            # Fit multivariate normal
            mean = np.mean(scaled_data, axis=0)
            cov = np.cov(scaled_data.T)
            
            self.bd_models[bd_model] = {
                'type': 'multivariate_normal',
                'mean': mean,
                'cov': cov,
                'scaler': scaler,
                'param_names': available_params,
                'n_samples': len(joint_data)
            }
            
            print(f"  Multivariate normal fitted successfully for {bd_model}")
        
        return self.bd_models
    
    def sample_bd_model(self):
        """Sample a BD model according to the learned probabilities"""
        if not self.bd_model_probs:
            return list(self.bd_models.keys())[0] if self.bd_models else None
        
        models = list(self.bd_model_probs.keys())
        probs = list(self.bd_model_probs.values())
        return np.random.choice(models, p=probs)
    
    def generate_pop_string(self, b_rate_string, d_rate_string, num_increments, initial_pop):
        pops = [initial_pop]
        
        for i in range(num_increments):
            pops.append(int((b_rate_string[i] - d_rate_string[i]) * pops[i]))
        
        return pops
    
    def generate_rate_strings(self, present_rate, function, alpha, max_time=1, num_intervals=5):
        time_points = np.linspace(0, max_time, num_intervals)
        rates = []
        
        if function.lower() == 'cst':
            rates = [present_rate]*num_intervals
            
        elif function.lower() == 'exp':
            #r(t) = r0 * exp(t * alpha) ==> r0 = r(t) / (exp(t * alpha))
            r0 = present_rate / (math.exp(max_time * alpha))
            
            for t in time_points:
                rates.append(max(0, r0 * math.exp(t * alpha)))
            
        elif function.lower() == 'lin':
            #r(t) = r0 + alpha * t ==> r0 = r(t) - alpha * t
            r0 = present_rate - alpha * max_time
            
            for t in time_points:
                rates.append(max(0, r0 + alpha * t))
        
        return rates
    
    def generate_bd_tree(self, pbirth_rate, pdeath_rate, bd_model, birth_alpha, death_alpha, max_time=10.0, max_attempts=15):
        """
        Generate a birth-death tree based on the specified model.
        
        Returns:
            DendroPy Tree object
        """
        
        success = False
        
        while not success and (max_attempts > 0):
            birth_rates = self.generate_rate_strings(pbirth_rate, bd_model[1:4], birth_alpha, max_time=max_time)
            death_rates = self.generate_rate_strings(pdeath_rate, bd_model[5:], death_alpha, max_time=max_time)

            assert len(birth_rates) == len(death_rates), "Birth and death rate lists must have same length"
            
            num_intervals = len(birth_rates)
            interval_duration = max_time / num_intervals
            
            # Start with a single lineage at max_time

            tree = dendropy.Tree()
            root = dendropy.Node()
            root.age = max_time
            tree.seed_node = root

            active_nodes = [root]
            current_time = max_time
            
            # Simulate each time interval (going forward in time)
            for i in range(num_intervals): 
                birth_rate = birth_rates[i]
                death_rate = death_rates[i]
                interval_end = current_time - interval_duration
                
                new_active_nodes = []
                
                for node in active_nodes:
                    # Simulate births and deaths in this interval
                    node_time = current_time
                    
                    while node_time > interval_end:
                        # Time to next event (birth or death)
                        total_rate = birth_rate + death_rate
                        if total_rate <= 0:
                            node_time = interval_end
                            break
                            
                        dt = np.random.exponential(1.0 / total_rate)
                        node_time -= dt
                        
                        if node_time <= interval_end:
                            break
                        
                        # Determine if birth or death
                        if np.random.random() < birth_rate / total_rate:
                            # Birth event - create two child nodes
                            left_child = dendropy.Node()
                            right_child = dendropy.Node()
                            left_child.age = node_time
                            right_child.age = node_time
                            
                            node.add_child(left_child)
                            node.add_child(right_child)
                            
                            # Set edge lengths
                            left_child.edge.length = current_time - node_time
                            right_child.edge.length = current_time - node_time
                            
                            # Update active nodes
                            new_active_nodes.extend([left_child, right_child])
                            break  # This lineage split
                        else:
                            # Death event - lineage goes extinct
                            break  # This lineage dies
                    else:
                        # Lineage survives the interval
                        new_active_nodes.append(node)
                
                active_nodes = new_active_nodes
                current_time = interval_end
                
                if not active_nodes:  # All lineages extinct
                    break
            
            # Set final edge lengths to present (time 0)
            for node in active_nodes:
                if node.edge and node.edge.length is not None:
                    node.edge.length += current_time
                elif node.edge:
                    node.edge.length = current_time
            
            # Only keep trees with surviving lineages
            if not active_nodes:
                max_attempts -= 1
            else:
                success = True
                # Assign taxa to tips
                tree.randomly_assign_taxa(create_required_taxa=True)
        if success: 
            print('Treebuilding success!')
        
        return success
    
    
    
    def sample_parameters(self, n_samples=100, param_group='core_parameters',
                         min_n_sequences_tips=20, max_n_sequences_tips=100,
                         q_scale=100, bias_correction=True):
        """
        Sample parameters from the appropriate BD model
        """
        if not self.bd_models:
            print("No BD models fitted. Please fit joint distributions first.")
            return None
        
        all_samples = []
        
        for i in range(n_samples):
            # Sample BD model
            bd_model = self.sample_bd_model()
            
            if bd_model not in self.bd_models:
                continue
            
            model_info = self.bd_models[bd_model]
            available_params = model_info['param_names']
            
            # Calculate bias corrections for this BD model
            bias_corrections = {}
            if bias_correction and hasattr(self, 'transformations'):
                numeric_data = self.bd_numeric_data[bd_model]
                for param in self.transformations.get('log_params', []):
                    if param in numeric_data.columns:
                        original_mean = numeric_data[param].mean()
                        log_values = np.log(numeric_data[param] + 1e-10)
                        naive_back_transform = np.exp(log_values.mean())
                        bias_corrections[param] = original_mean / naive_back_transform
            
            # Define parameters to exclude from bounds constraints
            unrestricted_params = {
                'n_sequences_tips',
                'best_BD_speciation_alpha', 'best_BD_extinction_alpha'
             }
            
            # Calculate bounds for this BD model
            param_bounds = {}
            restricted_params = []
            numeric_data = self.bd_numeric_data[bd_model]
            
            for param in available_params:
                if param in unrestricted_params:
                    continue
                    
                data = numeric_data[param].dropna()
                
                if len(data) < 5:
                    continue
                
                # Remove outliers
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)
                lower_bound = q1 - 1.5 * (q3 - q1)
                upper_bound = q3 + 1.5 * (q3 - q1)
                
                filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
                
                if len(filtered_data) < 5:
                    filtered_data = data
                
                param_bounds[param] = {
                    'median': np.percentile(filtered_data, 50),
                    'lower': np.percentile(filtered_data, 50 - q_scale/2),
                    'upper': np.percentile(filtered_data, 50 + q_scale/2)
                }
                restricted_params.append(param)
            
            # Generate samples with rejection sampling
            max_attempts = 500
            attempts = 0
            
            print(f'\nBeginning rejection sampling #{i} for model {str(bd_model)}...')
            while attempts < max_attempts:
                print(f'\rAttempt Number: {attempts}', end='')
                attempts += 1
                
                # Generate sample from this BD model
                if COPULAS_AVAILABLE and hasattr(model_info.get('model'), 'sample'):
                    sample_df = model_info['model'].sample(1)
                    sample = sample_df.iloc[0]
                else:
                    if model_info.get('type') == 'multivariate_normal':
                        sample_scaled = np.random.multivariate_normal(
                            model_info['mean'], 
                            model_info['cov'], 
                            1
                        )[0]
                        sample_array = model_info['scaler'].inverse_transform([sample_scaled])[0]
                        sample = pd.Series(sample_array, index=model_info['param_names'])
                
                # Apply rejection criteria
                accept_sample = True
                
                # Check bounds for restricted parameters
                for param in restricted_params:
                    if param in sample.index:
                        value = sample[param]
                        if not (param_bounds[param]['lower'] <= value <= param_bounds[param]['upper']):
                            accept_sample = False
                            break
                
                # Additional constraint for n_sequences_tips
                if (accept_sample and 'n_sequences_tips' in sample.index and 
                    (sample['n_sequences_tips'] <= min_n_sequences_tips or 
                     sample['n_sequences_tips'] >= max_n_sequences_tips)):
                    accept_sample = False
                
                # Additional constraint for non-extinct trees #!New feature... not yet tested!!!
                
                if accept_sample: #TODO: Add to if statement... check if each parameter of sample exists within sample.index
                    best_model = bd_model.replace('best_','')
                    
                    accept_sample = self.generate_bd_tree(sample['best_BD_speciation_rate'], sample['best_BD_extinction_rate'], best_model, sample['best_BD_speciation_alpha'], sample['best_BD_extinction_alpha'], max_time=sample['crown_age'])
                        
                    
                    '''
                    birth_rates = self.generate_rate_strings(sample['best_BD_speciation_rate'], best_model[1:4], sample['best_BD_speciation_alpha'], max_time=sample['crown_age'])
                    death_rates = self.generate_rate_strings(sample['best_BD_extinction_rate'], best_model[5:], sample['best_BD_extinction_alpha'], max_time=sample['crown_age'])
                    
                    if np.float64(birth_rates[0]) <= np.float64(death_rates[0]):
                        accept_sample = False
                    else:
                        print('\naccepted sample, ', birth_rates[0], ', ', death_rates[0])
                        print('acepted... alpha:', sample['best_BD_speciation_alpha'])
                    '''
                    
                    
                    '''
                    #Population survives first two increments
                    pop_string = self.generate_pop_string(birth_rates, death_rates, 2, sample['n_sequence_tips'])
                    
                    for pop in pop_string:
                        if pop <= 0:
                            accept_sample = False
                    '''
                    
                                    
                if accept_sample:
                    # Apply bias corrections
                    sample_dict = sample.to_dict()
                    for param, correction_factor in bias_corrections.items():
                        if param + '_log' in sample_dict:
                            sample_dict[param + '_log'] += np.log(correction_factor)
                    
                    # Add BD model indicator
                    sample_dict['bd_model'] = bd_model
                    all_samples.append(sample_dict)
                    break
        
        if all_samples:
            result = pd.DataFrame(all_samples)
            print(f"Generated {len(result)} samples from {len(set(result['bd_model']))} BD models")
            
            # Print BD model distribution in samples
            bd_counts = result['bd_model'].value_counts()
            print("BD model distribution in samples:")
            for model, count in bd_counts.items():
                print(f"  {model}: {count} ({count/len(result)*100:.1f}%)")
            
            return result
        else:
            print("Failed to generate sufficient samples")
            return None
    
    def validate_fit(self, output_folder, param_group='core_parameters'):
        """Validate the fitted joint distributions for each BD model"""
        if not self.bd_models:
            print("No BD models to validate")
            return
        
        # Create validation plots folder
        validation_folder = os.path.join(output_folder, 'validation_plots')
        os.makedirs(validation_folder, exist_ok=True)
        
        # Generate samples for each BD model separately
        all_validation_results = {}
        
        for bd_model in self.bd_models.keys():
            print(f"\nValidating BD model: {bd_model}")
            
            # Generate samples specifically from this BD model
            model_samples = []
            model_info = self.bd_models[bd_model]
            
            # Generate 500 samples from this specific model
            for _ in range(500):
                if COPULAS_AVAILABLE and hasattr(model_info.get('model'), 'sample'):
                    sample_df = model_info['model'].sample(1)
                    sample = sample_df.iloc[0]
                else:
                    if model_info.get('type') == 'multivariate_normal':
                        sample_scaled = np.random.multivariate_normal(
                            model_info['mean'], 
                            model_info['cov'], 
                            1
                        )[0]
                        sample_array = model_info['scaler'].inverse_transform([sample_scaled])[0]
                        sample = pd.Series(sample_array, index=model_info['param_names'])
                
                model_samples.append(sample)
            
            if model_samples:
                samples_df = pd.DataFrame(model_samples)
                all_validation_results[bd_model] = samples_df
        
        # Create validation plots for each BD model
        for bd_model, samples in all_validation_results.items():
            self._create_bd_model_validation_plot(validation_folder, bd_model, samples)
        
        # Create summary comparison across all BD models
        self._create_bd_models_summary_plot(validation_folder, all_validation_results)

    def _create_bd_model_validation_plot(self, output_folder, bd_model, samples):
        """Create validation plot for a specific BD model"""
        original_data = self.bd_numeric_data[bd_model]
        available_params = [p for p in samples.columns if p in original_data.columns]
        
        if len(available_params) == 0:
            return
        
        # Limit to most important parameters for readability
        if len(available_params) > 12:
            important_params = ['n_sequences_tips', 'alignment_length', 'crown_age', 
                              'gamma_shape', 'prop_invariant', 'insertion_rate', 
                              'deletion_rate', 'mean_insertion_length', 'mean_deletion_length']
            available_params = [p for p in important_params if p in available_params][:12]
        
        n_params = len(available_params)
        n_cols = min(4, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig_width = n_cols * 5
        fig_height = n_rows * 4
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        fig.suptitle(f'Parameter Validation: {bd_model}', fontsize=16, fontweight='bold')
        
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes_flat = axes.flatten() if n_params > 1 else axes
        
        for i, param in enumerate(available_params):
            ax = axes_flat[i]
            self._plot_parameter_comparison(ax, param, samples, original_data)
        
        # Hide unused subplots
        for i in range(n_params, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        safe_bd_model = bd_model.replace('_', '-').lower()
        plt.savefig(os.path.join(output_folder, f'validation_{safe_bd_model}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_parameter_comparison(self, ax, param, samples, original_data):
        """Plot comparison for a single parameter"""
        orig_data = original_data[param].dropna()
        samp_data = samples[param].dropna()
        
        # Determine appropriate number of bins
        n_bins = min(30, max(10, int(np.sqrt(len(orig_data)))))
        
        # Create histograms with better styling
        ax.hist(orig_data, bins=n_bins, alpha=0.6, 
               label='Original', density=True, color='skyblue', edgecolor='black')
        ax.hist(samp_data, bins=n_bins, alpha=0.6, 
               label='Sampled', density=True, color='orange', edgecolor='black')
        
        # Improve titles and labels
        clean_param_name = param.replace('_', ' ').title()
        ax.set_title(clean_param_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add statistics text
        orig_mean, orig_std = np.mean(orig_data), np.std(orig_data)
        samp_mean, samp_std = np.mean(samp_data), np.std(samp_data)
        
        stats_text = f'Orig: μ={orig_mean:.3f}, σ={orig_std:.3f}\nSamp: μ={samp_mean:.3f}, σ={samp_std:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=8, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def _create_bd_models_summary_plot(self, output_folder, all_validation_results):
        """Create summary comparison across all BD models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('BD Models Summary Comparison', fontsize=16, fontweight='bold')
        
        # Model sample counts
        model_counts = {model: len(samples) for model, samples in all_validation_results.items()}
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(model_counts)), list(model_counts.values()))
        ax1.set_xlabel('BD Model')
        ax1.set_ylabel('Training Samples')
        ax1.set_title('Training Sample Counts by BD Model')
        ax1.set_xticks(range(len(model_counts)))
        ax1.set_xticklabels([model.replace('best_', '') for model in model_counts.keys()], 
                           rotation=45, ha='right')
        
        # Parameter coverage comparison
        ax2 = axes[0, 1]
        param_coverage = {}
        for model, samples in all_validation_results.items():
            param_coverage[model] = len(samples.columns)
        
        bars = ax2.bar(range(len(param_coverage)), list(param_coverage.values()))
        ax2.set_xlabel('BD Model')
        ax2.set_ylabel('Number of Parameters')
        ax2.set_title('Parameter Coverage by BD Model')
        ax2.set_xticks(range(len(param_coverage)))
        ax2.set_xticklabels([model.replace('best_', '') for model in param_coverage.keys()], 
                           rotation=45, ha='right')
        
        # Model probabilities
        ax3 = axes[1, 0]
        if self.bd_model_probs:
            models = list(self.bd_model_probs.keys())
            probs = list(self.bd_model_probs.values())
            bars = ax3.bar(range(len(models)), probs)
            ax3.set_xlabel('BD Model')
            ax3.set_ylabel('Probability')
            ax3.set_title('BD Model Selection Probabilities')
            ax3.set_xticks(range(len(models)))
            ax3.set_xticklabels([model.replace('best_', '') for model in models], 
                               rotation=45, ha='right')
        
        # Parameter distribution comparison (example with one key parameter)
        ax4 = axes[1, 1]
        key_param = 'n_sequences_tips'  # Choose a parameter that should be in all models
        
        for model, samples in all_validation_results.items():
            if key_param in samples.columns:
                ax4.hist(samples[key_param], alpha=0.5, label=model.replace('best_', ''), 
                        density=True, bins=20)
        
        ax4.set_xlabel(key_param.replace('_', ' ').title())
        ax4.set_ylabel('Density')
        ax4.set_title(f'{key_param.replace("_", " ").title()} Distribution by BD Model')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'bd_models_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_for_simulation(self, param_group='core_parameters', n_samples=100):
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
            if col.endswith('_log'):
                original_col = col.replace('_log', '')
                export_data[original_col] = np.exp(export_data[col])
                export_data = export_data.drop(columns=[col])
                
            # Integer values
            if col in ['n_sequences_tips', 'alignment_length']:
                export_data[col] = export_data[col].astype(int)
            
            if col in ['prop_invariant', 'insertion_rate', 'deletion_rate', 'mean_insertion_length', 
                      'mean_deletion_length', 'normalized_colless_index']:
                export_data[col] = np.maximum(export_data[col], 0)
        
        # Create one-hot encoding from bd_model column
        if 'bd_model' in export_data.columns:
            # Create one-hot columns for each BD model
            bd_models = export_data['bd_model'].unique()
            for bd_model in bd_models:
                export_data[bd_model] = (export_data['bd_model'] == bd_model).astype(int)
            
            # Remove the original bd_model column
            export_data = export_data.drop(columns=['bd_model'])
        
        # Calculate indel_rate
        if 'insertion_rate' in export_data.columns and 'deletion_rate' in export_data.columns:
            export_data['indel_rate'] = export_data['insertion_rate'] + export_data['deletion_rate']
        
        return export_data
    
    def plot_parameter_correlations(self, output_folder):
        """Plot correlation matrix of parameters for each BD model"""
        # Create correlations plots folder
        correlations_folder = os.path.join(output_folder, 'correlation_plots')
        os.makedirs(correlations_folder, exist_ok=True)
        
        for bd_model, numeric_data in self.bd_numeric_data.items():
            # Focus on most relevant parameters
            key_params = []
            for group in self.parameter_groups.values():
                key_params.extend([p for p in group if p in numeric_data.columns])
            
            if len(key_params) > 20:  # Limit to avoid overcrowding
                key_params = key_params[:20]
            
            if len(key_params) < 2:
                continue
            
            corr_data = numeric_data[key_params].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f')
            plt.title(f'Parameter Correlation Matrix: {bd_model}')
            plt.tight_layout()
            
            safe_bd_model = bd_model.replace('_', '-').lower()
            plt.savefig(os.path.join(correlations_folder, f'correlations_{safe_bd_model}.png'), dpi=300)
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
    
    parameter_group = 'core_parameters'
    
    if parameter_file != 'none':
        if not os.path.exists(parameter_file):
            print(f"Error: Parameter file '{parameter_file}' does not exist.")
            sys.exit(1)
            
        print(f'Loading parameter file: {parameter_file}')
        
        #Initialize the fitter with the parameter file
        fitter = PhylogeneticParameterFitter(parameter_file)
        
        # Preprocess data
        print("Preprocessing data...")
        fitter.preprocess_data()
        
        # Plot correlations for each BD model
        print("Plotting parameter correlations for each BD model...")
        fitter.plot_parameter_correlations(output_folder)
        
        # Fit marginal distributions for each BD model
        print("Fitting marginal distributions for each BD model...")
        fitter.fit_marginal_distributions()
        
        # Fit joint distribution for each BD model
        print("Fitting joint distributions for each BD model...")
        fitter.fit_joint_distribution_copula(parameter_group)
        
        # Validate fit for each BD model
        print("Validating fits for each BD model...")
        fitter.validate_fit(output_folder, parameter_group)
        
        # Save the complete fitter object
        with open(os.path.join(output_folder, 'bd_models.pkl'), 'wb') as f:
            pickle.dump(fitter, f, pickle.HIGHEST_PROTOCOL)
        print(f"BD models saved to {output_folder}/bd_models.pkl")
        
    elif model_path != 'none':
        # Handle both old and new model file names
        possible_paths = [
            model_path,
            os.path.join(os.path.dirname(model_path), 'bd_models.pkl'),
            model_path.replace('model.pkl', 'bd_models.pkl')
        ]
        
        model_file = None
        for path in possible_paths:
            if os.path.exists(path):
                model_file = path
                break
        
        if model_file is None:
            print(f"Error: Model file not found. Tried:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\nPlease ensure you have trained a model first using the parameter file option.")
            sys.exit(1)
            
        print(f'Loading model: {model_file}')
        
        try:
            with open(model_file, 'rb') as f:
                fitter = pickle.load(f)
        except Exception as e:
            print(f"Error loading model file: {e}")
            print("The model file may be corrupted or from an incompatible version.")
            sys.exit(1)
    
        # Export parameters for simulation
        print("Exporting parameters for simulation...")
        simulation_params = fitter.export_for_simulation(parameter_group, n_samples)
    
        if simulation_params is not None:
            print(f"Generated {len(simulation_params)} parameter sets for simulation")
            print("First few parameter sets:")
            print(simulation_params.head())
            
            # Show BD model distribution in the exported parameters
            bd_cols = [col for col in simulation_params.columns if col.startswith('best_B') and not col.startswith('best_BD')]
            if bd_cols:
                print("\nBD model distribution in exported parameters:")
                for bd_col in bd_cols:
                    count = simulation_params[bd_col].sum()
                    if count > 0:
                        print(f"  {bd_col}: {count} ({count/len(simulation_params)*100:.1f}%)")
            
            # Save to CSV
            simulation_params.to_csv(os.path.join(output_folder, 'simulated_phylo_parameters.csv'), index=False)
            print(f"Parameters saved to '{os.path.join(output_folder, 'simulated_phylo_parameters.csv')}'")
            
            # Create a summary of the exported parameters
            summary_stats = simulation_params.describe()
            summary_stats.to_csv(os.path.join(output_folder, 'parameter_summary_stats.csv'))
            print(f"Parameter summary statistics saved to '{os.path.join(output_folder, 'parameter_summary_stats.csv')}'")
        else:
            print("Failed to generate simulation parameters.")


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
    print("indel_params = fitter.export_for_simulation('core_parameters', n_samples=10)")
    print("print(indel_params)")
    print("\n# The exported parameters will include BD model indicators as one-hot encoded columns")
    print("# Each row will have exactly one BD model set to 1, others set to 0")
    '''