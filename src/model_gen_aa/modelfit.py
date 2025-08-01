#!/usr/bin/env python3
"""
Enhanced Phylogenetic Parameter Distribution Fitting with Mixed Variable Support and Copulas

This version fits proper marginal distributions to each continuous parameter and then uses
copulas to model the joint dependence structure, providing much better modeling of
complex phylogenetic parameter distributions.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
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

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

class MarginalDistributionFitter:
    """Helper class to fit marginal distributions to individual parameters"""
    
    def __init__(self):
        # Extended list of distributions to try
        self.candidate_distributions = [
            stats.norm, stats.lognorm, stats.gamma, stats.beta, stats.expon,
            stats.weibull_min, stats.uniform, stats.chi2, stats.invgamma,
            stats.pareto, stats.genextreme, stats.gumbel_r, stats.gumbel_l,
            stats.logistic, stats.laplace, stats.t, stats.genpareto
        ]
        
    def fit_best_distribution(self, data, param_name=None):
        """
        Fit the best distribution to the data using multiple criteria
        """
        data = np.array(data).flatten()
        data = data[~np.isnan(data)]  # Remove NaN values
        
        if len(data) < 10:
            print(f"Warning: Too few data points ({len(data)}) for {param_name}, using normal distribution")
            return {'distribution': stats.norm, 'params': stats.norm.fit(data), 'score': -np.inf}
        
        results = []
        
        for distribution in self.candidate_distributions:
            try:
                # Handle special cases for bounded distributions
                if distribution == stats.beta:
                    # Beta requires data in [0,1]
                    if data.min() < 0 or data.max() > 1:
                        continue
                elif distribution == stats.uniform:
                    # Uniform is simple
                    params = (data.min(), data.max() - data.min())
                else:
                    # Fit parameters
                    params = distribution.fit(data)
                
                # Calculate goodness of fit using multiple criteria
                # 1. Kolmogorov-Smirnov test
                ks_stat, ks_p = kstest(data, lambda x: distribution.cdf(x, *params))
                
                # 2. Anderson-Darling test (if available for this distribution)
                try:
                    if distribution == stats.norm:
                        ad_stat, ad_crit, ad_sig = anderson(data, dist='norm')
                        ad_score = -ad_stat  # Lower is better
                    elif distribution == stats.expon:
                        ad_stat, ad_crit, ad_sig = anderson(data, dist='expon')
                        ad_score = -ad_stat
                    else:
                        ad_score = 0  # Not available
                except:
                    ad_score = 0
                
                # 3. Log-likelihood
                try:
                    log_likelihood = np.sum(distribution.logpdf(data, *params))
                    if not np.isfinite(log_likelihood):
                        log_likelihood = -np.inf
                except:
                    log_likelihood = -np.inf
                
                # 4. AIC (Akaike Information Criterion)
                k = len(params)  # number of parameters
                n = len(data)
                if log_likelihood != -np.inf:
                    aic = 2 * k - 2 * log_likelihood
                else:
                    aic = np.inf
                
                # Combined score (lower is better for AIC, higher for others)
                # Prioritize KS test p-value and log-likelihood
                combined_score = aic #ks_p * 0.5 + (log_likelihood / n) * 0.5 # + ad_score * 0.2
                #asdf
                results.append({
                    'distribution': distribution,
                    'params': params,
                    'ks_stat': ks_stat,
                    'ks_p': ks_p,
                    'log_likelihood': log_likelihood,
                    'aic': aic,
                    'score': combined_score,
                    'ad_score': ad_score
                })
                
            except Exception as e:
                # Skip distributions that fail to fit
                continue
        
        if not results:
            print(f"Warning: No distributions could be fitted to {param_name}, using normal")
            params = stats.norm.fit(data)
            return {'distribution': stats.norm, 'params': params, 'score': -np.inf}
        
        # Sort by combined score (higher is better)
        results.sort(key=lambda x: x['score'], reverse=True)
        best_fit = results[0]
        
        if param_name:
            print(f"Best distribution for {param_name}: {best_fit['distribution'].name} "
                  f"(score: {best_fit['score']:.4f}, KS p-value: {best_fit['ks_p']:.4f})")
        
        return best_fit
    
    def validate_fit(self, data, fit_result, param_name=None):
        """Validate the fitted distribution with diagnostic plots"""
        distribution = fit_result['distribution']
        params = fit_result['params']
        
        # Generate theoretical quantiles
        theoretical_quantiles = distribution.ppf(np.linspace(0.01, 0.99, 100), *params)
        
        # Q-Q plot data
        sorted_data = np.sort(data)
        empirical_quantiles = np.linspace(0, 1, len(sorted_data))
        theoretical_values = distribution.ppf(empirical_quantiles, *params)
        
        return {
            'qq_empirical': sorted_data,
            'qq_theoretical': theoretical_values,
            'pdf_x': theoretical_quantiles,
            'pdf_y': distribution.pdf(theoretical_quantiles, *params)
        }


class MixedPhylogeneticParameterFitter:
    """
    Enhanced fitter that properly handles mixed continuous/categorical parameters
    with marginal distribution fitting and copulas
    """
    
    def __init__(self, csv_file):
        """Initialize with CSV file containing TreeFam parameters"""
        self.data = pd.read_csv(csv_file)
        self.fitted_distributions = {}
        self.marginal_fitter = MarginalDistributionFitter()
        self.joint_model = None
        self.categorical_model = None
        self.continuous_model = None
        self.parameter_groups = self._define_parameter_groups()
        
    def _define_parameter_groups(self):
        """Define parameter groups with explicit categorical/continuous separation"""
        return {
            'continuous_parameters': [
                'n_sequences_tips', 'alignment_length', 'crown_age',
                'gamma_shape', 'prop_invariant',
                'insertion_rate', 'deletion_rate',
                'mean_insertion_length', 'mean_deletion_length',
                'normalized_colless_index', 'gamma',
                'best_BD_speciation_rate', 'best_BD_extinction_rate',
                'best_BD_speciation_alpha', 'best_BD_extinction_alpha'
            ],
            'categorical_parameters': [
                'best_BCSTDCST', 'best_BEXPDCST', 'best_BLINDCST', 
                'best_BCSTDEXP', 'best_BEXPDEXP', 'best_BLINDEXP', 
                'best_BCSTDLIN', 'best_BEXPDLIN', 'best_BLINDLIN'
            ]
        }
    
    def preprocess_data(self):
        """Enhanced preprocessing that handles mixed variable types"""
        # Remove non-numeric columns and handle missing values
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.numeric_data = self.data[numeric_cols].copy()
        self.numeric_data = self.numeric_data.dropna()
                
        # Separate continuous and categorical variables
        continuous_params = self.parameter_groups['continuous_parameters']
        categorical_params = self.parameter_groups['categorical_parameters']
        
        # Filter to available columns
        self.continuous_cols = [p for p in continuous_params if p in self.numeric_data.columns]
        self.categorical_cols = [p for p in categorical_params if p in self.numeric_data.columns]
        
        print(f"Continuous parameters: {len(self.continuous_cols)}")
        print(f"Categorical parameters: {len(self.categorical_cols)}")
        
        # Create categorical variable from one-hot encoding
        if self.categorical_cols:
            self._create_categorical_variable()
        
        print(f"Preprocessed data shape: {self.numeric_data.shape}")
        return self.numeric_data
    
    def _create_categorical_variable(self):
        """Convert one-hot encoded columns to single categorical variable"""
        # Find which model is selected for each row
        categorical_data = self.numeric_data[self.categorical_cols]
        
        # Handle rows where no model is selected (all zeros)
        row_sums = categorical_data.sum(axis=1)
        zero_rows = row_sums == 0
        
        if zero_rows.any():
            print(f"Warning: {zero_rows.sum()} rows have no model selected. Using most common model.")
            # Find most common model
            model_counts = categorical_data.sum(axis=0)
            most_common_model = model_counts.idxmax()
            
            # Assign most common model to zero rows
            for idx in categorical_data[zero_rows].index:
                self.numeric_data.at[idx, most_common_model] = 1
        
        # Handle rows with multiple models selected (sum > 1)
        multi_rows = row_sums > 1
        if multi_rows.any():
            print(f"Warning: {multi_rows.sum()} rows have multiple models selected. Using first selected.")
            for idx in categorical_data[multi_rows].index:
                # Find first model that's selected
                selected_models = categorical_data.loc[idx] == 1
                first_model = selected_models.idxmax()
                
                # Set all to 0, then set first to 1
                self.numeric_data.loc[idx, self.categorical_cols] = 0
                self.numeric_data.at[idx, first_model] = 1
        
        # Create single categorical variable
        bd_model_map = {col: col.replace('best_', '') for col in self.categorical_cols}
        
        self.numeric_data['bd_model_category'] = 'BCSTDCST'  # default
        for idx, row in self.numeric_data.iterrows():
            for col in self.categorical_cols:
                if row[col] == 1:
                    self.numeric_data.at[idx, 'bd_model_category'] = bd_model_map[col]
                    break
        
        # Calculate category probabilities for sampling
        self.category_probs = self.numeric_data['bd_model_category'].value_counts(normalize=True)
        print("Birth-death model distribution:")
        print(self.category_probs)
    
    def fit_marginal_distributions(self):
        """Fit marginal distributions to each continuous parameter - CORRECTED VERSION"""
        print("Fitting marginal distributions to continuous parameters...")
        
        self.marginal_fits = {}
        
        for param in self.continuous_cols:
            if param in self.numeric_data.columns:
                print(f"\\nFitting marginal distribution for {param}...")
                original_data = self.numeric_data[param].dropna()
                
                if len(original_data) < 10:
                    print(f"Skipping {param} - insufficient data ({len(original_data)} points)")
                    continue
                
                # Store original data info
                data_for_fitting = original_data.copy()
                
                heavy_tail_params = [
                    'insertion_rate', 'deletion_rate',
                    'mean_insertion_length', 'mean_deletion_length'
                    'best_BD_speciation_rate', 'best_BD_extinction_rate',
                    'best_BD_speciation_alpha', 'best_BD_extinction_alpha'
                ]
                
                # Apply transforms - BUT FIT TO ORIGINAL DATA FIRST
                log_transform = False
                logit_transform = False
                
                # Check if we should try log transform for heavy-tailed params
                if param in heavy_tail_params and (original_data > 0).all():
                    # Try both original and log-transformed, pick better fit
                    
                    # Fit to original data
                    fit_original = self.marginal_fitter.fit_best_distribution(original_data, f"{param}_original")
                    
                    # Try log-transformed data
                    log_data = np.log1p(original_data)
                    fit_log = self.marginal_fitter.fit_best_distribution(log_data, f"{param}_log")
                    
                    # Compare fits (higher KS p-value is better)
                    if fit_log.get('ks_p', 0) > fit_original.get('ks_p', 0) and fit_log.get('ks_p', 0) > 0.05:
                        print(f"  Using log-transformed data for {param}")
                        data_for_fitting = log_data
                        log_transform = True
                        fit_result = fit_log
                    else:
                        print(f"  Using original data for {param}")
                        fit_result = fit_original
                else:
                    # Standard fitting to original data
                    fit_result = self.marginal_fitter.fit_best_distribution(original_data, param)
                
                # Handle bounded parameters (0,1) with logit transform if needed
                bounded_params = ['prop_invariant']
                if param in bounded_params and not log_transform:
                    if (original_data > 0).all() and (original_data < 1).all():
                        eps = 1e-6
                        clipped_data = np.clip(original_data, eps, 1-eps)
                        logit_data = np.log(clipped_data/(1-clipped_data))
                        
                        fit_logit = self.marginal_fitter.fit_best_distribution(logit_data, f"{param}_logit")
                        
                        # Compare with original fit
                        if fit_logit.get('ks_p', 0) > fit_result.get('ks_p', 0) and fit_logit.get('ks_p', 0) > 0.05:
                            print(f"  Using logit-transformed data for {param}")
                            data_for_fitting = logit_data
                            logit_transform = True
                            fit_result = fit_logit
                
                # Store complete information
                fit_result['log_transform'] = log_transform
                fit_result['logit_transform'] = logit_transform
                fit_result['original_data'] = original_data
                fit_result['fitted_data'] = data_for_fitting
                fit_result['param_name'] = param
                
                self.marginal_fits[param] = fit_result
        
        print(f"\\nSuccessfully fitted marginal distributions for {len(self.marginal_fits)} parameters")
        return self.marginal_fits
    
    def fit_copula_models(self):
        """Fit copula models for each birth-death category"""
        print("Fitting copula models for joint dependence structure...")
        
        if not COPULAS_AVAILABLE:
            print("Warning: Copulas library not available. Falling back to multivariate normal.")
            return self.fit_multivariate_normal_models()
        
        self.copula_models = {}
        
        # Get list of parameters with fitted marginal distributions
        available_params = list(self.marginal_fits.keys())
        
        for category in self.category_probs.index:
            category_mask = self.numeric_data['bd_model_category'] == category
            category_data = self.numeric_data[category_mask][available_params]
            
            if len(category_data) < 10:
                print(f"Skipping {category} - insufficient data ({len(category_data)} samples)")
                continue
            
            print(f"Fitting copula model for {category} ({len(category_data)} samples)")
            
            try:
                # Transform data to uniform marginals using fitted distributions
                uniform_data = np.zeros_like(category_data)
                
                for i, param in enumerate(available_params):
                    marginal_fit = self.marginal_fits[param]
                    distribution = marginal_fit['distribution']
                    params = marginal_fit['params']
                    
                    # Transform to uniform using CDF
                    uniform_data[:, i] = distribution.cdf(category_data[param].values, *params)
                    
                    # Clip to avoid numerical issues
                    uniform_data[:, i] = np.clip(uniform_data[:, i], 1e-12, 1-1e-12)
                
                # Convert to DataFrame for copulas library
                uniform_df = pd.DataFrame(uniform_data, columns=available_params)
                
                # Fit Gaussian copula
                copula = GaussianMultivariate()
                copula.fit(uniform_df)
                
                self.copula_models[category] = {
                    'copula': copula,
                    'param_names': available_params,
                    'n_params': len(available_params)
                }
                
            except Exception as e:
                print(f"Error fitting copula for {category}: {e}")
                # Fallback to correlation matrix approach
                self.copula_models[category] = self._fit_correlation_fallback(
                    category_data, available_params)
        
        print(f"Successfully fitted copula models for {len(self.copula_models)} categories")
        return self.copula_models
    
    def _fit_correlation_fallback(self, category_data, available_params):
        """Fallback method using correlation matrix when copulas fail"""
        print("Using correlation matrix fallback...")
        
        # Transform to uniform using marginal CDFs
        uniform_data = np.zeros((len(category_data), len(available_params)))
        
        for i, param in enumerate(available_params):
            marginal_fit = self.marginal_fits[param]
            distribution = marginal_fit['distribution']
            params = marginal_fit['params']
            uniform_data[:, i] = distribution.cdf(category_data[param].values, *params)
            uniform_data[:, i] = np.clip(uniform_data[:, i], 1e-6, 1-1e-6)
        
        # Convert to normal using inverse normal CDF
        normal_data = stats.norm.ppf(uniform_data)
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(normal_data.T)
        
        # Add regularization if needed
        if np.linalg.det(corr_matrix) < 1e-10:
            corr_matrix += np.eye(len(available_params)) * 1e-6
        
        return {
            'type': 'correlation_fallback',
            'correlation_matrix': corr_matrix,
            'param_names': available_params,
            'n_params': len(available_params)
        }
    
    def fit_multivariate_normal_models(self):
        """Fallback to multivariate normal when copulas not available"""
        print("Fitting multivariate normal models (copulas not available)...")
        
        self.copula_models = {}
        available_params = list(self.marginal_fits.keys()) if hasattr(self, 'marginal_fits') else self.continuous_cols
        
        for category in self.category_probs.index:
            category_mask = self.numeric_data['bd_model_category'] == category
            category_data = self.numeric_data[category_mask][available_params]
            
            if len(category_data) < 5:
                continue
            
            # Standardize data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(category_data.dropna())
            
            mean = np.mean(scaled_data, axis=0)
            cov = np.cov(scaled_data.T)
            
            # Add regularization
            if np.linalg.det(cov) < 1e-10:
                cov += np.eye(len(mean)) * 1e-6
            
            self.copula_models[category] = {
                'type': 'multivariate_normal',
                'mean': mean,
                'cov': cov,
                'scaler': scaler,
                'param_names': available_params
            }
        
        return self.copula_models
    
    def fit_mixed_distribution(self):
        """Main method to fit the complete mixed distribution model"""
        # First fit marginal distributions
        self.fit_marginal_distributions()
        
        # Then fit copula models for joint dependence
        self.fit_copula_models()
        
        return self.copula_models
    
    def sample_mixed_parameters(self, n_samples=100, min_n_sequences_tips=20, 
                               n_std=4, bias_correction=True):
        """
        Enhanced sampling using marginal distributions and copulas
        """
        if not hasattr(self, 'copula_models') or not self.copula_models:
            print("No models fitted. Please fit mixed distribution first.")
            return None
        
        if not hasattr(self, 'marginal_fits') or not self.marginal_fits:
            print("No marginal distributions fitted.")
            return None
        
        samples = []
        
        for _ in range(n_samples):
            # Sample categorical variable (birth-death model)
            bd_model = np.random.choice(self.category_probs.index, p=self.category_probs.values)
            
            # Skip if no model available for this category
            if bd_model not in self.copula_models:
                bd_model = self.category_probs.index[0]
                if bd_model not in self.copula_models:
                    continue
            
            model = self.copula_models[bd_model]
            max_attempts = 500
            valid_sample = False
            
            for attempt in range(max_attempts):
                try:
                    sample_dict = {'bd_model_category': bd_model}
                    
                    # Sample from copula model
                    if model.get('type') == 'multivariate_normal':
                        # Fallback multivariate normal
                        print('fallback multivariate')
                        scaled_sample = np.random.multivariate_normal(model['mean'], model['cov'])
                        continuous_sample = model['scaler'].inverse_transform(scaled_sample.reshape(1, -1))[0]
                        
                        for param, value in zip(model['param_names'], continuous_sample):
                            sample_dict[param] = value
                    
                    elif model.get('type') == 'correlation_fallback':
                        # Correlation matrix fallback
                        normal_sample = np.random.multivariate_normal(
                            np.zeros(model['n_params']), model['correlation_matrix'])
                        uniform_sample = stats.norm.cdf(normal_sample)
                        
                        # Transform back using inverse marginal CDFs
                        for i, param in enumerate(model['param_names']):
                            marginal_fit = self.marginal_fits[param]
                            distribution = marginal_fit['distribution']
                            params = marginal_fit['params']
                            
                            # Get uniform value
                            uniform_value = uniform_sample[i]
                            uniform_value = np.clip(uniform_value, 1e-12, 1-1e-12)
                            
                            # Transform to parameter space using fitted distribution
                            value = distribution.ppf(uniform_value, *params)
                            
                            # Apply inverse transforms if they were used during fitting
                            if marginal_fit.get('log_transform', False):
                                value = np.expm1(value)  # Inverse of log1p
                            
                            if marginal_fit.get('logit_transform', False):
                                value = 1 / (1 + np.exp(-value))  # Inverse logit
                            
                            sample_dict[param] = value
                    
                    else:
                        # Full copula model
                        copula_sample = model['copula'].sample(1)
                        
                        # Transform back using inverse marginal CDFs
                        for param in model['param_names']:
                            if param in copula_sample.columns:
                                uniform_value = copula_sample[param].iloc[0]
                                uniform_value = np.clip(uniform_value, 1e-12, 1-1e-12)
                                
                                marginal_fit = self.marginal_fits[param]
                                distribution = marginal_fit['distribution']
                                params = marginal_fit['params']
                                
                                # Transform to parameter space
                                value = distribution.ppf(uniform_value, *params)
                                
                                # Apply inverse transforms
                                if marginal_fit.get('log_transform', False):
                                    value = np.expm1(value)
                                
                                if marginal_fit.get('logit_transform', False):
                                    value = 1 / (1 + np.exp(-value))
                                
                                sample_dict[param] = value
                    
                    # Add categorical variables (one-hot encoding)
                    for col in self.categorical_cols:
                        sample_dict[col] = 1 if col == f'best_{bd_model}' else 0
                    
                    # Apply constraints
                    constraints_passed = True
                    
                    # Essential constraints
                    if ('n_sequences_tips' in sample_dict and 
                        sample_dict['n_sequences_tips'] <= min_n_sequences_tips):
                        constraints_passed = False
                    
                    # Check for reasonable bounds (looser than before)
                    if constraints_passed:
                        for param in model['param_names']:
                            if param in sample_dict and param in self.numeric_data.columns:
                                value = sample_dict[param]
                                if not np.isfinite(value):
                                    constraints_passed = False
                                    break
                                
                                # Very loose bounds check
                                orig_data = self.numeric_data[param]
                                mean, std = orig_data.mean(), orig_data.std()
                                if abs(value - mean) > n_std * std:
                                    constraints_passed = False
                                    break
                    
                    if constraints_passed:
                        valid_sample = True
                        samples.append(sample_dict)
                        break
                        
                except Exception as e:
                    continue
            
            if not valid_sample:
                print(f"Warning: Could not generate valid sample for {bd_model}")
        
        if samples:
            result_df = pd.DataFrame(samples)
            print(f"Generated {len(result_df)} valid samples using marginal distributions + copulas")
            return result_df
        else:
            print("Failed to generate any valid samples")
            return None
    
    def export_for_simulation(self, n_samples=100, bias_correction=True):
        """Export mixed parameters for simulation"""
        samples = self.sample_mixed_parameters(n_samples, bias_correction=bias_correction)
        
        if samples is None:
            return None
        
        export_data = samples.copy()
        
        # Ensure proper data types and constraints
        if 'n_sequences_tips' in export_data.columns:
            export_data['n_sequences_tips'] = export_data['n_sequences_tips'].astype(int)
        if 'alignment_length' in export_data.columns:
            export_data['alignment_length'] = export_data['alignment_length'].astype(int)
        
        # Ensure non-negative values for rates
        rate_cols = ['prop_invariant', 'insertion_rate', 'deletion_rate', 
                    'mean_insertion_length', 'mean_deletion_length']
        for col in rate_cols:
            if col in export_data.columns:
                export_data[col] = np.maximum(export_data[col], 0)
        
        # Add combined indel rate
        if 'insertion_rate' in export_data.columns and 'deletion_rate' in export_data.columns:
            export_data['indel_rate'] = export_data['insertion_rate'] + export_data['deletion_rate']
        
        # Remove helper columns
        if 'bd_model_category' in export_data.columns:
            export_data = export_data.drop(columns=['bd_model_category'])
        
        return export_data
    
    def plot_marginal_fits(self, output_folder):
        """Create diagnostic plots for marginal distribution fits"""
        if not hasattr(self, 'marginal_fits'):
            print("No marginal fits to plot")
            return
        
        n_params = len(self.marginal_fits)
        n_cols = min(4, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (param, fit_result) in enumerate(self.marginal_fits.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get original data
            data = self.numeric_data[param].dropna()
            
            # Plot histogram
            ax.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', 
                   edgecolor='black', label='Data')
            
            # Plot fitted distribution
            distribution = fit_result['distribution']
            params = fit_result['params']
            
            x_range = np.linspace(data.min(), data.max(), 100)
            fitted_pdf = distribution.pdf(x_range, *params)
            ax.plot(x_range, fitted_pdf, 'r-', linewidth=2, 
                   label=f'{distribution.name}')
            
            ax.set_title(f'{param}\n{distribution.name} (p={fit_result["ks_p"]:.3f})')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'marginal_distribution_fits.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create Q-Q plots
        self._create_qq_plots(output_folder)
    
    def _create_qq_plots(self, output_folder):
        """Create Q-Q plots for marginal fits"""
        n_params = len(self.marginal_fits)
        n_cols = min(4, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten() if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (param, fit_result) in enumerate(self.marginal_fits.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get data and fit
            data = self.numeric_data[param].dropna()
            validation = self.marginal_fitter.validate_fit(data, fit_result, param)
            
            # Q-Q plot
            ax.scatter(validation['qq_theoretical'], validation['qq_empirical'], 
                      alpha=0.6, s=20)
            
            # Add diagonal line
            min_val = min(validation['qq_theoretical'].min(), validation['qq_empirical'].min())
            max_val = max(validation['qq_theoretical'].max(), validation['qq_empirical'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            ax.set_title(f'{param} Q-Q Plot')
            ax.set_xlabel('Theoretical Quantiles')
            ax.set_ylabel('Sample Quantiles')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'marginal_qq_plots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # ============================================================================
    # COMPATIBILITY METHODS (keeping original interface)
    # ============================================================================
    
    def sample_parameters(self, n_samples=100, param_group='key_parameters', **kwargs):
        """Compatibility wrapper"""
        return self.sample_mixed_parameters(n_samples, **kwargs)
    
    def _update_parameter_groups_for_compatibility(self):
        """Update parameter groups for compatibility"""
        all_params = self.continuous_cols + self.categorical_cols
        self.parameter_groups['key_parameters'] = all_params
    
    def validate_fit(self, output_folder, param_group='key_parameters'):
        """Enhanced validation with marginal distribution plots"""
        self._update_parameter_groups_for_compatibility()
        
        # Plot marginal distribution fits
        self.plot_marginal_fits(output_folder)
        
        # Generate samples and create comparison plots
        samples = self.sample_mixed_parameters(n_samples=1000)
        
        if samples is None:
            return
        
        # Rest of validation (reuse original methods)
        params = self.parameter_groups[param_group]
        available_params = [p for p in params if p in self.numeric_data.columns and p in samples.columns]
        
        param_categories = self._categorize_parameters(available_params)
        
        if len(available_params) > 12:
            self._create_categorized_plots(output_folder, param_categories, samples)
        else:
            self._create_single_organized_plot(output_folder, available_params, samples)
        
        # Create summary statistics plot
        self._create_summary_stats_plot(output_folder, available_params, samples)
        
        # Add mixed-model specific validation
        self._validate_mixed_model_specific(output_folder, samples)
        
        # Create marginal distribution summary
        self._create_marginal_summary(output_folder)
    
    def _create_marginal_summary(self, output_folder):
        """Create summary table of marginal distribution fits"""
        if not hasattr(self, 'marginal_fits'):
            return
        
        summary_data = []
        for param, fit_result in self.marginal_fits.items():
            summary_data.append({
                'Parameter': param,
                'Distribution': fit_result['distribution'].name,
                'KS_Statistic': fit_result['ks_stat'],
                'KS_P_Value': fit_result['ks_p'],
                'Log_Likelihood': fit_result['log_likelihood'],
                'AIC': fit_result['aic'],
                'Score': fit_result['score']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_folder, 'marginal_distribution_summary.csv'), index=False)
        
        # Create visualization of distribution choices
        plt.figure(figsize=(12, 8))
        
        # Count distribution types
        dist_counts = summary_df['Distribution'].value_counts()
        
        plt.subplot(2, 2, 1)
        dist_counts.plot(kind='bar', alpha=0.7)
        plt.title('Distribution Types Used')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        plt.hist(summary_df['KS_P_Value'], bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
        plt.xlabel('KS Test P-Value')
        plt.ylabel('Frequency')
        plt.title('Goodness of Fit Distribution')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.scatter(summary_df['AIC'], summary_df['KS_P_Value'], alpha=0.7)
        plt.xlabel('AIC')
        plt.ylabel('KS P-Value')
        plt.title('AIC vs. KS P-Value')
        
        plt.subplot(2, 2, 4)
        good_fits = summary_df['KS_P_Value'] >= 0.05
        plt.pie([good_fits.sum(), (~good_fits).sum()], 
                labels=['Good Fit (p≥0.05)', 'Poor Fit (p<0.05)'],
                autopct='%1.1f%%', startangle=90)
        plt.title('Fit Quality Summary')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'marginal_distribution_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Marginal distribution summary saved to {output_folder}/marginal_distribution_summary.csv")
    
    def _validate_mixed_model_specific(self, output_folder, samples):
        """Additional validation specific to mixed model approach"""
        
        # Validate one-hot encoding
        one_hot_cols = [c for c in samples.columns if c.startswith('best_B') and not c.startswith('best_BD')]
        if one_hot_cols:
            row_sums = samples[one_hot_cols].sum(axis=1)
            all_valid = (row_sums == 1).all()
            
            plt.figure(figsize=(15, 10))
            
            # Plot 1: Row sums distribution
            plt.subplot(2, 3, 1)
            plt.hist(row_sums, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(x=1, color='red', linestyle='--', label='Expected sum = 1')
            plt.xlabel('Sum of One-Hot Columns')
            plt.ylabel('Frequency')
            plt.title('One-Hot Encoding Validation')
            plt.legend()
            
            # Plot 2: Model distribution comparison
            plt.subplot(2, 3, 2)
            
            # Original distribution
            orig_counts = self.category_probs
            sampled_counts = samples['bd_model_category'].value_counts(normalize=True) if 'bd_model_category' in samples.columns else pd.Series()
            
            # If bd_model_category not in samples, reconstruct from one-hot
            if 'bd_model_category' not in samples.columns:
                bd_models = []
                for idx, row in samples.iterrows():
                    for col in one_hot_cols:
                        if row[col] == 1:
                            bd_models.append(col.replace('best_', ''))
                            break
                sampled_counts = pd.Series(bd_models).value_counts(normalize=True)
            
            x_pos = np.arange(len(orig_counts))
            width = 0.35
            
            plt.bar(x_pos - width/2, orig_counts.values, width, label='Original', alpha=0.7)
            
            # Align sampled counts with original order
            sampled_aligned = [sampled_counts.get(model, 0) for model in orig_counts.index]
            plt.bar(x_pos + width/2, sampled_aligned, width, label='Sampled', alpha=0.7)
            
            plt.xlabel('Birth-Death Model')
            plt.ylabel('Probability')
            plt.title('Model Distribution Comparison')
            plt.xticks(x_pos, orig_counts.index, rotation=45)
            plt.legend()
            
            # Plot 3: Copula model types
            plt.subplot(2, 3, 3)
            model_types = []
            for model in self.copula_models.values():
                if model.get('type') == 'correlation_fallback':
                    model_types.append('Correlation Fallback')
                elif model.get('type') == 'multivariate_normal':
                    model_types.append('Multivariate Normal')
                else:
                    model_types.append('Gaussian Copula')
            
            model_type_counts = pd.Series(model_types).value_counts()
            model_type_counts.plot(kind='bar', alpha=0.7)
            plt.title('Copula Model Types Used')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            # Plot 4: Parameter coverage
            plt.subplot(2, 3, 4)
            marginal_params = set(self.marginal_fits.keys()) if hasattr(self, 'marginal_fits') else set()
            continuous_params = set(self.continuous_cols)
            
            coverage_data = [
                len(marginal_params & continuous_params),  # Successfully fitted
                len(continuous_params - marginal_params)   # Failed to fit
            ]
            
            plt.pie(coverage_data, labels=['Successfully Fitted', 'Failed to Fit'],
                   autopct='%1.1f%%', startangle=90)
            plt.title('Marginal Distribution Coverage')
            
            # Plot 5: Distribution quality
            if hasattr(self, 'marginal_fits'):
                plt.subplot(2, 3, 5)
                p_values = [fit['ks_p'] for fit in self.marginal_fits.values()]
                good_fits = sum(1 for p in p_values if p >= 0.05)
                poor_fits = len(p_values) - good_fits
                
                plt.pie([good_fits, poor_fits], 
                       labels=[f'Good Fit (p≥0.05): {good_fits}', f'Poor Fit (p<0.05): {poor_fits}'],
                       autopct='%1.1f%%', startangle=90)
                plt.title('Marginal Fit Quality')
            
            # Plot 6: Sample validation metrics
            plt.subplot(2, 3, 6)
            validation_metrics = []
            
            # Check for NaN values
            nan_count = samples.isnull().sum().sum()
            validation_metrics.append(('NaN Values', nan_count))
            
            # Check for infinite values
            inf_count = np.isinf(samples.select_dtypes(include=[np.number])).sum().sum()
            validation_metrics.append(('Inf Values', inf_count))
            
            # Check one-hot encoding
            onehot_valid = (row_sums == 1).sum()
            onehot_invalid = len(samples) - onehot_valid
            validation_metrics.append(('Valid OneHot', onehot_valid))
            validation_metrics.append(('Invalid OneHot', onehot_invalid))
            
            metric_names = [m[0] for m in validation_metrics]
            metric_values = [m[1] for m in validation_metrics]
            
            bars = plt.bar(metric_names, metric_values, alpha=0.7)
            plt.title('Sample Validation Metrics')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            
            # Color code bars
            for i, bar in enumerate(bars):
                if 'Invalid' in metric_names[i] or metric_names[i] in ['NaN Values', 'Inf Values']:
                    if metric_values[i] > 0:
                        bar.set_color('red')
                    else:
                        bar.set_color('green')
                else:
                    bar.set_color('blue')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'enhanced_mixed_model_validation.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Enhanced validation: One-hot encoding {'PASSED' if all_valid else 'FAILED'}")
            print(f"Row sums range: {row_sums.min():.3f} to {row_sums.max():.3f}")
            print(f"NaN values: {nan_count}, Inf values: {inf_count}")
    
    def plot_parameter_correlations(self, output_folder):
        """Enhanced correlation plot for mixed model"""
        # Use only continuous parameters for correlation
        continuous_data = self.numeric_data[self.continuous_cols]
        
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Original data correlation
        plt.subplot(2, 2, 1)
        corr_data = continuous_data.corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
        plt.title('Original Data Correlation Matrix')
        
        # Plot 2: Transformed data correlation (if marginals fitted)
        if hasattr(self, 'marginal_fits'):
            plt.subplot(2, 2, 2)
            
            # Transform to uniform using fitted marginals
            uniform_data = pd.DataFrame(index=continuous_data.index)
            for param in self.continuous_cols:
                if param in self.marginal_fits:
                    fit = self.marginal_fits[param]
                    uniform_values = fit['distribution'].cdf(continuous_data[param].values, *fit['params'])
                    uniform_data[param] = np.clip(uniform_values, 1e-6, 1-1e-6)
            
            if not uniform_data.empty:
                uniform_corr = uniform_data.corr()
                sns.heatmap(uniform_corr, annot=True, cmap='coolwarm', center=0,
                           square=True, fmt='.2f', cbar_kws={'label': 'Correlation'})
                plt.title('Uniform-Transformed Correlation Matrix')
        
        # Plot 3: Categorical variable distribution
        plt.subplot(2, 2, 3)
        if hasattr(self, 'category_probs'):
            self.category_probs.plot(kind='bar', alpha=0.7)
            plt.title('Birth-Death Model Distribution')
            plt.ylabel('Probability')
            plt.xlabel('Model')
            plt.xticks(rotation=45)
        
        # Plot 4: Copula model summary
        plt.subplot(2, 2, 4)
        if hasattr(self, 'copula_models'):
            model_info = []
            for category, model in self.copula_models.items():
                model_type = model.get('type', 'gaussian_copula')
                n_params = model.get('n_params', 0)
                model_info.append(f"{category}\n({model_type})\n{n_params} params")
            
            plt.text(0.1, 0.9, 'Fitted Models:', transform=plt.gca().transAxes, 
                    fontsize=12, fontweight='bold')
            
            for i, info in enumerate(model_info):
                plt.text(0.1, 0.8 - i*0.15, info, transform=plt.gca().transAxes, 
                        fontsize=10, verticalalignment='top')
            
            plt.axis('off')
            plt.title('Copula Models Summary')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'enhanced_parameter_correlations.png'), dpi=300)
        plt.close()
    
    # ============================================================================
    # ORIGINAL PLOTTING METHODS (keeping for compatibility)
    # ============================================================================
    
    def _categorize_parameters(self, available_params):
        """Categorize parameters for better organization"""
        categories = {
            'Tree Structure': [p for p in available_params if any(keyword in p.lower() for keyword in 
                            ['n_sequences', 'crown_age', 'colless', 'gamma'])],
            'Sequence Evolution': [p for p in available_params if any(keyword in p.lower() for keyword in 
                                ['alignment_length', 'gamma_shape', 'prop_invariant'])],
            'Indel Parameters': [p for p in available_params if any(keyword in p.lower() for keyword in 
                                ['insertion', 'deletion'])],
            'Birth-Death Models': [p for p in available_params if p.startswith('best_B')]
        }
        
        # Add any uncategorized parameters to a general category
        categorized = set()
        for cat_params in categories.values():
            categorized.update(cat_params)
        
        uncategorized = [p for p in available_params if p not in categorized]
        if uncategorized:
            categories['Other'] = uncategorized
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v}
        
        return categories

    def _create_single_organized_plot(self, output_folder, available_params, samples):
        """Create a single well-organized plot for all parameters"""
        n_params = len(available_params)
        n_cols = min(4, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        # Dynamic figure size
        fig_width = n_cols * 5
        fig_height = n_rows * 4
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        
        # Handle different subplot configurations
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes_flat = axes.flatten() if n_params > 1 else axes
        
        for i, param in enumerate(available_params):
            ax = axes_flat[i]
            self._plot_parameter_comparison(ax, param, samples)
        
        # Hide unused subplots
        for i in range(n_params, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout(pad=2.0)
        plt.savefig(os.path.join(output_folder, 'param_fits.png'), 
                dpi=300, bbox_inches='tight')
        plt.close()

    def _create_categorized_plots(self, output_folder, param_categories, samples):
        """Create separate plots for each parameter category"""
        for category, params in param_categories.items():
            if not params:
                continue
                
            n_params = len(params)
            n_cols = min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            fig_width = n_cols * 5
            fig_height = n_rows * 4
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
            fig.suptitle(f'{category} Parameters', fontsize=16, fontweight='bold')
            
            if n_params == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            axes_flat = axes.flatten() if n_params > 1 else axes
            
            for i, param in enumerate(params):
                ax = axes_flat[i]
                self._plot_parameter_comparison(ax, param, samples)
            
            # Hide unused subplots
            for i in range(n_params, len(axes_flat)):
                axes_flat[i].set_visible(False)
            
            plt.tight_layout(pad=2.0)
            
            # Save with category name
            safe_category = category.replace(' ', '_').lower()
            plt.savefig(os.path.join(output_folder, f'param_fits_{safe_category}.png'), 
                    dpi=300, bbox_inches='tight')
            plt.close()

    def _plot_parameter_comparison(self, ax, param, samples):
        """Plot comparison for a single parameter with CORRECTED marginal distribution overlay"""
        orig_data = self.numeric_data[param].dropna()
        samp_data = samples[param].dropna()
        
        # Determine appropriate number of bins
        n_bins = min(30, max(10, int(np.sqrt(len(orig_data)))))
        
        # Create histograms
        ax.hist(orig_data, bins=n_bins, alpha=0.6, 
            label='Original', density=True, color='skyblue', edgecolor='black')
        ax.hist(samp_data, bins=n_bins, alpha=0.6, 
            label='Sampled', density=True, color='orange', edgecolor='black')
        
        # Overlay fitted marginal distribution if available - CORRECTED VERSION
        if hasattr(self, 'marginal_fits') and param in self.marginal_fits:
            fit = self.marginal_fits[param]
            
            if 'distribution' in fit and fit['distribution'] is not None:
                distribution = fit['distribution']
                params = fit['params']
                
                # CRITICAL FIX: Plot on the ORIGINAL data scale
                x_range = np.linspace(orig_data.min(), orig_data.max(), 200)
                
                if fit.get('log_transform', False):
                    # Distribution was fitted to log-transformed data
                    # Transform x_range to log scale, get PDF, then transform back
                    log_x_range = np.log1p(x_range)
                    log_pdf = distribution.pdf(log_x_range, *params)
                    # Apply Jacobian for transformation: d/dx log(1+x) = 1/(1+x)
                    fitted_pdf = log_pdf / (1 + x_range)
                    
                elif fit.get('logit_transform', False):
                    # Distribution was fitted to logit-transformed data
                    eps = 1e-6
                    clipped_x = np.clip(x_range, eps, 1-eps)
                    logit_x_range = np.log(clipped_x/(1-clipped_x))
                    logit_pdf = distribution.pdf(logit_x_range, *params)
                    # Apply Jacobian for logit: d/dx logit(x) = 1/(x(1-x))
                    fitted_pdf = logit_pdf / (clipped_x * (1 - clipped_x))
                    
                else:
                    # No transform, direct PDF
                    fitted_pdf = distribution.pdf(x_range, *params)
                
                # Only plot if PDF is valid
                if np.all(np.isfinite(fitted_pdf)) and np.any(fitted_pdf > 0):
                    ax.plot(x_range, fitted_pdf, 'r-', linewidth=2, alpha=0.8,
                        label=f'Fitted {distribution.name}')
                else:
                    ax.text(0.7, 0.5, 'Fit failed', transform=ax.transAxes, 
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
                    
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
        
        # Add marginal fit info if available
        if hasattr(self, 'marginal_fits') and param in self.marginal_fits:
            fit = self.marginal_fits[param]
            stats_text += f'\nFit: {fit["distribution"].name} (p={fit["ks_p"]:.3f})'
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=8, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    def _create_summary_stats_plot(self, output_folder, available_params, samples):
        """Create a summary statistics comparison plot"""
        
        # Calculate summary statistics
        stats_data = []
        for param in available_params:
            orig_data = self.numeric_data[param].dropna()
            samp_data = samples[param].dropna()
            
            # KS test for distribution comparison
            ks_stat, ks_pvalue = kstest(samp_data, lambda x: stats.percentileofscore(orig_data, x)/100)
            
            # Add marginal fit info if available
            marginal_dist = 'Unknown'
            marginal_ks_p = np.nan
            if hasattr(self, 'marginal_fits') and param in self.marginal_fits:
                fit = self.marginal_fits[param]
                marginal_dist = fit['distribution'].name
                marginal_ks_p = fit['ks_p']
            
            stats_data.append({
                'Parameter': param.replace('_', ' ').title(),
                'Original_Mean': np.mean(orig_data),
                'Sampled_Mean': np.mean(samp_data),
                'Original_Std': np.std(orig_data),
                'Sampled_Std': np.std(samp_data),
                'KS_Statistic': ks_stat,
                'KS_P_Value': ks_pvalue,
                'Marginal_Distribution': marginal_dist,
                'Marginal_KS_P': marginal_ks_p
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Mean comparison
        ax1.scatter(stats_df['Original_Mean'], stats_df['Sampled_Mean'], alpha=0.7)
        ax1.plot([stats_df['Original_Mean'].min(), stats_df['Original_Mean'].max()], 
                [stats_df['Original_Mean'].min(), stats_df['Original_Mean'].max()], 
                'r--', alpha=0.8)
        ax1.set_xlabel('Original Mean')
        ax1.set_ylabel('Sampled Mean')
        ax1.set_title('Mean Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Std comparison
        ax2.scatter(stats_df['Original_Std'], stats_df['Sampled_Std'], alpha=0.7)
        ax2.plot([stats_df['Original_Std'].min(), stats_df['Original_Std'].max()], 
                [stats_df['Original_Std'].min(), stats_df['Original_Std'].max()], 
                'r--', alpha=0.8)
        ax2.set_xlabel('Original Std')
        ax2.set_ylabel('Sampled Std')
        ax2.set_title('Standard Deviation Comparison')
        ax2.grid(True, alpha=0.3)
        
        # KS statistics comparison (sampling vs marginal fit)
        bars = ax3.bar(range(len(stats_df)), stats_df['KS_P_Value'], alpha=0.7, label='Sampling KS')
        
        # Add marginal fit KS p-values if available
        marginal_p_values = stats_df['Marginal_KS_P'].dropna()
        if not marginal_p_values.empty:
            marginal_indices = stats_df['Marginal_KS_P'].dropna().index
            ax3.bar([i + 0.4 for i in marginal_indices], marginal_p_values, 
                   width=0.4, alpha=0.7, label='Marginal Fit KS', color='orange')
        
        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='p=0.05')
        ax3.set_xlabel('Parameter Index')
        ax3.set_ylabel('KS P-Value')
        ax3.set_title('KS Test P-Values')
        ax3.legend()
        ax3.set_xticks(range(len(stats_df)))
        ax3.set_xticklabels([p[:10] + '...' if len(p) > 10 else p 
                            for p in stats_df['Parameter']], rotation=45)
        
        # Distribution types used
        ax4.pie(stats_df['Marginal_Distribution'].value_counts().values,
               labels=stats_df['Marginal_Distribution'].value_counts().index,
               autopct='%1.1f%%', startangle=90)
        ax4.set_title('Marginal Distributions Used')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'enhanced_param_fits_summary.png'), 
                dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics to CSV
        stats_df.to_csv(os.path.join(output_folder, 'enhanced_fit_validation_stats.csv'), index=False)
        print(f"Enhanced validation statistics saved to {output_folder}/enhanced_fit_validation_stats.csv")


def main():
    """Enhanced main function with marginal distribution fitting"""
    print("Starting enhanced mixed parameter fitting with marginal distributions and copulas...")
    
    if len(sys.argv) < 2:
        print("Usage: python enhanced_modelfit_mixed.py <output_folder> [parameter_file] [model_path] [n_samples]")
        sys.exit(1)
    
    output_folder = sys.argv[1]
    parameter_file = sys.argv[2] if len(sys.argv) > 2 else 'none'
    model_path = sys.argv[3] if len(sys.argv) > 3 else 'none'
    n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if parameter_file != 'none':
        # Fit new model
        fitter = MixedPhylogeneticParameterFitter(parameter_file)
        fitter.preprocess_data()
        
        # Plot correlations (enhanced version)
        print("Plotting enhanced parameter correlations...")
        fitter.plot_parameter_correlations(output_folder)
        
        # Fit the mixed distribution with marginals and copulas
        print("Fitting marginal distributions and copula models...")
        fitter.fit_mixed_distribution()
        
        # Validate fit with enhanced plots
        print("Validating fit with enhanced diagnostics...")
        fitter.validate_fit(output_folder)
        
        # Save model
        with open(os.path.join(output_folder, 'model.pkl'), 'wb') as f:
            pickle.dump(fitter, f)
        print(f"Enhanced mixed model saved to {output_folder}/model.pkl")
        
        # Print summary of fitted distributions
        if hasattr(fitter, 'marginal_fits'):
            print("\nMarginal Distribution Summary:")
            for param, fit in fitter.marginal_fits.items():
                print(f"  {param}: {fit['distribution'].name} (KS p-value: {fit['ks_p']:.4f})")
        
        if hasattr(fitter, 'copula_models'):
            print(f"\nCopula Models fitted for {len(fitter.copula_models)} birth-death categories")
        
    elif model_path != 'none':
        # Load existing model and generate samples
        with open(model_path, 'rb') as f:
            fitter = pickle.load(f)
        
        # Generate samples
        print(f"Generating {n_samples} parameter sets using enhanced model...")
        simulation_params = fitter.export_for_simulation(n_samples)
        
        if simulation_params is not None:
            print(f"Generated {len(simulation_params)} parameter sets")
            
            # Verify one-hot encoding
            one_hot_cols = [c for c in simulation_params.columns if c.startswith('best_B') and not c.startswith('best_BD')]
            if one_hot_cols:
                row_sums = simulation_params[one_hot_cols].sum(axis=1)
                print(f"One-hot encoding check: {(row_sums == 1).all()} (all rows sum to 1)")
                print(f"Row sums range: {row_sums.min()} to {row_sums.max()}")
            
            # Save results
            simulation_params.to_csv(os.path.join(output_folder, 'enhanced_mixed_phylo_parameters.csv'), index=False)
            print(f"Parameters saved to {output_folder}/enhanced_mixed_phylo_parameters.csv")
            
            # Print summary of used distributions
            if hasattr(fitter, 'marginal_fits'):
                dist_types = [fit['distribution'].name for fit in fitter.marginal_fits.values()]
                dist_counts = pd.Series(dist_types).value_counts()
                print(f"\nDistributions used in sampling:")
                for dist, count in dist_counts.items():
                    print(f"  {dist}: {count} parameters")


if __name__ == "__main__":
    main()