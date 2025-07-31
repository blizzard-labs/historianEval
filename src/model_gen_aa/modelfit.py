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
            'key_parameters' : ['n_sequences_tips', 'alignment_length', 'crown_age',
                                'gamma_shape', 'prop_invariant',
                                'insertion_rate', 'deletion_rate',
                                'mean_insertion_length', 'mean_deletion_length',
                                'normalized_colless_index', 'gamma',
                                'best_BD_speciation_rate', 'best_BD_extinction_rate',
                                'best_BD_speciation_alpha', 'best_BD_extinction_alpha',
                                'best_BCSTDCST', 'best_BEXPDCST', 'best_BLINDCST', 'best_BCSTDEXP', 'best_BEXPDEXP',
                                'best_BLINDEXP', 'best_BCSTDLIN', 'best_BEXPDLIN', 'best_BLINDLIN']
        }

    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        # Remove non-numeric columns and handle missing values
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.numeric_data = self.data[numeric_cols].copy()
        
        # Handle missing values
        self.numeric_data = self.numeric_data.dropna()
        
        # Log-transform rate parameters (they're often log-normal)
        rate_params = ['gamma_shape', 'insertion_rate', 'deletion_rate', 'mean_insertion_length', 'mean_deletion_length',
                       'best_BD_speciation_rate', 'best_BD_extinction_rate']
        
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
    
    def sample_parameters(self, n_samples=100, param_group='key_parameters', 
                                               min_n_sequences_tips=20, n_std=1.5):
        """
        Enhanced rejection sampling with parameter-specific bounds and replacement strategy
        Excludes certain parameters from bounds constraints (best_B* model selection parameters)
        
        Parameters:
        - n_std: Number of standard deviations to use as bounds (default 1.5)
        """
        if self.joint_model is None:
            print("No joint model fitted. Please fit a joint distribution first.")
            return None
        
        params = self.parameter_groups[param_group]
        available_params = [p for p in params if p in self.numeric_data.columns]
        
        # Define parameters to exclude from bounds constraints (model selection indicators)
        unrestricted_params = {
            'best_BCSTDCST', 'best_BEXPDCST', 'best_BLINDCST',
            'best_BCSTDEXP', 'best_BEXPDEXP', 'best_BLINDEXP', 
            'best_BCSTDLIN', 'best_BEXPDLIN', 'best_BLINDLIN',
            'best_BD_speciation_alpha', 'best_BD_extinction_alpha'
        }
        
        # Calculate bounds for each parameter (excluding unrestricted ones)
        param_bounds = {}
        restricted_params = []
        
        for param in available_params:
            if param in unrestricted_params:
                continue  # Skip bounds calculation for unrestricted parameters
                
            data = self.numeric_data[param].dropna()
            mean = np.mean(data)
            std = np.std(data)
            param_bounds[param] = {
                'mean': mean,
                'std': std,
                'lower': mean - n_std * std,
                'upper': mean + n_std * std
            }
            restricted_params.append(param)
        
        print(f"Applying {n_std}-std bounds to {len(restricted_params)} parameters")
        print(f"Unrestricted parameters: {[p for p in available_params if p in unrestricted_params]}")
        
        # Generate samples in batches with rejection sampling
        accepted_samples = []
        total_attempts = 0
        batch_size = max(100, n_samples * 2)
        
        while len(accepted_samples) < n_samples and total_attempts < 100000:
            # Generate a batch of samples
            if COPULAS_AVAILABLE and hasattr(self.joint_model, 'sample'):
                batch_samples = self.joint_model.sample(batch_size)
            else:
                if self.joint_model['type'] == 'multivariate_normal':
                    samples_scaled = np.random.multivariate_normal(
                        self.joint_model['mean'], 
                        self.joint_model['cov'], 
                        batch_size
                    )
                    batch_samples = self.joint_model['scaler'].inverse_transform(samples_scaled)
                    batch_samples = pd.DataFrame(batch_samples, columns=self.joint_model['param_names'])
            
            # Apply rejection criteria
            for idx in range(len(batch_samples)):
                sample = batch_samples.iloc[idx]
                accept_sample = True
                
                # Check bounds only for restricted parameters
                for param in restricted_params:
                    if param in sample.index:
                        value = sample[param]
                        if not (param_bounds[param]['lower'] <= value <= param_bounds[param]['upper']):
                            accept_sample = False
                            break
                
                # Additional constraint for n_sequences_tips
                if (accept_sample and 'n_sequences_tips' in sample.index and 
                    sample['n_sequences_tips'] <= min_n_sequences_tips):
                    accept_sample = False
                
                if accept_sample:
                    accepted_samples.append(sample)
                    if len(accepted_samples) >= n_samples:
                        break
            
            total_attempts += batch_size
        
        if accepted_samples:
            result = pd.DataFrame(accepted_samples)
            acceptance_rate = len(accepted_samples) / total_attempts * 100
            print(f"Generated {len(result)} samples with {acceptance_rate:.1f}% acceptance rate")
            print(f"Samples constrained to within {n_std} standard deviation(s) for {len(restricted_params)} parameters")
            print(f"{len(unrestricted_params & set(available_params))} parameters left unrestricted")
            return result.iloc[:n_samples]  # Return exactly n_samples
        else:
            print(f"Failed to generate sufficient samples within {n_std}-std bounds")
            return None
    
    def validate_fit(self, output_folder, param_group='key_parameters'):
        """Validate the fitted joint distribution with organized subplots"""
        if self.joint_model is None:
            print("No joint model to validate")
            return
        
        # Generate samples
        samples = self.sample_parameters(n_samples=1000, param_group=param_group)
        
        if samples is None:
            return
        
        # Compare distributions
        params = self.parameter_groups[param_group]
        available_params = [p for p in params if p in self.numeric_data.columns and p in samples.columns]
        
        # Group parameters by category for better organization
        param_categories = self._categorize_parameters(available_params)
        
        # Create separate plots for each category or one large organized plot
        if len(available_params) > 12:
            self._create_categorized_plots(output_folder, param_categories, samples)
        else:
            self._create_single_organized_plot(output_folder, available_params, samples)
        
        # Create summary statistics plot
        self._create_summary_stats_plot(output_folder, available_params, samples)

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
        """Plot comparison for a single parameter"""
        orig_data = self.numeric_data[param].dropna()
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

    def _create_summary_stats_plot(self, output_folder, available_params, samples):
        """Create a summary statistics comparison plot"""
        
        # Calculate summary statistics
        stats_data = []
        for param in available_params:
            orig_data = self.numeric_data[param].dropna()
            samp_data = samples[param].dropna()
            
            # KS test for distribution comparison
            ks_stat, ks_pvalue = kstest(samp_data, lambda x: stats.percentileofscore(orig_data, x)/100)
            
            stats_data.append({
                'Parameter': param.replace('_', ' ').title(),
                'Original_Mean': np.mean(orig_data),
                'Sampled_Mean': np.mean(samp_data),
                'Original_Std': np.std(orig_data),
                'Sampled_Std': np.std(samp_data),
                'KS_Statistic': ks_stat,
                'KS_P_Value': ks_pvalue
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
        
        # KS statistics
        bars = ax3.bar(range(len(stats_df)), stats_df['KS_Statistic'])
        ax3.set_xlabel('Parameter Index')
        ax3.set_ylabel('KS Statistic')
        ax3.set_title('Kolmogorov-Smirnov Test Statistics')
        ax3.set_xticks(range(len(stats_df)))
        ax3.set_xticklabels([p[:10] + '...' if len(p) > 10 else p 
                            for p in stats_df['Parameter']], rotation=45)
        
        # Color bars by p-value
        for i, (bar, pval) in enumerate(zip(bars, stats_df['KS_P_Value'])):
            if pval < 0.01:
                bar.set_color('red')
            elif pval < 0.05:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # P-values
        ax4.bar(range(len(stats_df)), -np.log10(stats_df['KS_P_Value'] + 1e-10))
        ax4.set_xlabel('Parameter Index')
        ax4.set_ylabel('-log10(p-value)')
        ax4.set_title('KS Test P-Values (-log10 scale)')
        ax4.axhline(y=-np.log10(0.05), color='r', linestyle='--', alpha=0.7, label='p=0.05')
        ax4.set_xticks(range(len(stats_df)))
        ax4.set_xticklabels([p[:10] + '...' if len(p) > 10 else p 
                            for p in stats_df['Parameter']], rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'param_fits_summary.png'), 
                dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save statistics to CSV
        stats_df.to_csv(os.path.join(output_folder, 'fit_validation_stats.csv'), index=False)
        print(f"Validation statistics saved to {output_folder}/fit_validation_stats.csv")
    
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
            if col.endswith('_log'):
                original_col = col.replace('_log', '')
                export_data[original_col] = np.exp(export_data[col])
                export_data = export_data.drop(columns=[col])
                
            # Integer values
            if col in ['n_sequences_tips', 'alignment_length']:
                export_data[col] = export_data[col].astype(int)
            
            if col in ['prop_invariant', 'insertion_rate', 'deletion_rate', 'mean_insertion_length', 'mean_deletion_length', 'normalized_colless_index', 'best_BD_speciation_alpha', 'best_BD_extinction_alpha']:
                export_data[col] = np.maximum(export_data[col], 0)
            
        one_hot_cols = [c for c in export_data.columns if c.startswith('best_B') and not c.startswith('best_BD')]
        max_idx = export_data[one_hot_cols].idxmax(axis=1)
        export_data[one_hot_cols] = 0
        
        for i, colname in enumerate(max_idx):
            export_data.at[i, colname] = 1

        export_data = self.enforce_onehot_constraint(export_data)
        export_data['indel_rate'] = export_data['insertion_rate'] + export_data['deletion_rate']
        
        return export_data
    
    
    def enforce_onehot_constraint(self, export_data):
        """Ensure exactly one birth-death model is selected per row"""
        one_hot_cols = [c for c in export_data.columns if c.startswith('best_B') and not c.startswith('best_BD')]
        
        if not one_hot_cols:
            return export_data
        
        # Get model probabilities from training data
        original_probs = self.numeric_data[one_hot_cols].mean()
        
        for idx in export_data.index:
            row_values = export_data.loc[idx, one_hot_cols]
            
            if row_values.sum() == 0:
                # All zeros - sample according to original distribution
                chosen_model = np.random.choice(one_hot_cols, p=original_probs.values)
                export_data.loc[idx, one_hot_cols] = 0
                export_data.loc[idx, chosen_model] = 1
            elif row_values.sum() != 1:
                # Multiple selections or non-binary values - use softmax
                probabilities = np.exp(row_values) / np.sum(np.exp(row_values))
                chosen_model = np.random.choice(one_hot_cols, p=probabilities)
                export_data.loc[idx, one_hot_cols] = 0
                export_data.loc[idx, chosen_model] = 1
        
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
    print(f'Loaded model: {model_path}')
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
        #print("Fitting joint distribution for amino acid frequencies...")
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