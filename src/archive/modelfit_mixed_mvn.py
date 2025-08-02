#!/usr/bin/env python3
"""
Enhanced Phylogenetic Parameter Distribution Fitting with Mixed Variable Support

This version properly handles the mixed continuous/categorical nature of phylogenetic parameters.
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

class MixedPhylogeneticParameterFitter:
    """
    Enhanced fitter that properly handles mixed continuous/categorical parameters
    """
    
    def __init__(self, csv_file):
        """Initialize with CSV file containing TreeFam parameters"""
        self.data = pd.read_csv(csv_file)
        self.fitted_distributions = {}
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
        
        # Transform continuous variables
        self._transform_continuous_variables()
        
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
    
    def _transform_continuous_variables(self):
        """Transform continuous variables with appropriate transformations"""
        # Log-transform rate parameters
        rate_params = ['gamma_shape', 'insertion_rate', 'deletion_rate', 
                      'mean_insertion_length', 'mean_deletion_length',
                      'best_BD_speciation_rate', 'best_BD_extinction_rate']
        
        for param in rate_params:
            if param in self.continuous_cols:
                # Add small constant to avoid log(0)
                self.numeric_data[param + '_log'] = np.log(self.numeric_data[param] + 1e-10)
        
        # Logit-transform proportions
        prop_params = ['prop_invariant']
        
        for param in prop_params:
            if param in self.continuous_cols:
                # Logit transform: log(p/(1-p)), handling edge cases
                p = self.numeric_data[param].clip(1e-10, 1-1e-10)
                self.numeric_data[param + '_logit'] = np.log(p / (1 - p))
        
        # Store transformation info for bias correction
        self.transformations = {
            'log_params': [p for p in rate_params if p in self.continuous_cols],
            'logit_params': [p for p in prop_params if p in self.continuous_cols]
        }
    
    def fit_mixed_distribution(self):
        """Fit joint distribution for continuous variables + categorical model"""
        # Get continuous variables (including transformed ones)
        continuous_vars = []
        for param in self.continuous_cols:
            continuous_vars.append(param)
            # Add transformed versions if they exist
            if param + '_log' in self.numeric_data.columns:
                continuous_vars.append(param + '_log')
            if param + '_logit' in self.numeric_data.columns:
                continuous_vars.append(param + '_logit')
        
        # Remove duplicates and filter to existing columns
        continuous_vars = list(set(continuous_vars))
        continuous_vars = [v for v in continuous_vars if v in self.numeric_data.columns]
        
        print(f"Fitting joint distribution for {len(continuous_vars)} continuous variables...")
        
        # Fit separate models for each birth-death model category
        self.conditional_models = {}
        
        for category in self.category_probs.index:
            category_mask = self.numeric_data['bd_model_category'] == category
            category_data = self.numeric_data[category_mask][continuous_vars]
            
            if len(category_data) < 5:  # Skip if too few samples
                print(f"Skipping {category} - insufficient data ({len(category_data)} samples)")
                continue
            
            print(f"Fitting model for {category} ({len(category_data)} samples)")
            
            # Fit multivariate normal for this category
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(category_data.dropna())
            
            if len(scaled_data) < 2:
                continue
                
            mean = np.mean(scaled_data, axis=0)
            cov = np.cov(scaled_data.T)
            
            # Add small regularization to diagonal if needed
            if np.linalg.det(cov) < 1e-10:
                cov += np.eye(len(mean)) * 1e-6
            
            self.conditional_models[category] = {
                'mean': mean,
                'cov': cov,
                'scaler': scaler,
                'param_names': continuous_vars
            }
        
        print(f"Successfully fitted models for {len(self.conditional_models)} categories")
        return self.conditional_models
    
    def sample_mixed_parameters(self, n_samples=100, min_n_sequences_tips=20, n_std=1.5, 
                               bias_correction=True):
        """
        Enhanced sampling with bias correction and looser bounds
        
        Parameters:
        - n_std: Increased default to 2.5 standard deviations (was 1.5)
        - bias_correction: Apply bias correction for log-transformed variables
        """
        if not hasattr(self, 'conditional_models') or not self.conditional_models:
            print("No conditional models fitted. Please fit mixed distribution first.")
            return None
        
        samples = []
        
        # Calculate bias correction factors for log-transformed parameters
        bias_corrections = {}
        if bias_correction and hasattr(self, 'transformations'):
            for param in self.transformations.get('log_params', []):
                if param in self.numeric_data.columns:
                    # Calculate empirical bias correction
                    original_mean = self.numeric_data[param].mean()
                    log_values = np.log(self.numeric_data[param] + 1e-10)
                    naive_back_transform = np.exp(log_values.mean())
                    bias_corrections[param] = original_mean / naive_back_transform
        
        for _ in range(n_samples):
            # Sample categorical variable (birth-death model)
            bd_model = np.random.choice(self.category_probs.index, p=self.category_probs.values)
            
            # Skip if no model available for this category
            if bd_model not in self.conditional_models:
                # Fall back to most common category
                bd_model = self.category_probs.index[0]
                if bd_model not in self.conditional_models:
                    continue
            
            model = self.conditional_models[bd_model]
            
            # Sample continuous variables conditioned on categorical choice
            max_attempts = 500  # Increased attempts
            valid_sample = False
            
            for attempt in range(max_attempts):
                try:
                    # Generate sample from multivariate normal
                    scaled_sample = np.random.multivariate_normal(model['mean'], model['cov'])
                    continuous_sample = model['scaler'].inverse_transform(scaled_sample.reshape(1, -1))[0]
                    
                    # Create sample dictionary
                    sample_dict = dict(zip(model['param_names'], continuous_sample))
                    
                    # Add categorical variable
                    sample_dict['bd_model_category'] = bd_model
                    
                    # Convert to one-hot encoding
                    for col in self.categorical_cols:
                        sample_dict[col] = 1 if col == f'best_{bd_model}' else 0
                    
                    # Apply looser constraints - only essential ones
                    constraints_passed = True
                    
                    # Essential constraint: minimum sequences
                    if ('n_sequences_tips' in sample_dict and 
                        sample_dict['n_sequences_tips'] <= min_n_sequences_tips):
                        constraints_passed = False
                    
                    '''
                    # Essential constraint: non-negative rates (but allow wider range)
                    rate_params = ['insertion_rate', 'deletion_rate', 'mean_insertion_length', 'mean_deletion_length']
                    for param in rate_params:
                        if param in sample_dict and sample_dict[param] < 0:
                            constraints_passed = False
                            break
                    '''
                    
                    if constraints_passed:
                        for param in model['param_names']:
                            if param in sample_dict and param in self.numeric_data.columns:
                                orig_data = self.numeric_data[param]
                                value = sample_dict[param]
                                mean, std = orig_data.mean(), orig_data.std()
                                
                                # Only reject extreme outliers (1.5+ standard deviations)
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
            
            # Apply bias correction to log-transformed parameters
            if bias_correction and bias_corrections:
                print("Applying bias correction to log-transformed parameters...")
                for param, correction_factor in bias_corrections.items():
                    if param + '_log' in result_df.columns:
                        # Apply correction in log space
                        result_df[param + '_log'] += np.log(correction_factor)
                        print(f"  {param}: correction factor = {correction_factor:.3f}")
            
            print(f"Generated {len(result_df)} valid samples with {n_std}-std bounds")
            print("Bias correction applied:", bias_correction)
            return result_df
        else:
            print("Failed to generate any valid samples")
            return None
    
    def export_for_simulation(self, n_samples=100, bias_correction=True):
        """Export mixed parameters for simulation with bias correction"""
        samples = self.sample_mixed_parameters(n_samples, bias_correction=bias_correction)
        
        if samples is None:
            return None
        
        # Convert back from transformed parameters
        export_data = samples.copy()
        
        # Inverse transforms with bias correction
        for col in list(export_data.columns):
            # Inverse logit transform
            if col.endswith('_logit'):
                original_col = col.replace('_logit', '')
                if original_col in self.continuous_cols:
                    export_data[original_col] = np.exp(export_data[col]) / (1 + np.exp(export_data[col]))
                    export_data = export_data.drop(columns=[col])
            
            # Inverse log transform
            elif col.endswith('_log'):
                original_col = col.replace('_log', '')
                if original_col in self.continuous_cols:
                    export_data[original_col] = np.exp(export_data[col])
                    export_data = export_data.drop(columns=[col])
        
        # Ensure proper data types and constraints
        if 'n_sequences_tips' in export_data.columns:
            export_data['n_sequences_tips'] = export_data['n_sequences_tips'].astype(int)
        if 'alignment_length' in export_data.columns:
            export_data['alignment_length'] = export_data['alignment_length'].astype(int)
        
        # Ensure non-negative values for rates (but don't over-constrain)
        rate_cols = ['prop_invariant', 'insertion_rate', 'deletion_rate', 
                    'mean_insertion_length', 'mean_deletion_length']
        for col in rate_cols:
            if col in export_data.columns:
                export_data[col] = np.maximum(export_data[col], 1e-10)  # Allow very small positive values
        
        # Add combined indel rate
        if 'insertion_rate' in export_data.columns and 'deletion_rate' in export_data.columns:
            export_data['indel_rate'] = export_data['insertion_rate'] + export_data['deletion_rate']
        
        # Remove helper columns
        if 'bd_model_category' in export_data.columns:
            export_data = export_data.drop(columns=['bd_model_category'])
        
        return export_data
    
    # ============================================================================
    # COMPATIBILITY METHODS FOR OLD PLOTTING FUNCTIONS
    # ============================================================================
    
    def sample_parameters(self, n_samples=100, param_group='key_parameters', **kwargs):
        """Compatibility wrapper for old plotting functions"""
        if param_group == 'key_parameters':
            return self.sample_mixed_parameters(n_samples, **kwargs)
        else:
            print(f"Warning: param_group '{param_group}' not supported in mixed fitter")
            return self.sample_mixed_parameters(n_samples, **kwargs)
    
    def _update_parameter_groups_for_compatibility(self):
        """Update parameter groups to match old structure for plotting"""
        all_params = self.continuous_cols + self.categorical_cols
        self.parameter_groups['key_parameters'] = all_params
    
    def validate_fit(self, output_folder, param_group='key_parameters'):
        """Enhanced validation compatible with old plotting methods"""
        # Update parameter groups for compatibility
        self._update_parameter_groups_for_compatibility()
        
        if not hasattr(self, 'conditional_models') or not self.conditional_models:
            print("No conditional models to validate")
            return
        
        # Generate samples using mixed approach
        samples = self.sample_mixed_parameters(n_samples=1000)
        
        if samples is None:
            return
        
        # Use the original plotting methods
        params = self.parameter_groups[param_group]
        available_params = [p for p in params if p in self.numeric_data.columns and p in samples.columns]
        
        # Group parameters by category for better organization
        param_categories = self._categorize_parameters(available_params)
        
        # Create plots (reuse old methods)
        if len(available_params) > 12:
            self._create_categorized_plots(output_folder, param_categories, samples)
        else:
            self._create_single_organized_plot(output_folder, available_params, samples)
        
        # Create summary statistics plot
        self._create_summary_stats_plot(output_folder, available_params, samples)
        
        # Add mixed-model specific validation
        self._validate_mixed_model_specific(output_folder, samples)
    
    def _validate_mixed_model_specific(self, output_folder, samples):
        """Additional validation specific to mixed model approach"""
        
        # Validate one-hot encoding
        one_hot_cols = [c for c in samples.columns if c.startswith('best_B') and not c.startswith('best_BD')]
        if one_hot_cols:
            row_sums = samples[one_hot_cols].sum(axis=1)
            all_valid = (row_sums == 1).all()
            
            plt.figure(figsize=(10, 6))
            
            # Plot 1: Row sums distribution
            plt.subplot(1, 2, 1)
            plt.hist(row_sums, bins=20, alpha=0.7, edgecolor='black')
            plt.axvline(x=1, color='red', linestyle='--', label='Expected sum = 1')
            plt.xlabel('Sum of One-Hot Columns')
            plt.ylabel('Frequency')
            plt.title('One-Hot Encoding Validation')
            plt.legend()
            
            # Plot 2: Model distribution comparison
            plt.subplot(1, 2, 2)
            
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
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'mixed_model_validation.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"One-hot encoding validation: {'PASSED' if all_valid else 'FAILED'}")
            print(f"Row sums range: {row_sums.min():.3f} to {row_sums.max():.3f}")
    
    def plot_parameter_correlations(self, output_folder):
        """Enhanced correlation plot for mixed model"""
        # Use only continuous parameters for correlation
        continuous_data = self.numeric_data[self.continuous_cols]
        
        corr_data = continuous_data.corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Continuous Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'parameter_correlations.png'), dpi=300)
        plt.close()
        
        # Also plot categorical variable distribution
        if hasattr(self, 'category_probs'):
            plt.figure(figsize=(10, 6))
            self.category_probs.plot(kind='bar', alpha=0.7)
            plt.title('Birth-Death Model Distribution')
            plt.ylabel('Probability')
            plt.xlabel('Model')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, 'bd_model_distribution.png'), dpi=300)
            plt.close()
    
    # ============================================================================
    # ORIGINAL PLOTTING METHODS (copied from original code for compatibility)
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


def main():
    """Enhanced main function"""
    print("Starting mixed parameter fitting workflow...")
    
    if len(sys.argv) < 2:
        print("Usage: python modelfit_mixed.py <output_folder> [parameter_file] [model_path] [n_samples]")
        sys.exit(1)
    
    output_folder = sys.argv[1]
    parameter_file = sys.argv[2] if len(sys.argv) > 2 else 'none'
    model_path = sys.argv[3] if len(sys.argv) > 3 else 'none'
    n_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if parameter_file != 'none':
        # Fit new model
        fitter = MixedPhylogeneticParameterFitter(parameter_file)
        fitter.preprocess_data()
        
        # Plot correlations (like original script)
        print("Plotting parameter correlations...")
        fitter.plot_parameter_correlations(output_folder)
        
        # Fit the mixed distribution
        fitter.fit_mixed_distribution()
        
        # Validate fit with plots (like original script)
        print("Validating fit...")
        fitter.validate_fit(output_folder)
        
        # Save model
        with open(os.path.join(output_folder, 'model.pkl'), 'wb') as f:
            pickle.dump(fitter, f)
        print(f"Mixed model saved to {output_folder}/model.pkl")
        
    elif model_path != 'none':
        # Load existing model and generate samples
        with open(model_path, 'rb') as f:
            fitter = pickle.load(f)
        
        # Generate samples
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
            simulation_params.to_csv(os.path.join(output_folder, 'mixed_phylo_parameters.csv'), index=False)
            print(f"Parameters saved to {output_folder}/mixed_phylo_parameters.csv")


if __name__ == "__main__":
    main()