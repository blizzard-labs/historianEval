#!/usr/bin/env python3
"""
GTR+IG Parameter Extraction Script using ModelTest-NG
Indel Parameters from Basic Estimation
Tree Topology Parameters from TreePar (R)

This script processes a folder of FASTA multiple sequence alignments and extracts:
1. GTR+IG substitution parameters (6 substitution rates + gamma shape + proportion invariant)
2. Indel parameters (insertion/deletion rates and length distributions)

Requirements:
- modeltest-ng (installed and in PATH)
- biopython
- numpy
- pandas
- matplotlib
- seaborn
- scipy
"""

import os
import sys
import glob
import subprocess
import tempfile
import shutil
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from scipy import stats
from Bio import AlignIO, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import warnings
warnings.filterwarnings('ignore')

class GTRIndelParameterExtractor:
    def __init__(self, input_folder, output_folder="results", modeltest_path="modeltest-ng"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.modeltest_path = modeltest_path
        self.results = []
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # Create temp directory for ModelTest-NG runs
        self.temp_dir = os.path.join(output_folder, "temp_modeltest")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # GTR substitution types (6 parameters)
        self.substitution_types = ['A<->C', 'A<->G', 'A<->T', 'C<->G', 'C<->T', 'G<->T']
        
        # Check if ModelTest-NG is available
        self.check_modeltest_ng()
    
    
    def read_alignment(self, filepath):
        """Read FASTA alignment file"""
        try:
            alignment = AlignIO.read(filepath, "fasta")
            return alignment
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None
    
    
    def check_modeltest_ng(self):
        """Check if ModelTest-NG is available"""
        try:
            result = subprocess.run([self.modeltest_path, "--version"], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"Found ModelTest-NG: {result.stdout.strip()}")
            else:
                print("Warning: ModelTest-NG not found or not working properly")
                print("Falling back to basic parameter estimation")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"Warning: Could not run ModelTest-NG ({e})")
            print("Falling back to basic parameter estimation")
    
    def run_modeltest_ng(self, alignment_file):
        """Run ModelTest-NG on alignment file"""
        base_name = os.path.splitext(os.path.basename(alignment_file))[0]
        output_prefix = os.path.join(self.temp_dir, base_name)
        
        cmd = [
            self.modeltest_path,
            "-i", alignment_file,
            "-o", output_prefix,
            "-m", "GTR",  # Force GTR+I+G model
            "-t", "ml",
            "-p", "5",
            "--force"  # Overwrite existing files    
        ]
        
        # "-q"  # Quiet mode
                
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return self.parse_modeltest_output(output_prefix)
            else:
                print(f"ModelTest-NG failed for {alignment_file}")
                print(f"Error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"ModelTest-NG timed out for {alignment_file}")
            return None
        except Exception as e:
            print(f"Error running ModelTest-NG on {alignment_file}: {e}")
            return None
    
    
    
    def parse_modeltest_output(self, output_prefix):
        """Parse ModelTest-NG output files"""
        log_file = output_prefix + ".log"
        
        if not os.path.exists(log_file):
            return None
        
        params = {}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Find the best model section - look for GTR+I or GTR+I+G4
            best_model_match = re.search(r'Best model according to AIC.*?Model:\s+(GTR\+[IG4+]+)', content, re.DOTALL)
            if best_model_match:
                params['best_model'] = best_model_match.group(1)
            
            # Extract log likelihood from best model section
            lnl_pattern = r'lnL:\s+([-\d\.]+)'
            lnl_match = re.search(lnl_pattern, content)
            if lnl_match:
                params['log_likelihood'] = float(lnl_match.group(1))
            
            # Extract base frequencies - look for pattern like "0.4235 0.1520 0.2021 0.2224"
            freq_pattern = r'Frequencies:\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)'
            freq_match = re.search(freq_pattern, content)
            if freq_match:
                params['freq_A'] = float(freq_match.group(1))
                params['freq_C'] = float(freq_match.group(2))
                params['freq_G'] = float(freq_match.group(3))
                params['freq_T'] = float(freq_match.group(4))
            
            # Extract substitution rates - pattern like "0.8709 0.4190 0.6092 1.2658 0.9465 1.0000"
            # Order is typically: A-C, A-G, A-T, C-G, C-T, G-T
            rates_pattern = r'Subst\. Rates:\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)'
            rates_match = re.search(rates_pattern, content)
            if rates_match:
                params['rate_AC'] = float(rates_match.group(1))
                params['rate_AG'] = float(rates_match.group(2))
                params['rate_AT'] = float(rates_match.group(3))
                params['rate_CG'] = float(rates_match.group(4))
                params['rate_CT'] = float(rates_match.group(5))
                params['rate_GT'] = float(rates_match.group(6))
            
            # Extract proportion of invariant sites
            pinv_pattern = r'P\.Inv:\s+([\d\.]+)'
            pinv_match = re.search(pinv_pattern, content)
            if pinv_match:
                params['prop_invariant'] = float(pinv_match.group(1))
            
            # Extract gamma shape parameter from model averaged estimates
            # Look for "Alpha:" followed by a number
            alpha_pattern = r'Alpha:\s+([\d\.]+)'
            alpha_match = re.search(alpha_pattern, content)
            if alpha_match:
                params['gamma_shape'] = float(alpha_match.group(0).split(' ')[-1].strip()) #some changes
            
            # Also try to get gamma shape from "Gamma shape:" line (might be "-" if not applicable)
            gamma_shape_pattern = r'Gamma shape:\s+([\d\.\-]+)'
            gamma_shape_match = re.search(gamma_shape_pattern, content)
            if gamma_shape_match and gamma_shape_match.group(1) != '-':
                # Only override if we got a numeric value
                try:
                    params['gamma_shape'] = float(gamma_shape_match.group(1))
                except ValueError:
                    pass  # Keep the previous value if this is "-"
            
            # Extract AIC score
            aic_pattern = r'Score:\s+([\d\.]+)'
            aic_match = re.search(aic_pattern, content)
            if aic_match:
                params['aic_score'] = float(aic_match.group(1))
            
            # Extract model weight
            weight_pattern = r'Weight:\s+([\d\.]+)'
            weight_match = re.search(weight_pattern, content)
            if weight_match:
                params['model_weight'] = float(weight_match.group(1))
            
            # Try to extract model-averaged P.Inv if available
            model_avg_pinv_pattern = r'Model averaged estimates.*?P\.Inv:\s+([\d\.]+)'
            model_avg_pinv_match = re.search(model_avg_pinv_pattern, content, re.DOTALL)
            if model_avg_pinv_match:
                params['model_avg_p_inv'] = float(model_avg_pinv_match.group(0).split()[-1])
            
            # Try to extract model-averaged frequencies if available
            model_avg_freq_pattern = r'Model averaged estimates.*?Frequencies:\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)'
            model_avg_freq_match = re.search(model_avg_freq_pattern, content, re.DOTALL)
            if model_avg_freq_match:
                params['model_avg_freq_A'] = float(model_avg_freq_match.group(1))
                params['model_avg_freq_C'] = float(model_avg_freq_match.group(2))
                params['model_avg_freq_G'] = float(model_avg_freq_match.group(3))
                params['model_avg_freq_T'] = float(model_avg_freq_match.group(4))
            
            return params if params else None
            
        except Exception as e:
            print(f"Error extracting GTR parameters: {e}")
            return None
    
    
    def estimate_gtr_parameters_modeltest(self, alignment_file):
        """Estimate GTR parameters using ModelTest-NG"""
            
        params = self.run_modeltest_ng(alignment_file) 
        
        return params
    
        """Calculate base frequencies from alignment"""
        base_counts = Counter()
        total_bases = 0
        
        for record in alignment:
            sequence = str(record.seq).upper()
            for base in sequence:
                if base in 'ACGT':
                    base_counts[base] += 1
                    total_bases += 1
        
        if total_bases == 0:
            return {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}
        
        return {base: count/total_bases for base, count in base_counts.items()}
    
    
    def count_substitutions(self, alignment):
        """Count substitution types between sequences"""
        sub_counts = defaultdict(int)
        total_comparisons = 0
        
        sequences = [str(record.seq).upper() for record in alignment]
        n_seqs = len(sequences)
        
        # Pairwise comparisons
        for i in range(n_seqs):
            for j in range(i+1, n_seqs):
                seq1, seq2 = sequences[i], sequences[j]
                
                for k in range(len(seq1)):
                    base1, base2 = seq1[k], seq2[k]
                    
                    if base1 in 'ACGT' and base2 in 'ACGT' and base1 != base2:
                        # Normalize substitution (A->C same as C->A)
                        sub_type = tuple(sorted([base1, base2]))
                        sub_counts[sub_type] += 1
                        total_comparisons += 1
        
        return sub_counts, total_comparisons
    
    
    def calculate_base_frequencies(self, alignment):
        """Calculate base frequencies from alignment"""
        base_counts = Counter()
        total_bases = 0
        
        for record in alignment:
            sequence = str(record.seq).upper()
            for base in sequence:
                if base in 'ACGT':
                    base_counts[base] += 1
                    total_bases += 1
        
        if total_bases == 0:
            return {'A': 0.25, 'C': 0.25, 'G': 0.25, 'T': 0.25}
        
        return {base: count/total_bases for base, count in base_counts.items()}
    
    
    def estimate_gtr_parameters(self, alignment):
        """Estimate GTR substitution parameters (fallback method)"""
        print('USING FALLBACK METHOD....')
        base_freqs = self.calculate_base_frequencies(alignment)
        sub_counts, total_subs = self.count_substitutions(alignment)
        
        if total_subs == 0:
            return None
        
        # Calculate relative rates (normalized to G<->T = 1)
        gtr_params = {}
        substitutions = [('A','C'), ('A','G'), ('A','T'), ('C','G'), ('C','T'), ('G','T')]
        
        for sub in substitutions:
            count = sub_counts.get(sub, 0)
            rate = count / total_subs if total_subs > 0 else 0
            gtr_params[f"{sub[0]}<->{sub[1]}"] = rate
        
        # Normalize rates (set G<->T as reference = 1.0)
        gt_rate = gtr_params.get('G<->T', 1e-10)
        if gt_rate > 0:
            for key in gtr_params:
                gtr_params[key] = gtr_params[key] / gt_rate
        
        return {
            'base_frequencies': base_freqs,
            'substitution_rates': gtr_params,
            'total_substitutions': total_subs
        }
    
    def analyze_indels(self, alignment):
        """Analyze indel patterns in the alignment"""
        indel_stats = {
            'insertion_events': 0,
            'deletion_events': 0,
            'insertion_lengths': [],
            'deletion_lengths': [],
            'total_gaps': 0
        }
        
        sequences = [str(record.seq).upper() for record in alignment]
        n_seqs = len(sequences)
        align_length = len(sequences[0]) if sequences else 0
        
        # Analyze gaps column by column
        for pos in range(align_length):
            column = [seq[pos] if pos < len(seq) else '-' for seq in sequences]
            gap_count = column.count('-')
            indel_stats['total_gaps'] += gap_count
        
        # Find indel events by comparing sequences
        for i, seq in enumerate(sequences):
            gap_runs = []
            in_gap = False
            gap_start = 0
            
            for j, char in enumerate(seq):
                if char == '-':
                    if not in_gap:
                        gap_start = j
                        in_gap = True
                else:
                    if in_gap:
                        gap_length = j - gap_start
                        gap_runs.append(gap_length)
                        in_gap = False
            
            # Handle gap at end of sequence
            if in_gap:
                gap_length = len(seq) - gap_start
                gap_runs.append(gap_length)
            
            # Count as deletions (gaps in this sequence)
            indel_stats['deletion_events'] += len(gap_runs)
            indel_stats['deletion_lengths'].extend(gap_runs)
        
        # Simple insertion estimation (inverse of deletions) #!!! TODO: Change to more accurate method of detecting indel events
        indel_stats['insertion_events'] = indel_stats['deletion_events']
        indel_stats['insertion_lengths'] = indel_stats['deletion_lengths'].copy()
        
        return indel_stats
    
    def estimate_gamma_parameters(self, alignment):
        """Estimate gamma distribution parameters for rate heterogeneity"""
        # Simplified estimation based on site variability
        sequences = [str(record.seq).upper() for record in alignment]
        n_seqs = len(sequences)
        align_length = len(sequences[0]) if sequences else 0
        
        site_variability = []
        
        for pos in range(align_length):
            column = [seq[pos] if pos < len(seq) else 'N' for seq in sequences]
            valid_bases = [base for base in column if base in 'ACGT']
            
            if len(valid_bases) > 1:
                # Calculate Shannon entropy as measure of variability
                base_counts = Counter(valid_bases)
                total = len(valid_bases)
                entropy = -sum((count/total) * np.log2(count/total) for count in base_counts.values())
                site_variability.append(entropy)
        
        if not site_variability:
            return {'gamma_shape': 1.0, 'prop_invariant': 0.0}
        
        # Rough gamma shape estimation
        mean_var = np.mean(site_variability)
        var_var = np.var(site_variability)
        
        if var_var > 0:
            gamma_shape = (mean_var ** 2) / var_var
        else:
            gamma_shape = 1.0
        
        # Proportion of invariant sites
        invariant_sites = sum(1 for v in site_variability if v == 0)
        prop_invariant = invariant_sites / align_length if align_length > 0 else 0
        
        return {
            'gamma_shape': max(0.1, gamma_shape),  # Minimum reasonable value
            'prop_invariant': prop_invariant
        }

    
    def process_alignment(self, filepath):
        """Process a single alignment file"""
        print(f"Processing: {os.path.basename(filepath)}")
        
        alignment = self.read_alignment(filepath)
        if alignment is None or len(alignment) < 2:
            print(f"Skipping {filepath}: Invalid alignment or too few sequences")
            return None
        
        # Try ModelTest-NG first, fallback to basic estimation
        modeltest_params = self.estimate_gtr_parameters_modeltest(filepath) #! not returning parameters
        
        if modeltest_params and len(modeltest_params) > 4:  # Check if we got reasonable parameters
            print(f"  - Using ModelTest-NG parameters")
            gtr_params = {
                'base_frequencies': {
                    'A': modeltest_params.get('freq_A', 0.25),
                    'C': modeltest_params.get('freq_C', 0.25),
                    'G': modeltest_params.get('freq_G', 0.25),
                    'T': modeltest_params.get('freq_T', 0.25)
                },
                'substitution_rates': {
                    'A<->C': modeltest_params.get('rate_AC', 1.0),
                    'A<->G': modeltest_params.get('rate_AG', 1.0),
                    'A<->T': modeltest_params.get('rate_AT', 1.0),
                    'C<->G': modeltest_params.get('rate_CG', 1.0),
                    'C<->T': modeltest_params.get('rate_CT', 1.0),
                    'G<->T': modeltest_params.get('rate_GT', 1.0)
                }
            }
            gamma_shape = modeltest_params.get('gamma_shape', 1.0)
            prop_invariant = modeltest_params.get('prop_invariant', 0.0)
            
        else:
            print(f"  - Using fallback parameter estimation")
            # Fallback to basic estimation
            gtr_params = self.estimate_gtr_parameters(alignment)
            if gtr_params is None:
                print(f"Skipping {filepath}: Could not estimate GTR parameters")
                return None
            
            gamma_params = self.estimate_gamma_parameters(alignment)
            gamma_shape = gamma_params['gamma_shape']
            prop_invariant = gamma_params['prop_invariant']
        
        # Always calculate indel parameters from alignment
        indel_params = self.analyze_indels(alignment)
        
        # Compile results
        result = {
            'filename': os.path.basename(filepath),
            'n_sequences': len(alignment),
            'alignment_length': alignment.get_alignment_length(),
            'method': 'ModelTest-NG' if modeltest_params and len(modeltest_params) > 4 else 'Basic',
            
            # GTR substitution rates
            'rate_AC': gtr_params['substitution_rates'].get('A<->C', 0),
            'rate_AG': gtr_params['substitution_rates'].get('A<->G', 0),
            'rate_AT': gtr_params['substitution_rates'].get('A<->T', 0),
            'rate_CG': gtr_params['substitution_rates'].get('C<->G', 0),
            'rate_CT': gtr_params['substitution_rates'].get('C<->T', 0),
            'rate_GT': gtr_params['substitution_rates'].get('G<->T', 0),
            
            # Base frequencies
            'freq_A': gtr_params['base_frequencies'].get('A', 0.25),
            'freq_C': gtr_params['base_frequencies'].get('C', 0.25),
            'freq_G': gtr_params['base_frequencies'].get('G', 0.25),
            'freq_T': gtr_params['base_frequencies'].get('T', 0.25),
            
            # Gamma + Invariant
            'gamma_shape': gamma_shape,
            'prop_invariant': prop_invariant,
            
            # Indel parameters
            'indel_rate': (indel_params['insertion_events'] + indel_params['deletion_events']) / 
                         alignment.get_alignment_length() if alignment.get_alignment_length() > 0 else 0,
            'mean_indel_length': np.mean(indel_params['insertion_lengths'] + indel_params['deletion_lengths']) 
                               if indel_params['insertion_lengths'] + indel_params['deletion_lengths'] else 0,
            'total_gaps': indel_params['total_gaps']
        }
        
        return result
    
    def cleanup(self):
        pass #! temporary
        """Clean up temporary files"""
        
        '''
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print("Cleaned up temporary files")
        '''
    
    def process_folder(self):
        """Process all FASTA files in the input folder"""
        fasta_files = glob.glob(os.path.join(self.input_folder, "*.fa")) + \
                     glob.glob(os.path.join(self.input_folder, "*.fasta")) + \
                     glob.glob(os.path.join(self.input_folder, "*.fas"))
        
        if not fasta_files:
            print(f"No FASTA files found in {self.input_folder}")
            return
        
        print(f"Found {len(fasta_files)} FASTA files")
        
        for filepath in fasta_files:
            result = self.process_alignment(filepath)
            if result:
                self.results.append(result)
        
        print(f"Successfully processed {len(self.results)} alignments")
        
        # Clean up temporary files
        self.cleanup()
    
    def save_results(self):
        """Save results to CSV and generate summary statistics"""
        if not self.results:
            print("No results to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Save raw results
        csv_path = os.path.join(self.output_folder, "gtr_indel_parameters.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        # Generate summary statistics
        summary_stats = df.describe()
        summary_path = os.path.join(self.output_folder, "parameter_summary.csv")
        summary_stats.to_csv(summary_path)
        print(f"Summary statistics saved to: {summary_path}")
        
        return df
    
    def plot_distributions(self, df):
        """Generate distribution plots for parameters"""
        # GTR substitution rates
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('GTR Substitution Rate Distributions', fontsize=16)
        
        rate_columns = ['rate_AC', 'rate_AG', 'rate_AT', 'rate_CG', 'rate_CT', 'rate_GT']
        rate_labels = ['A↔C', 'A↔G', 'A↔T', 'C↔G', 'C↔T', 'G↔T']
        
        for i, (col, label) in enumerate(zip(rate_columns, rate_labels)):
            ax = axes[i//3, i%3]
            df[col].hist(bins=20, ax=ax, alpha=0.7, edgecolor='black')
            ax.set_title(f'{label} Rate')
            ax.set_xlabel('Rate')
            ax.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'gtr_rate_distributions.png'), dpi=300)
        plt.close()
        
        # Base frequencies
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Base Frequency Distributions', fontsize=16)
        
        freq_columns = ['freq_A', 'freq_C', 'freq_G', 'freq_T']
        freq_labels = ['A', 'C', 'G', 'T']
        
        for i, (col, label) in enumerate(zip(freq_columns, freq_labels)):
            ax = axes[i//2, i%2]
            df[col].hist(bins=20, ax=ax, alpha=0.7, edgecolor='black')
            ax.set_title(f'{label} Frequency')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'base_frequency_distributions.png'), dpi=300)
        plt.close()
        
        # Gamma and invariant sites
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        df['gamma_shape'].hist(bins=20, ax=ax1, alpha=0.7, edgecolor='black')
        ax1.set_title('Gamma Shape Parameter')
        ax1.set_xlabel('Shape (α)')
        ax1.set_ylabel('Frequency')
        
        df['prop_invariant'].hist(bins=20, ax=ax2, alpha=0.7, edgecolor='black')
        ax2.set_title('Proportion of Invariant Sites')
        ax2.set_xlabel('Proportion')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'gamma_invariant_distributions.png'), dpi=300)
        plt.close()
        
        # Indel parameters
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        df['indel_rate'].hist(bins=20, ax=ax1, alpha=0.7, edgecolor='black')
        ax1.set_title('Indel Rate')
        ax1.set_xlabel('Rate')
        ax1.set_ylabel('Frequency')
        
        df['mean_indel_length'].hist(bins=20, ax=ax2, alpha=0.7, edgecolor='black')
        ax2.set_title('Mean Indel Length')
        ax2.set_xlabel('Length')
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'indel_distributions.png'), dpi=300)
        plt.close()
        
        print("Distribution plots saved to output folder")

def main():
    print('started script')
    if len(sys.argv) < 2:
        print("Usage: python gtr_extractor.py <input_folder> [modeltest_path]")
        print("Example: python gtr_extractor.py ./alignments/")
        print("Example: python gtr_extractor.py ./alignments/ /usr/local/bin/modeltest-ng")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    modeltest_path = sys.argv[2] if len(sys.argv) > 2 else "modeltest-ng"
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        sys.exit(1)
    
    # Initialize extractor
    extractor = GTRIndelParameterExtractor(input_folder, modeltest_path=modeltest_path, output_folder="data/algn_params")
    
    # Process all alignments
    extractor.process_folder()
    
    # Save results and generate plots
    df = extractor.save_results()
    if df is not None:
        extractor.plot_distributions(df)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()