#!/usr/bin/env python3
"""
Protein Evolution Parameter Extraction Script using ModelTest-NG
Indel Parameters from Basic Estimation

This script processes a folder of FASTA multiple sequence alignments (protein) and extracts:
1. Protein substitution model parameters (LG, WAG, JTT, etc.)
2. Amino acid frequencies
3. Gamma shape parameter and proportion of invariant sites
4. Indel parameters (insertion/deletion rates and length distributions)

Requirements:
- modeltest-ng (installed and in PATH, with protein model support)
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

class ProteinParameterExtractor:
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
        
        # Standard 20 amino acids
        self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        # Common protein evolution models
        self.protein_models = ['LG', 'WAG', 'JTT', 'Dayhoff', 'DCMut', 'CpREV', 
                              'mtREV', 'rtREV', 'VT', 'Blosum62', 'mtMam', 'mtArt', 'HIVb', 'HIVw']
        
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
        """Run ModelTest-NG on protein alignment file"""
        base_name = os.path.splitext(os.path.basename(alignment_file))[0]
        output_prefix = os.path.join(self.temp_dir, base_name)
        
        cmd = [
            self.modeltest_path,
            "-i", alignment_file,
            "-o", output_prefix,
            "-d", "aa",  # Specify amino acid data type
            "-t", "ml",
            "-p", "6",   # Number of threads 
            "-m", "LG"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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
        """Parse ModelTest-NG output files for protein models"""
        log_file = output_prefix + ".log"
        
        if not os.path.exists(log_file):
            return None
        
        params = {}
        
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Find the best model section
            best_model_match = re.search(r'Best model according to.*?Model:\s+([A-Za-z0-9+]+)', content, re.DOTALL)
            if best_model_match:
                params['best_model'] = best_model_match.group(1)
            
            # Extract log likelihood
            lnl_pattern = r'lnL:\s+([-\d\.]+)'
            lnl_match = re.search(lnl_pattern, content)
            if lnl_match:
                params['log_likelihood'] = float(lnl_match.group(1))
            
            # Extract amino acid frequencies (20 values)
            # Look for frequency section with 20 values
            freq_pattern = r'Frequencies:\s+((?:[\d\.]+\s+){19}[\d\.]+)'
            freq_match = re.search(freq_pattern, content)
            if freq_match:
                freq_values = [float(x) for x in freq_match.group(1).split()]
                if len(freq_values) == 20:
                    for i, aa in enumerate(self.amino_acids):
                        params[f'freq_{aa}'] = freq_values[i]
            
            # Extract proportion of invariant sites
            pinv_pattern = r'P\.Inv:\s+([\d\.]+)'
            pinv_match = re.search(pinv_pattern, content)
            if pinv_match:
                params['prop_invariant'] = float(pinv_match.group(1))
            
            # Extract gamma shape parameter
            alpha_pattern = r'Alpha:\s+([\d\.]+)'
            alpha_match = re.search(alpha_pattern, content)
            if alpha_match:
                params['gamma_shape'] = float(alpha_match.group(1))
            
            # Also try alternative gamma shape pattern
            gamma_shape_pattern = r'Gamma shape:\s+([\d\.\-]+)'
            gamma_shape_match = re.search(gamma_shape_pattern, content)
            if gamma_shape_match and gamma_shape_match.group(1) != '-':
                try:
                    params['gamma_shape'] = float(gamma_shape_match.group(1))
                except ValueError:
                    pass
            
            # Extract AIC score
            aic_pattern = r'AIC.*?Score:\s+([\d\.]+)'
            aic_match = re.search(aic_pattern, content, re.DOTALL)
            if aic_match:
                params['aic_score'] = float(aic_match.group(1))
            
            # Extract BIC score
            bic_pattern = r'BIC.*?Score:\s+([\d\.]+)'
            bic_match = re.search(bic_pattern, content, re.DOTALL)
            if bic_match:
                params['bic_score'] = float(bic_match.group(1))
            
            # Extract model weight
            weight_pattern = r'Weight:\s+([\d\.]+)'
            weight_match = re.search(weight_pattern, content)
            if weight_match:
                params['model_weight'] = float(weight_match.group(1))
            
            return params if params else None
            
        except Exception as e:
            print(f"Error parsing ModelTest-NG output: {e}")
            return None
    
    def calculate_aa_frequencies(self, alignment):
        """Calculate amino acid frequencies from alignment"""
        aa_counts = Counter()
        total_aas = 0
        
        for record in alignment:
            sequence = str(record.seq).upper()
            for aa in sequence:
                if aa in self.amino_acids:
                    aa_counts[aa] += 1
                    total_aas += 1
        
        if total_aas == 0:
            # Equal frequencies if no data
            return {aa: 1/20 for aa in self.amino_acids}
        
        return {aa: aa_counts.get(aa, 0)/total_aas for aa in self.amino_acids}
    
    def count_aa_substitutions(self, alignment):
        """Count amino acid substitution types between sequences"""
        sub_counts = defaultdict(int)
        total_comparisons = 0
        
        sequences = [str(record.seq).upper() for record in alignment]
        n_seqs = len(sequences)
        
        # Pairwise comparisons
        for i in range(n_seqs):
            for j in range(i+1, n_seqs):
                seq1, seq2 = sequences[i], sequences[j]
                
                for k in range(min(len(seq1), len(seq2))):
                    aa1, aa2 = seq1[k], seq2[k]
                    
                    if aa1 in self.amino_acids and aa2 in self.amino_acids and aa1 != aa2:
                        # Count substitution
                        sub_type = tuple(sorted([aa1, aa2]))
                        sub_counts[sub_type] += 1
                        total_comparisons += 1
        
        return sub_counts, total_comparisons
    
    def estimate_protein_parameters(self, alignment):
        """Estimate protein evolution parameters (fallback method)"""
        print('USING FALLBACK METHOD FOR PROTEIN PARAMETERS...')
        aa_freqs = self.calculate_aa_frequencies(alignment)
        sub_counts, total_subs = self.count_aa_substitutions(alignment)
        
        if total_subs == 0:
            return None
        
        # Calculate basic substitution statistics
        most_common_subs = Counter(sub_counts).most_common(10)
        
        return {
            'aa_frequencies': aa_freqs,
            'total_substitutions': total_subs,
            'most_common_substitutions': most_common_subs,
            'substitution_diversity': len(sub_counts)
        }
    
    def analyze_indels(self, alignment):
        """Analyze indel patterns in the protein alignment"""
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
        
        # Simple insertion estimation
        indel_stats['insertion_events'] = indel_stats['deletion_events']
        indel_stats['insertion_lengths'] = indel_stats['deletion_lengths'].copy()
        
        return indel_stats
    
    def estimate_gamma_parameters(self, alignment):
        """Estimate gamma distribution parameters for rate heterogeneity"""
        sequences = [str(record.seq).upper() for record in alignment]
        n_seqs = len(sequences)
        align_length = len(sequences[0]) if sequences else 0
        
        site_variability = []
        
        for pos in range(align_length):
            column = [seq[pos] if pos < len(seq) else 'X' for seq in sequences]
            valid_aas = [aa for aa in column if aa in self.amino_acids]
            
            if len(valid_aas) > 1:
                # Calculate Shannon entropy as measure of variability
                aa_counts = Counter(valid_aas)
                total = len(valid_aas)
                entropy = -sum((count/total) * np.log2(count/total) for count in aa_counts.values())
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
            'gamma_shape': max(0.1, gamma_shape),
            'prop_invariant': prop_invariant
        }
    
    def process_alignment(self, filepath):
        """Process a single protein alignment file"""
        print(f"Processing: {os.path.basename(filepath)}")
        
        alignment = self.read_alignment(filepath)
        if alignment is None or len(alignment) < 2:
            print(f"Skipping {filepath}: Invalid alignment or too few sequences")
            return None
        
        # Try ModelTest-NG first, fallback to basic estimation
        modeltest_params = self.run_modeltest_ng(filepath)
        
        if modeltest_params and len(modeltest_params) > 4:
            print(f"  - Using ModelTest-NG parameters")
            
            # Extract amino acid frequencies
            aa_freqs = {}
            for aa in self.amino_acids:
                aa_freqs[aa] = modeltest_params.get(f'freq_{aa}', 1/20)
            
            gamma_shape = modeltest_params.get('gamma_shape', 1.0)
            prop_invariant = modeltest_params.get('prop_invariant', 0.0)
            best_model = modeltest_params.get('best_model', 'Unknown')
            log_likelihood = modeltest_params.get('log_likelihood', 0.0)
            aic_score = modeltest_params.get('aic_score', 0.0)
            bic_score = modeltest_params.get('bic_score', 0.0)
            
        else:
            print(f"  - Using fallback parameter estimation")
            # Fallback to basic estimation
            protein_params = self.estimate_protein_parameters(alignment)
            if protein_params is None:
                print(f"Skipping {filepath}: Could not estimate protein parameters")
                return None
            
            aa_freqs = protein_params['aa_frequencies']
            gamma_params = self.estimate_gamma_parameters(alignment)
            gamma_shape = gamma_params['gamma_shape']
            prop_invariant = gamma_params['prop_invariant']
            best_model = 'Basic_Estimation'
            log_likelihood = 0.0
            aic_score = 0.0
            bic_score = 0.0
        
        # Always calculate indel parameters from alignment
        indel_params = self.analyze_indels(alignment)
        
        # Compile results
        result = {
            'filename': os.path.basename(filepath),
            'n_sequences': len(alignment),
            'alignment_length': alignment.get_alignment_length(),
            'best_model': best_model,
            'log_likelihood': log_likelihood,
            'aic_score': aic_score,
            'bic_score': bic_score,
            'method': 'ModelTest-NG' if modeltest_params and len(modeltest_params) > 4 else 'Basic',
        }
        
        # Add amino acid frequencies
        for aa in self.amino_acids:
            result[f'freq_{aa}'] = aa_freqs.get(aa, 1/20)
        
        # Add gamma and invariant parameters
        result.update({
            'gamma_shape': gamma_shape,
            'prop_invariant': prop_invariant,
            
            # Indel parameters
            'indel_rate': (indel_params['insertion_events'] + indel_params['deletion_events']) / 
                         alignment.get_alignment_length() if alignment.get_alignment_length() > 0 else 0,
            'mean_indel_length': np.mean(indel_params['insertion_lengths'] + indel_params['deletion_lengths']) 
                               if indel_params['insertion_lengths'] + indel_params['deletion_lengths'] else 0,
            'total_gaps': indel_params['total_gaps']
        })
        
        return result
    
    def cleanup(self):
        """Clean up temporary files"""
        pass  # Keep temp files for debugging
        
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
        csv_path = os.path.join(self.output_folder, "protein_evolution_parameters.csv")
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        # Generate summary statistics
        summary_stats = df.describe()
        summary_path = os.path.join(self.output_folder, "parameter_summary.csv")
        summary_stats.to_csv(summary_path)
        print(f"Summary statistics saved to: {summary_path}")
        
        return df
    
    def plot_distributions(self, df):
        """Generate distribution plots for protein evolution parameters"""
        
        # Amino acid frequency distributions
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        fig.suptitle('Amino Acid Frequency Distributions', fontsize=16)
        
        for i, aa in enumerate(self.amino_acids):
            ax = axes[i//5, i%5]
            col = f'freq_{aa}'
            if col in df.columns:
                df[col].hist(bins=15, ax=ax, alpha=0.7, edgecolor='black')
                ax.set_title(f'{aa} Frequency')
                ax.set_xlabel('Frequency')
                ax.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'aa_frequency_distributions.png'), dpi=300)
        plt.close()
        
        # Model distribution
        if 'best_model' in df.columns:
            plt.figure(figsize=(12, 8))
            model_counts = df['best_model'].value_counts()
            model_counts.plot(kind='bar')
            plt.title('Distribution of Best-Fit Protein Evolution Models')
            plt.xlabel('Model')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_folder, 'model_distribution.png'), dpi=300)
            plt.close()
        
        # Gamma and invariant sites
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if 'gamma_shape' in df.columns:
            df['gamma_shape'].hist(bins=20, ax=ax1, alpha=0.7, edgecolor='black')
            ax1.set_title('Gamma Shape Parameter')
            ax1.set_xlabel('Shape (α)')
            ax1.set_ylabel('Frequency')
        
        if 'prop_invariant' in df.columns:
            df['prop_invariant'].hist(bins=20, ax=ax2, alpha=0.7, edgecolor='black')
            ax2.set_title('Proportion of Invariant Sites')
            ax2.set_xlabel('Proportion')
            ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'gamma_invariant_distributions.png'), dpi=300)
        plt.close()
        
        # Indel parameters
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if 'indel_rate' in df.columns:
            df['indel_rate'].hist(bins=20, ax=ax1, alpha=0.7, edgecolor='black')
            ax1.set_title('Indel Rate')
            ax1.set_xlabel('Rate')
            ax1.set_ylabel('Frequency')
        
        if 'mean_indel_length' in df.columns:
            df['mean_indel_length'].hist(bins=20, ax=ax2, alpha=0.7, edgecolor='black')
            ax2.set_title('Mean Indel Length')
            ax2.set_xlabel('Length')
            ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'indel_distributions.png'), dpi=300)
        plt.close()
        
        # Model fit comparison (if available)
        if 'aic_score' in df.columns and 'bic_score' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df['aic_score'], df['bic_score'], alpha=0.6)
            plt.xlabel('AIC Score')
            plt.ylabel('BIC Score')
            plt.title('AIC vs BIC Scores for Protein Evolution Models')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_folder, 'aic_vs_bic.png'), dpi=300)
            plt.close()
        
        print("Distribution plots saved to output folder")

def main():
    print('Started protein evolution parameter extraction script')
    if len(sys.argv) < 2:
        print("Usage: python protein_extractor.py <input_folder> [output_folder] [modeltest_path]")
        print("Example: python protein_extractor.py ./protein_alignments/")
        print("Example: python protein_extractor.py ./protein_alignments/ ./results/")
        print("Example: python protein_extractor.py ./protein_alignments/ ./results/ /usr/local/bin/modeltest-ng")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "protein_results"
    modeltest_path = sys.argv[3] if len(sys.argv) > 3 else "modeltest-ng"
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        sys.exit(1)
    
    # Initialize extractor
    extractor = ProteinParameterExtractor(input_folder, output_folder=output_folder, modeltest_path=modeltest_path)
    
    # Process all alignments
    extractor.process_folder()
    
    # Save results and generate plots
    df = extractor.save_results()
    if df is not None:
        extractor.plot_distributions(df)
    
    print("Protein evolution analysis complete!")

if __name__ == "__main__":
    main()