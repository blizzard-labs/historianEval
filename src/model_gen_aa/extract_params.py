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
from Bio import AlignIO, SeqIO, Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import warnings

import tree_metrics

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
    
    def read_tree(self, filepath):
        """Read NEWICK tree file"""
        try:
            tree = Phylo.read(filepath, "newick")
            return tree
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
    
    def run_modeltest_ng(self, alignment_file, tree_file="", model="LG"):
        """Run ModelTest-NG on protein alignment file"""
        base_name = os.path.splitext(os.path.basename(alignment_file))[0]
        output_prefix = os.path.join(self.temp_dir, base_name)
        
        if len(tree_file) > 0:
            cmd = [
                self.modeltest_path,
                "-i", alignment_file,
                "-o", output_prefix,
                "-d", "aa",
                "-t", "user",
                "--utree", tree_file,
                "-p", "6",
                "-m", model            ]
        else:
            cmd = [
                self.modeltest_path,
                "-i", alignment_file,
                "-o", output_prefix,
                "-d", "aa",  # Specify amino acid data type
                "-t", "ml",
                "-p", "6",   # Number of threads 
                "-m", model
            ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=6000)
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
            pinv_pattern = r'Inv\. sites prop:\s+([\d.]+)'
            pinv_match = re.search(pinv_pattern, content)
            if pinv_match:
                params['prop_invariant'] = float(pinv_match.group(1))
            else:
                params['prop_invariant'] = float(0) 
            
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
        """Analyze indel patterns in the protein alignment with proper rate calculation"""
        print("  - Analyzing indel patterns...")
        
        sequences = [str(record.seq).upper() for record in alignment]
        n_seqs = len(sequences)
        align_length = len(sequences[0]) if sequences else 0
        
        if n_seqs < 2:
            return {
                'insertion_rate': 0.0,
                'deletion_rate': 0.0,
                'insertion_events': 0,
                'deletion_events': 0,
                'insertion_lengths': [],
                'deletion_lengths': [],
                'mean_insertion_length': 0.0,
                'mean_deletion_length': 0.0,
                'total_gaps': 0,
                'indel_to_substitution_ratio': 0.0
            }
        
        # Count total gaps across all sequences
        total_gaps = 0
        gap_columns = 0
        
        # Analyze column by column
        for pos in range(align_length):
            column = [seq[pos] if pos < len(seq) else '-' for seq in sequences]
            gap_count = column.count('-')
            total_gaps += gap_count
            
            # Count columns with any gaps
            if gap_count > 0:
                gap_columns += 1
        
        # Count indel events more conservatively
        # Method 1: Count gap runs (consecutive gaps) as single events
        insertion_events = 0
        deletion_events = 0
        insertion_lengths = []
        deletion_lengths = []
        
        # For each sequence, find gap runs
        for seq in sequences:
            gap_runs = self._find_gap_runs(seq)
            for start, length in gap_runs:
                # Consider this a deletion event
                deletion_events += 1
                deletion_lengths.append(length)
        
        # Alternative method: Count indel events based on parsimony
        # This is more conservative and realistic
        parsimony_indels = self._count_parsimony_indels(sequences)
        
        # Calculate total ungapped sites for rate normalization
        total_ungapped_sites = 0
        for pos in range(align_length):
            column = [seq[pos] if pos < len(seq) else '-' for seq in sequences]
            ungapped_count = sum(1 for aa in column if aa in self.amino_acids)
            total_ungapped_sites += ungapped_count
        
        # More realistic rate calculation
        # Rate per site per sequence (evolutionary rate)
        if total_ungapped_sites > 0:
            # Use parsimony-based counts for more realistic rates
            insertion_rate = parsimony_indels['insertions'] / (align_length * n_seqs) * 100  # Scale to reasonable range
            deletion_rate = parsimony_indels['deletions'] / (align_length * n_seqs) * 100
            
            '''
            # Alternative: Use gap density as proxy for indel rate
            gap_density = total_gaps / (align_length * n_seqs)
            
            # More conservative approach: base rates on gap density
            insertion_rate = gap_density * 0.1  # Assume 10% of gaps are insertions
            deletion_rate = gap_density * 0.9   # Assume 90% of gaps are deletions
            '''
            
        else:
            insertion_rate = 0.0
            deletion_rate = 0.0
        
        # Calculate mean lengths
        mean_insertion_length = np.mean(insertion_lengths) if insertion_lengths else 0.0
        mean_deletion_length = np.mean(deletion_lengths) if deletion_lengths else 0.0
        
        # If no lengths recorded, estimate from gap runs
        if mean_deletion_length == 0.0 and deletion_events > 0:
            all_gap_lengths = []
            for seq in sequences:
                gap_runs = self._find_gap_runs(seq)
                all_gap_lengths.extend([length for start, length in gap_runs])
            mean_deletion_length = np.mean(all_gap_lengths) if all_gap_lengths else 1.0
        
        # Estimate insertion length (harder to determine from alignment)
        if mean_insertion_length == 0.0:
            mean_insertion_length = mean_deletion_length * 0.8  # Typically shorter than deletions
        
        # Calculate indel to substitution ratio
        sub_counts, total_subs = self.count_aa_substitutions(alignment)
        total_indel_events = parsimony_indels['insertions'] + parsimony_indels['deletions']
        indel_to_sub_ratio = total_indel_events / total_subs if total_subs > 0 else 0.0 #! Conservative estimate
        
        return {
            'indel_method' : 'parsimony',
            'insertion_rate': max(0.0, min(insertion_rate, 1.0)),  # Cap at reasonable values
            'deletion_rate': max(0.0, min(deletion_rate, 1.0)),
            'insertion_events': parsimony_indels['insertions'],
            'deletion_events': parsimony_indels['deletions'],
            'insertion_lengths': insertion_lengths,
            'deletion_lengths': deletion_lengths,
            'mean_insertion_length': max(1.0, mean_insertion_length),
            'mean_deletion_length': max(1.0, mean_deletion_length),
            'total_gaps': total_gaps,
            'indel_to_substitution_ratio': indel_to_sub_ratio
        }
    
    def _count_parsimony_indels(self, sequences):
        """Count indel events using parsimony principle"""
        align_length = len(sequences[0]) if sequences else 0
        n_seqs = len(sequences)
        
        insertions = 0
        deletions = 0
        
        # For each column, count the minimum number of indel events needed
        for pos in range(align_length):
            column = [seq[pos] if pos < len(seq) else '-' for seq in sequences]
            gap_count = column.count('-')
            aa_count = sum(1 for aa in column if aa in self.amino_acids)
            
            # If column has both gaps and amino acids, there was likely an indel event
            if gap_count > 0 and aa_count > 0:
                # Use parsimony: minimum number of events to explain the pattern
                if gap_count < aa_count:
                    # More likely to be deletions in fewer sequences
                    deletions += 1
                else:
                    # More likely to be insertions in fewer sequences
                    insertions += 1
        
        # Count consecutive indel regions to avoid overcounting
        # Group consecutive indel columns together
        indel_regions = self._group_consecutive_indel_columns(sequences)
        
        return {
            'insertions': len([r for r in indel_regions if r['type'] == 'insertion']),
            'deletions': len([r for r in indel_regions if r['type'] == 'deletion'])
        }
    
    def _group_consecutive_indel_columns(self, sequences):
        """Group consecutive columns with indels into single events"""
        align_length = len(sequences[0]) if sequences else 0
        regions = []
        current_region = None
        
        for pos in range(align_length):
            column = [seq[pos] if pos < len(seq) else '-' for seq in sequences]
            gap_count = column.count('-')
            aa_count = sum(1 for aa in column if aa in self.amino_acids)
            
            if gap_count > 0 and aa_count > 0:
                # This column has an indel
                indel_type = 'deletion' if gap_count < aa_count else 'insertion'
                
                if current_region is None:
                    # Start new region
                    current_region = {
                        'start': pos,
                        'end': pos,
                        'type': indel_type,
                        'length': 1
                    }
                elif current_region['type'] == indel_type and pos == current_region['end'] + 1:
                    # Extend current region
                    current_region['end'] = pos
                    current_region['length'] += 1
                else:
                    # End current region and start new one
                    regions.append(current_region)
                    current_region = {
                        'start': pos,
                        'end': pos,
                        'type': indel_type,
                        'length': 1
                    }
            else:
                # No indel in this column
                if current_region is not None:
                    regions.append(current_region)
                    current_region = None
        
        # Don't forget the last region
        if current_region is not None:
            regions.append(current_region)
        
        return regions
    
    def _find_gap_runs(self, sequence):
        """Find consecutive gap runs in a sequence"""
        gap_runs = []
        in_gap = False
        gap_start = 0
        
        for i, char in enumerate(sequence):
            if char == '-':
                if not in_gap:
                    gap_start = i
                    in_gap = True
            else:
                if in_gap:
                    gap_length = i - gap_start
                    if gap_length > 0:  # Only count non-zero length gaps
                        gap_runs.append((gap_start, gap_length))
                    in_gap = False
        
        # Handle gap at end of sequence
        if in_gap:
            gap_length = len(sequence) - gap_start
            if gap_length > 0:
                gap_runs.append((gap_start, gap_length))
        
        return gap_runs

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
    
    def process_alignment(self, filepath, tree_file="", only_indels=False):
        """Process a single protein alignment file"""
        print(f"Processing: {os.path.basename(filepath)}")
        
        alignment = self.read_alignment(filepath)
        if alignment is None or len(alignment) < 2:
            print(f"Skipping {filepath}: Invalid alignment or too few sequences")
            return None
        
        if not only_indels:
            if (len(tree_file) > 0):
                tree = self.read_tree(tree_file)
                
                if tree is None:
                    print(f"Skipping reading tree {tree_file}: Invalid tree")
                    modeltest_params = self.run_modeltest_ng(filepath)
                else:
                    modeltest_params = self.run_modeltest_ng(filepath, tree_file=tree_file)
            else:
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
                print(f'Model-test params: {modeltest_params}')
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
        else:
            print(f'- Skipping ModelTest-NG, only calculating indel parameters')
        
        # Always calculate indel parameters from alignment
        indel_params = self.analyze_indels(alignment)
        
        # Compile results
        result = {
            'filename': os.path.basename(filepath),
            'n_sequences': len(alignment),
            'alignment_length': alignment.get_alignment_length(),
        }
        
        if not only_indels:
            #Add modeltest results
            result.update({
                'best_model': best_model,
                'log_likelihood': log_likelihood,
                'aic_score': aic_score,
                'bic_score': bic_score,
                'method': 'ModelTest-NG' if modeltest_params and len(modeltest_params) > 4 else 'Basic',
            })
            
            # Add amino acid frequencies
            for aa in self.amino_acids:
                result[f'freq_{aa}'] = aa_freqs.get(aa, 1/20)
            
            # Add gamma and invariant parameters
            result.update({
                'gamma_shape': gamma_shape,
                'prop_invariant': prop_invariant,
            })
        
        # Add indel parameters
        result.update({
            'indel_method': indel_params['indel_method'],
            'insertion_rate': indel_params['insertion_rate'],
            'deletion_rate': indel_params['deletion_rate'],
            'insertion_events': indel_params['insertion_events'],
            'deletion_events': indel_params['deletion_events'],
            'mean_insertion_length': indel_params['mean_insertion_length'],
            'mean_deletion_length': indel_params['mean_deletion_length'],
            'total_gaps': indel_params['total_gaps'],
            'indel_to_substitution_ratio': indel_params['indel_to_substitution_ratio']
        })
        
        #Add tree shape parameters
        if len(tree_file) > 0:
            with open (tree_file, "r") as f:
                tree_str = f.read()
                tree_params = tree_metrics.analyze_tree_balance(tree_str)
        
        result.update({
            'n_internal_nodes': tree_params['n_internal_nodes'],
            'max_depth' : tree_params['max_depth'],
            'colless_index': tree_params['colless_index'],
            'normalized_colless_index' : tree_params['normalized_colless_index']
        })
        
        return result
    
    def cleanup(self):
        """Clean up temporary files"""
        pass  # Keep temp files for debugging
        
    def process_folder(self, only_indels=False):
        """Process all FASTA files in the input folder"""
        fasta_files = glob.glob(os.path.join(self.input_folder, "*.fa")) + \
                     glob.glob(os.path.join(self.input_folder, "*.fasta")) + \
                     glob.glob(os.path.join(self.input_folder, "*.fas"))
        
        if not fasta_files:
            print(f"No FASTA files found in {self.input_folder}")
            return
        
        print(f"Found {len(fasta_files)} FASTA files")
        
        tree_folder = self.input_folder.replace("alignments", "trees")
        
        for filepath in fasta_files:
            tree_file = ""
            
            if os.path.exists(tree_folder):
                for f in os.listdir(tree_folder): # checking for pre-existant tree file
                    c = filepath
                    
                    if f.replace(".tree", "").strip() == c.split("/")[-1].replace(".fasta", "").replace(".fa", ""):
                        tree_file = os.path.join(tree_folder, f)
                        print(f"Pre-existant tree found for {f}!")
                    
                    ''' ORTHOMAM DB
                    if f.replace(".tree", "").strip() == c.split("/")[-1].replace("_AA.fasta", "").replace(".fa", ""):
                        tree_file = os.path.join(tree_folder, f)
                        print(f"Pre-existant tree found for {f}!")
                    '''
            
            if (len(tree_file) > 0):
                result = self.process_alignment(filepath, tree_file, only_indels=only_indels)
            else:
                result = self.process_alignment(filepath, only_indels=only_indels)
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
        
        # Improved indel parameters plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if 'insertion_rate' in df.columns:
            df['insertion_rate'].hist(bins=20, ax=axes[0,0], alpha=0.7, edgecolor='black')
            axes[0,0].set_title('Insertion Rate')
            axes[0,0].set_xlabel('Rate per site')
            axes[0,0].set_ylabel('Frequency')
        
        if 'deletion_rate' in df.columns:
            df['deletion_rate'].hist(bins=20, ax=axes[0,1], alpha=0.7, edgecolor='black')
            axes[0,1].set_title('Deletion Rate')
            axes[0,1].set_xlabel('Rate per site')
            axes[0,1].set_ylabel('Frequency')
        
        if 'mean_insertion_length' in df.columns:
            df['mean_insertion_length'].hist(bins=20, ax=axes[1,0], alpha=0.7, edgecolor='black')
            axes[1,0].set_title('Mean Insertion Length')
            axes[1,0].set_xlabel('Length (amino acids)')
            axes[1,0].set_ylabel('Frequency')
        
        if 'mean_deletion_length' in df.columns:
            df['mean_deletion_length'].hist(bins=20, ax=axes[1,1], alpha=0.7, edgecolor='black')
            axes[1,1].set_title('Mean Deletion Length')
            axes[1,1].set_xlabel('Length (amino acids)')
            axes[1,1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'indel_distributions.png'), dpi=300)
        plt.close

def main():
    print('Started protein evolution parameter extraction script')
    if len(sys.argv) < 2:
        print("Usage: python protein_extractor.py <input_folder> [output_folder] [modeltest_path] [only_indels]")
        print("Example: python protein_extractor.py ./protein_alignments/")
        print("Example: python protein_extractor.py ./protein_alignments/ ./results/")
        print("Example: python protein_extractor.py ./protein_alignments/ ./results/ /usr/local/bin/modeltest-ng false")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "protein_results"
    modeltest_path = sys.argv[3] if len(sys.argv) > 3 else "modeltest-ng"
    only_indels = (sys.argv[4].lower() == 'true') if len(sys.argv) > 4 else False
    
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        sys.exit(1)
    
    # Initialize extractor
    extractor = ProteinParameterExtractor(input_folder, output_folder=output_folder, modeltest_path=modeltest_path)
    
    # Process all alignments
    extractor.process_folder(only_indels=only_indels)
    
    # Save results and generate plots
    df = extractor.save_results()
    if df is not None:
        extractor.plot_distributions(df)
    
    print("Protein evolution analysis complete!")

if __name__ == "__main__":
    main()