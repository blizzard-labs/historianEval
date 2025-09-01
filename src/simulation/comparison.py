import os
import sys
from ete3 import Tree
import subprocess
import re
import pandas as pd

class outputCompare:
    def __init__(self, output_folder, results_file='comparison_results.csv'):
        self.results = {}
        
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder, exist_ok=True)
            
        self.output_folder = output_folder
        self.results_file = os.path.join(output_folder, results_file)
        
        for file in os.listdir(os.path.dirname(output_folder)):
            if file.endswith('.fastas'):
                self.posterior_alignments = os.path.join(os.path.dirname(output_folder), file)
            elif file.endswith('MAP.tree'):
                self.map_tree = os.path.join(os.path.dirname(output_folder), file)
            
        for file in os.listdir(os.path.dirname(os.path.dirname(output_folder))):
            if file.endswith('alignment.fasta'):
                self.truth_alignment = os.path.join(os.path.dirname(os.path.dirname(output_folder)), file)
            elif file.endswith('guide.tree'):
                self.truth_tree = os.path.join(os.path.dirname(os.path.dirname(output_folder)), file)
        
        
    def compute_posterior_decoding(self):
        #cut-range C1.P1.fastas --skip=200 | alignment-chop-internal --tree treetraceCCD1-MAP.tree | alignment-max > C1-max.fasta
        cmd = [
            'cut-range', self.posterior_alignments, '--skip=200', '|',
            'alignment-chop-internal', '--tree', self.map_tree, '|',
            'alignment-max', '>', os.path.join(self.output_folder, 'posterior_decoded.fasta')
        ]
        
        try:
            subprocess.run(' '.join(cmd), shell=True, check=True)
            self.decoded_alignment = os.path.join(self.output_folder, 'posterior_decoded.fasta')
            
            print(f'Posterior decoding successfully run on {self.posterior_alignments}')
        except subprocess.CalledProcessError as e:
            print(f'Error running posterior decoding on {self.map_tree}: {e}')
    
    def compute_sp_scores(self):
        if not hasattr(self, 'decoded_alignment'):
            raise ValueError("Decoded alignment not found. Please run compute_posterior_decoding() first.")
        
        reference, estimate = self.truth_alignment, self.decoded_alignment
        log_path = os.path.join(self.output_folder, "sp_scores.log")
        
        cmd = [
            "java", "-jar",
            "tools/FastSP.jar",
            "-r", reference,
            "-e", estimate
        ]
        
        try:
            with open(log_path, 'w') as log_f:
                subprocess.run(cmd, stdout=log_f, stderr=log_f, check=True)
            print(f'FastSP successfully run on {estimate}')
        except subprocess.CalledProcessError as e:
            print(f'Error running FastSP on {estimate}: {e}')

        with open(log_path, 'r') as log_f:
            contents = log_f.read()
        
        result = {}
        
        patterns = {
        'shared_homologies': r'Number of shared homologies:\s*(\d+)',
        'homologies_reference': r'Number of homologies in the reference alignment:\s*(\d+)',
        'homologies_estimated': r'Number of homologies in the estimated alignment:\s*(\d+)',
        'correctly_aligned_columns': r'Number of correctly aligned columns:\s*(\d+)',
        'aligned_columns_ref': r'Number of aligned columns in ref\. alignment:\s*(\d+)',
        'singleton_insertion_ref': r'Number of singleton and \(uncollapsed\) insertion columns in ref\. alignment:\s*(\d+)\s*(\d+)',
        'aligned_columns_est': r'Number of aligned columns in est\. alignment:\s*(\d+)',
        'singleton_insertion_est': r'Number of singleton and \(uncollapsed\) insertion columns in est\. alignment:\s*(\d+)\s*(\d+)',
        'sp_score': r'SP-Score\s+([\d.]+)',
        'modeler': r'Modeler\s+([\d.]+)',
        'spfn': r'SPFN\s+([\d.]+)',
        'spfp': r'SPFP\s+([\d.]+)',
        'compression_naive': r'Compression \(naive\)\s+([\d.]+)',
        'compression': r'Compression\s+([\d.]+)',
        'tc': r'TC\s+([\d.]+)'
        }
        
        print('Extracting SP scores from log file...')
        for key, pattern in patterns.items():
            match = re.search(pattern, contents)
            if match:
                if key in ['singleton_insertion_ref', 'singleton_insertion_est']:
                    # These have two values, store as tuple
                    result[key] = (int(match.group(1)), int(match.group(2)))
                elif key in ['shared_homologies', 'homologies_reference', 'homologies_estimated',
                            'correctly_aligned_columns', 'aligned_columns_ref', 'aligned_columns_est']:
                    # These are integers
                    result[key] = int(match.group(1))
                else:
                    # These are floats
                    result[key] = float(match.group(1))
        self.results = self.results | result
        
        return result
    
    def calculate_rfl_distance(self, tree1, tree2, k=1):
        """
        Calculate RFL (RF with Lengths) distance between two trees.
        
        This implements the general algorithm where the absolute difference in branch lengths
        is raised to power k before summing:
        - k=1: Robinson & Foulds (1979) - sum of absolute differences
        - k=2: Kuhner & Felsenstein (1994) - sum of squared differences
        
        Args:
            tree1 (Tree): First ETE3 tree object
            tree2 (Tree): Second ETE3 tree object  
            k (int): Power to raise absolute differences (1 or 2)
            
        Returns:
            float: RFL distance
        """
        
        def get_splits_with_lengths(tree):
            """Get bipartitions (splits) with their associated branch lengths."""
            splits = {}
            for node in tree.traverse():
                if not node.is_leaf() and not node.is_root():
                    # Get the split defined by this node
                    leaves_in_clade = set(node.get_leaf_names())
                    all_leaves = set(tree.get_leaf_names())
                    leaves_out_clade = all_leaves - leaves_in_clade
                    
                    # Create a canonical representation of the split
                    # Use the smaller partition as the key for consistency
                    if len(leaves_in_clade) <= len(leaves_out_clade):
                        split = frozenset(leaves_in_clade)
                    else:
                        split = frozenset(leaves_out_clade)
                    
                    # Store the split with its branch length
                    if len(split) > 0 and len(split) < len(all_leaves):  # Exclude trivial splits
                        splits[split] = node.dist if node.dist is not None else 0.0
            
            return splits
        
        try:
            # Get splits with branch lengths for both trees
            splits1 = get_splits_with_lengths(tree1)
            splits2 = get_splits_with_lengths(tree2)
            
            # Calculate RFL distance
            rfl_distance = 0.0
            
            # Get all unique splits from both trees
            all_splits = set(splits1.keys()) | set(splits2.keys())
            
            for split in all_splits:
                length1 = splits1.get(split, 0.0)  # Length 0 if split not in tree1
                length2 = splits2.get(split, 0.0)  # Length 0 if split not in tree2
                
                # Add the absolute difference raised to power k
                rfl_distance += abs(length1 - length2) ** k
            
            return rfl_distance
        except Exception as e:
            print(f"Error calculating RFL distance: {e}")
            return None


    def compute_rf_scores(self, tree1_path, tree2_path):
        if not hasattr(self, 'truth_tree') or not hasattr(self, 'map_tree'):
            raise ValueError("Truth tree or MAP tree not found. Please ensure they are set correctly.")
        
        tree1_path, tree2_path = self.truth_tree, self.map_tree
        # Validate input files
        if not os.path.exists(tree1_path):
            raise FileNotFoundError(f"Tree file not found: {tree1_path}")
        if not os.path.exists(tree2_path):
            raise FileNotFoundError(f"Tree file not found: {tree2_path}")
        
        try:
            results = {}
            
            # Load trees from Newick files
            tree1 = Tree(tree1_path, format=1)
                        
            with open(tree2_path, 'r') as f:
                newick_str2 = re.sub(r'\[&posterior=[^\]]+\]', '', f.read().strip()) + ';'
                
            tree2 = Tree(newick_str2, format=1)
                        
            tree1_leaves = set(tree1.get_leaf_names())
            tree2_leaves = set(tree2.get_leaf_names())
            common_leaves = tree1_leaves.intersection(tree2_leaves)
                        
                        
            rf_result = tree1.robinson_foulds(tree2)
            
            results['rf_distance'] = rf_result[0]
            results['max_rf_distance'] = rf_result[1]
            results['n_common_leaves'] = len(common_leaves)
            results['rfl_distance'] = self.calculate_rfl_distance(tree1, tree2, k=2)
            
            self.results = self.results | results
            
            return results
        except Exception as e:
            raise Exception(f"Error processing trees: {str(e)}")
        
    def export_results(self):
        """Export comparison results to a CSV file."""
        if not self.results:
            print("No results to export.")
            return
        
        df = pd.DataFrame([self.results])
        df.to_csv(self.results_file, sep=',', index=False)
        print(f"Results exported to {self.results_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python comparison.py <output_folder> [results_file]")
        print("Example: python src/simulation/comparison.py data/simulation/SCOPt4e1/seq_1/historian/outputStats comparison_results.csv")
    
    output_folder = sys.argv[1]
    results_file = sys.argv[2] if len(sys.argv) > 2 else 'comparison_results.csv'
    
    #Initialize comparison object
    compare = outputCompare(output_folder, results_file)
    
    print("Beginning comparison for folder:", output_folder, "==========================")
    try:
        print('Computing posterior decoding alignment...')
        compare.compute_posterior_decoding()
        print('Computing SP scores...')
        sp_scores = compare.compute_sp_scores()
        print('Computing RF scores...')
        rf_scores = compare.compute_rf_scores(compare.truth_tree, compare.map_tree)
        
        print("SP Scores:", sp_scores)
        print("RF Scores:", rf_scores)
        
        compare.export_results()
    except Exception as e:
        print(f"Error during comparison: {e}")


if __name__ == '__main__':
    main()