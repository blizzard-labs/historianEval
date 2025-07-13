#!/usr/bin/env python3
"""
SPR Tree Variant Generator

This script performs SPR (Subtree Pruning and Regrafting) operations on a consensus tree
until it reaches specified RF (Robinson-Foulds) distances, generating multiple variant trees.

Dependencies:
- dendropy: pip install dendropy
- numpy: pip install numpy

Usage:
python spr_tree_generator.py input_tree.nwk output_folder threshold1,threshold2,threshold3 --replicates 10
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import dendropy
    from dendropy.calculate import treecompare
    import numpy as np
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install required packages:")
    print("pip install dendropy numpy")
    sys.exit(1)


class SPRTreeGenerator:
    """Class for generating tree variants using SPR operations."""
    
    def __init__(self, consensus_tree_path: str, random_seed: Optional[int] = None):
        """
        Initialize the SPR tree generator.
        
        Args:
            consensus_tree_path: Path to the consensus tree file
            random_seed: Random seed for reproducibility
        """
        self.consensus_tree_path = consensus_tree_path
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Load the consensus tree
        self.consensus_tree = self._load_tree(consensus_tree_path)
        self.taxon_namespace = self.consensus_tree.taxon_namespace
        
    def _load_tree(self, tree_path: str) -> dendropy.Tree:
        """Load a tree from file."""
        try:
            tree = dendropy.Tree.get_from_path(tree_path, "newick")
            return tree
        except Exception as e:
            raise ValueError(f"Error loading tree from {tree_path}: {e}")
    
    def _calculate_rf_distance(self, tree1: dendropy.Tree, tree2: dendropy.Tree) -> int:
        """Calculate Robinson-Foulds distance between two trees."""
        try:
            # Ensure both trees use the same taxon namespace
            tree1_copy = tree1.clone()
            tree2_copy = tree2.clone()
            tree1_copy.taxon_namespace = self.taxon_namespace
            tree2_copy.taxon_namespace = self.taxon_namespace
            
            rf_distance = treecompare.symmetric_difference(tree1_copy, tree2_copy)
            return rf_distance
        except Exception as e:
            print(f"Warning: Error calculating RF distance: {e}")
            return -1
    
    def _perform_spr_operation(self, tree: dendropy.Tree) -> dendropy.Tree:
        """
        Perform a single SPR operation on the tree.
        
        Args:
            tree: Input tree to modify
            
        Returns:
            Modified tree after SPR operation
        """
        tree_copy = tree.clone()
        
        # Get all internal nodes (potential pruning points)
        internal_nodes = [node for node in tree_copy.preorder_node_iter() 
                         if not node.is_leaf() and node.parent_node is not None]
        
        if len(internal_nodes) < 2:
            return tree_copy  # Can't perform SPR on tree with too few internal nodes
        
        # Select a random subtree to prune
        prune_node = random.choice(internal_nodes)
        
        # Get all possible regrafting positions (nodes, not edges)
        possible_regraft_nodes = []
        for node in tree_copy.preorder_node_iter():
            if (node != prune_node and 
                node not in prune_node.preorder_iter() and
                node != prune_node.parent_node and
                node.parent_node is not None):  # Exclude root
                possible_regraft_nodes.append(node)
        
        if not possible_regraft_nodes:
            return tree_copy  # No valid regrafting positions
        
        # Select random regrafting position
        regraft_node = random.choice(possible_regraft_nodes)
        
        # Perform SPR operation
        try:
            # Store the parent of the node we're regrafting to
            regraft_parent = regraft_node.parent_node
            
            # Remove the subtree to be moved
            prune_parent = prune_node.parent_node
            prune_parent.remove_child(prune_node)
            
            # Create new internal node for regrafting
            new_internal = dendropy.Node()
            
            # Insert new internal node between regraft_node and its parent
            regraft_parent.remove_child(regraft_node)
            regraft_parent.add_child(new_internal)
            new_internal.add_child(regraft_node)
            new_internal.add_child(prune_node)
            
            # Clean up any unifurcations created
            tree_copy.suppress_unifurcations()
            
        except Exception as e:
            print(f"Warning: SPR operation failed: {e}")
            return tree_copy
        
        return tree_copy
    
    def generate_variant_tree(self, target_rf_distance: int, max_iterations: int = 1200) -> Tuple[dendropy.Tree, int]:
        """
        Generate a variant tree with approximately the target RF distance.
        
        Args:
            target_rf_distance: Target RF distance from consensus tree
            max_iterations: Maximum number of SPR operations to try
            
        Returns:
            Tuple of (variant_tree, actual_rf_distance)
        """
        current_tree = self.consensus_tree.clone()
        current_rf = 0
        iterations = 0
        
        while current_rf < target_rf_distance and iterations < max_iterations:
            # Perform SPR operation
            new_tree = self._perform_spr_operation(current_tree)
            new_rf = self._calculate_rf_distance(self.consensus_tree, new_tree)
            
            if new_rf == -1:  # Error in RF calculation
                iterations += 1
                continue
            
            # Accept the new tree if it moves us closer to the target
            if new_rf > current_rf:
                current_tree = new_tree
                current_rf = new_rf
                
            iterations += 1
            
            # Optional: print progress for long runs
            if iterations % 100 == 0:
                print(f"  Iteration {iterations}: RF distance = {current_rf}")
        
        return current_tree, current_rf
    
    def generate_multiple_variants(self, thresholds: List[float], replicates: int = 10, 
                                 output_folder: str = "spr_variants") -> None:
        """
        Generate multiple variant trees for each threshold RF distance.
        
        Args:
            thresholds: List of target RF distances
            replicates: Number of variant trees to generate per threshold
            output_folder: Output directory for generated trees
        """
        # Create output directory
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Generating SPR variant trees...")
        print(f"Consensus tree: {self.consensus_tree_path}")
        print(f"Output folder: {output_folder}")
        print(f"Thresholds: {thresholds}")
        print(f"Replicates per threshold: {replicates}")
        print()
        
        # Generate summary report
        summary_file = output_path / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write("SPR Tree Variant Generation Summary\n")
            f.write("="*40 + "\n\n")
            f.write(f"Consensus tree: {self.consensus_tree_path}\n")
            f.write(f"Thresholds: {thresholds}\n")
            f.write(f"Replicates per threshold: {replicates}\n\n")
            f.write("Results:\n")
            f.write("-"*20 + "\n")
        
        counter = 0
        for threshold in thresholds:
            counter += 1
            print(f"Processing threshold RF distance: {threshold}")
            
            # Create subfolder for this threshold
            threshold_folder = output_path / f"seq_{counter}"
            threshold_folder.mkdir(exist_ok=True)
            
            successful_variants = 0
            
            for replicate in range(replicates):
                print(f"  Generating replicate {replicate + 1}/{replicates}...")
                
                try:
                    variant_tree, actual_rf = self.generate_variant_tree(threshold)
                    
                    # Save the variant tree
                    output_file = threshold_folder / f"variant_{replicate + 1:03d}_rf_{actual_rf}.nwk"
                    variant_tree.write_to_path(str(output_file), "newick")
                    
                    successful_variants += 1
                    
                    print(f"    Saved: {output_file.name} (RF distance: {actual_rf})")
                    
                except Exception as e:
                    print(f"    Error generating replicate {replicate + 1}: {e}")
            
            # Update summary
            with open(summary_file, 'a') as f:
                f.write(f"RF threshold {threshold}: {successful_variants}/{replicates} successful variants\n")
            
            print(f"  Completed: {successful_variants}/{replicates} successful variants")
            print()
        
        print(f"All variant trees generated successfully!")
        print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SPR variant trees with specified RF distances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python spr_tree_generator.py consensus.nwk output_folder 2,4,6,8 --replicates 5
  python spr_tree_generator.py tree.nwk variants 1,3,5 --replicates 10 --seed 42
        """
    )
    
    parser.add_argument("consensus_tree", help="Path to consensus tree file (Newick format)")
    parser.add_argument("output_folder", help="Output folder for variant trees")
    parser.add_argument("thresholds", help="Comma-separated list of RF distance thresholds")
    parser.add_argument("--replicates", "-r", type=int, default=10,
                       help="Number of variant trees per threshold (default: 10)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                       help="Random seed for reproducibility")
    parser.add_argument("--max-iterations", "-m", type=int, default=1200,
                       help="Maximum SPR operations per variant (default: 1000)")
    
    args = parser.parse_args()
    
    # Parse thresholds
    try:
        thresholds = [float(x.strip()) for x in args.thresholds.split(",")]
    except ValueError:
        print("Error: Thresholds must be comma-separated integers")
        sys.exit(1)
    
    # Validate inputs
    if not os.path.exists(args.consensus_tree):
        print(f"Error: Consensus tree file not found: {args.consensus_tree}")
        sys.exit(1)
    
    if any(t <= 0 for t in thresholds):
        print("Error: All thresholds must be positive integers")
        sys.exit(1)
    
    if args.replicates <= 0:
        print("Error: Number of replicates must be positive")
        sys.exit(1)
    
    try:
        # Create generator and run
        generator = SPRTreeGenerator(args.consensus_tree, args.seed)
        generator.generate_multiple_variants(thresholds, args.replicates, args.output_folder)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()