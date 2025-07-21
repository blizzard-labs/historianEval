#!/usr/bin/env python3
"""
Birth-Death Tree Simulated Annealing Optimizer

This script generates birth-death trees using DendroPy and performs simulated annealing
optimization through SPR (Subtree Pruning and Regrafting) moves to minimize the distance
to a target vector of normalized Colless imbalance metric and gamma statistic.
"""

import dendropy
import numpy as np
import random
import math
from typing import Tuple, List, Optional
import argparse
import copy


class BDTreeOptimizer:
    """Birth-Death Tree optimizer using simulated annealing."""
    
    def __init__(self, birth_rate: float, death_rate: float, bd_model: str,
                 target_colless: float, target_gamma: float, num_taxa: int = 20):
        """
        Initialize the optimizer.
        
        Args:
            birth_rate: Birth rate parameter
            death_rate: Death rate parameter
            bd_model: BD model type (e.g., 'best_BCSTDCST', 'best_BEXPDCST', etc.)
            target_colless: Target normalized Colless imbalance
            target_gamma: Target gamma statistic
            num_taxa: Number of taxa in the tree
        """
        self.birth_rate = birth_rate
        self.death_rate = death_rate
        self.bd_model = bd_model
        self.target_vector = np.array([target_colless, target_gamma])
        self.num_taxa = num_taxa
        self.best_tree = None
        self.best_distance = float('inf')
        self.current_tree = None
        
    def generate_bd_tree(self) -> dendropy.Tree:
        """
        Generate a birth-death tree based on the specified model.
        
        Returns:
            DendroPy Tree object
        """
        # Create a simple birth-death process tree
        # Note: This is a simplified implementation - you may need to adjust
        # based on the specific BD model requirements
        
        taxon_namespace = dendropy.TaxonNamespace([f"T{i+1}" for i in range(self.num_taxa)])
        
        if 'CST' in self.bd_model:
            # Constant rate model
            tree = dendropy.simulate.treesim.birth_death_tree(
                birth_rate=self.birth_rate,
                death_rate=self.death_rate,
                taxon_namespace=taxon_namespace,
                num_extant_tips=self.num_taxa
            )
        elif 'EXP' in self.bd_model:
            # Exponential model - simulate with varying rates
            # This is a simplified approach
            tree = dendropy.simulate.treesim.birth_death_tree(
                birth_rate=self.birth_rate,
                death_rate=self.death_rate,
                taxon_namespace=taxon_namespace,
                num_extant_tips=self.num_taxa
            )
        elif 'LIN' in self.bd_model:
            # Linear model - another simplified approach
            tree = dendropy.simulate.treesim.birth_death_tree(
                birth_rate=self.birth_rate,
                death_rate=self.death_rate,
                taxon_namespace=taxon_namespace,
                num_extant_tips=self.num_taxa
            )
        else:
            # Default to constant rate
            tree = dendropy.simulate.treesim.birth_death_tree(
                birth_rate=self.birth_rate,
                death_rate=self.death_rate,
                taxon_namespace=taxon_namespace,
                num_extant_tips=self.num_taxa
            )
            
        return tree
    
    def calculate_colless_imbalance(self, tree: dendropy.Tree) -> float:
        """
        Calculate the normalized Colless imbalance metric.
        
        Args:
            tree: DendroPy Tree object
            
        Returns:
            Normalized Colless imbalance value
        """
        def colless_recursive(node):
            if node.is_leaf():
                return 0, 1
            
            children = node.child_nodes()
            if len(children) != 2:
                return 0, sum(1 for _ in node.leaf_iter())
            
            left_imbalance, left_leaves = colless_recursive(children[0])
            right_imbalance, right_leaves = colless_recursive(children[1])
            
            imbalance = left_imbalance + right_imbalance + abs(left_leaves - right_leaves)
            total_leaves = left_leaves + right_leaves
            
            return imbalance, total_leaves
        
        if tree.seed_node is None:
            return 0.0
            
        imbalance, n_leaves = colless_recursive(tree.seed_node)
        
        # Normalize by maximum possible imbalance for n leaves
        if n_leaves <= 2:
            return 0.0
        
        max_imbalance = (n_leaves - 1) * (n_leaves - 2) / 2
        return imbalance / max_imbalance if max_imbalance > 0 else 0.0
    
    def calculate_gamma_statistic(self, tree: dendropy.Tree) -> float:
        """
        Calculate the gamma statistic (Pybus & Harvey 2000).
        
        Args:
            tree: DendroPy Tree object
            
        Returns:
            Gamma statistic value
        """
        # Get all internal nodes (excluding root and leaves)
        internal_nodes = [node for node in tree.internal_nodes() if node != tree.seed_node]
        
        if len(internal_nodes) < 2:
            return 0.0
        
        # Calculate node depths (distance from tips)
        node_depths = []
        for node in internal_nodes:
            # Calculate distance to furthest leaf
            max_depth = max(node.distance_from_tip() for leaf in node.leaf_iter() 
                           for _ in [node.distance_from_node(leaf)])
            node_depths.append(max_depth)
        
        if len(node_depths) < 2:
            return 0.0
            
        node_depths.sort()
        n = len(node_depths)
        
        # Calculate gamma statistic
        sum_distances = sum(node_depths[:-1])  # Exclude the root
        expected_sum = n * (n + 1) / 4
        variance = n * (n + 1) * (2 * n + 1) / 24
        
        if variance <= 0:
            return 0.0
            
        gamma = (sum_distances - expected_sum) / math.sqrt(variance)
        return gamma
    
    def calculate_tree_statistics(self, tree: dendropy.Tree) -> np.ndarray:
        """
        Calculate both Colless imbalance and gamma statistic for a tree.
        
        Args:
            tree: DendroPy Tree object
            
        Returns:
            NumPy array containing [colless_imbalance, gamma_statistic]
        """
        colless = self.calculate_colless_imbalance(tree)
        gamma = self.calculate_gamma_statistic(tree)
        return np.array([colless, gamma])
    
    def calculate_distance(self, tree_stats: np.ndarray) -> float:
        """
        Calculate Euclidean distance between tree statistics and target vector.
        
        Args:
            tree_stats: Array containing tree statistics
            
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(tree_stats - self.target_vector)
    
    def perform_spr_move(self, tree: dendropy.Tree) -> dendropy.Tree:
        """
        Perform a Subtree Pruning and Regrafting (SPR) move on the tree.
        
        Args:
            tree: Input tree
            
        Returns:
            Modified tree after SPR move
        """
        # Make a copy to avoid modifying the original
        new_tree = copy.deepcopy(tree)
        
        # Get all internal nodes that can be pruned (not root, and have parent)
        internal_nodes = [node for node in new_tree.internal_nodes() 
                         if node.parent_node is not None and node != new_tree.seed_node]
        
        if len(internal_nodes) < 2:
            return new_tree
        
        # Select a subtree to prune
        prune_node = random.choice(internal_nodes)
        prune_parent = prune_node.parent_node
        
        # Remove the subtree
        prune_parent.remove_child(prune_node)
        
        # If parent now has only one child, collapse it
        if len(prune_parent.child_nodes()) == 1 and prune_parent != new_tree.seed_node:
            grandparent = prune_parent.parent_node
            if grandparent:
                remaining_child = prune_parent.child_nodes()[0]
                grandparent.remove_child(prune_parent)
                grandparent.add_child(remaining_child)
        
        # Find possible regraft positions (all edges except those in the pruned subtree)
        possible_edges = []
        for node in new_tree.preorder_node_iter():
            if node.parent_node is not None:
                possible_edges.append((node.parent_node, node))
        
        if not possible_edges:
            # If no valid regraft position, return original tree
            return tree
        
        # Select random edge to regraft
        regraft_parent, regraft_child = random.choice(possible_edges)
        
        # Create new internal node
        new_internal = dendropy.Node()
        
        # Insert new internal node on the selected edge
        regraft_parent.remove_child(regraft_child)
        regraft_parent.add_child(new_internal)
        new_internal.add_child(regraft_child)
        new_internal.add_child(prune_node)
        
        return new_tree
    
    def simulated_annealing(self, initial_temp: float = 1.0, cooling_rate: float = 0.95,
                          min_temp: float = 1e-6, max_iterations: int = 10000) -> dendropy.Tree:
        """
        Perform simulated annealing optimization.
        
        Args:
            initial_temp: Starting temperature
            cooling_rate: Temperature reduction factor
            min_temp: Minimum temperature (stopping criterion)
            max_iterations: Maximum number of iterations
            
        Returns:
            Best tree found during optimization
        """
        # Generate initial tree
        self.current_tree = self.generate_bd_tree()
        current_stats = self.calculate_tree_statistics(self.current_tree)
        current_distance = self.calculate_distance(current_stats)
        
        # Initialize best solution
        self.best_tree = copy.deepcopy(self.current_tree)
        self.best_distance = current_distance
        
        temperature = initial_temp
        iteration = 0
        
        print(f"Starting simulated annealing...")
        print(f"Initial distance: {current_distance:.6f}")
        print(f"Target vector: {self.target_vector}")
        print(f"Initial stats: {current_stats}")
        
        while temperature > min_temp and iteration < max_iterations:
            # Generate neighbor through SPR move
            neighbor_tree = self.perform_spr_move(self.current_tree)
            neighbor_stats = self.calculate_tree_statistics(neighbor_tree)
            neighbor_distance = self.calculate_distance(neighbor_stats)
            
            # Calculate acceptance probability
            delta = neighbor_distance - current_distance
            
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                # Accept the neighbor
                self.current_tree = neighbor_tree
                current_distance = neighbor_distance
                current_stats = neighbor_stats
                
                # Update best solution if better
                if neighbor_distance < self.best_distance:
                    self.best_tree = copy.deepcopy(neighbor_tree)
                    self.best_distance = neighbor_distance
                    print(f"Iteration {iteration}: New best distance: {self.best_distance:.6f}")
                    print(f"Best stats: {self.calculate_tree_statistics(self.best_tree)}")
            
            # Cool down
            temperature *= cooling_rate
            iteration += 1
            
            # Progress report
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}: temp={temperature:.6f}, "
                      f"current_dist={current_distance:.6f}, best_dist={self.best_distance:.6f}")
        
        print(f"\nOptimization completed!")
        print(f"Final best distance: {self.best_distance:.6f}")
        print(f"Final best stats: {self.calculate_tree_statistics(self.best_tree)}")
        print(f"Target vector: {self.target_vector}")
        
        return self.best_tree


def main():
    """Main function to run the optimization."""
    parser = argparse.ArgumentParser(description="Birth-Death Tree Simulated Annealing Optimizer")
    parser.add_argument("--birth_rate", type=float, default=1.0, help="Birth rate parameter")
    parser.add_argument("--death_rate", type=float, default=0.5, help="Death rate parameter")
    parser.add_argument("--bd_model", type=str, default="best_BCSTDCST",
                       choices=['best_BCSTDCST', 'best_BEXPDCST', 'best_BLINDCST',
                               'best_BCSTDEXP', 'best_BEXPDEXP', 'best_BLINDEXP',
                               'best_BCSTDLIN', 'best_BEXPDLIN', 'best_BLINDLIN'],
                       help="Birth-death model type")
    parser.add_argument("--target_colless", type=float, default=0.5,
                       help="Target normalized Colless imbalance")
    parser.add_argument("--target_gamma", type=float, default=0.0,
                       help="Target gamma statistic")
    parser.add_argument("--num_taxa", type=int, default=20, help="Number of taxa in the tree")
    parser.add_argument("--initial_temp", type=float, default=1.0, help="Initial temperature")
    parser.add_argument("--cooling_rate", type=float, default=0.95, help="Cooling rate")
    parser.add_argument("--max_iterations", type=int, default=10000, help="Maximum iterations")
    parser.add_argument("--output", type=str, help="Output file for the best tree (Newick format)")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = BDTreeOptimizer(
        birth_rate=args.birth_rate,
        death_rate=args.death_rate,
        bd_model=args.bd_model,
        target_colless=args.target_colless,
        target_gamma=args.target_gamma,
        num_taxa=args.num_taxa
    )
    
    # Run optimization
    best_tree = optimizer.simulated_annealing(
        initial_temp=args.initial_temp,
        cooling_rate=args.cooling_rate,
        max_iterations=args.max_iterations
    )
    
    # Output results
    if args.output:
        best_tree.write(path=args.output, schema="newick")
        print(f"Best tree saved to: {args.output}")
    else:
        print(f"\nBest tree (Newick format):")
        print(best_tree.as_string("newick"))


if __name__ == "__main__":
    main()