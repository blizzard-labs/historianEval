#!/usr/bin/env python3
"""
Phylogenetic Tree Consensus Analysis Script (RF-Length Version)

This script:
1. Reads all phylogenetic trees from a specified folder
2. Generates a majority consensus tree using split frequencies
3. Calculates RF-length distances from each tree to the consensus
4. Outputs results to a CSV file

Requirements:
- ete3: pip install ete3
- pandas: pip install pandas
"""

import os
import sys
import pandas as pd
from ete3 import Tree
import argparse
from pathlib import Path
import numpy as np
from collections import defaultdict

def load_trees_from_folder(folder_path, extensions=('.nwk', '.newick', '.tre', '.tree')):
    """
    Load all phylogenetic trees from a folder.
    
    Args:
        folder_path (str): Path to folder containing tree files
        extensions (tuple): File extensions to consider as tree files
    
    Returns:
        dict: Dictionary with filename as key and Tree object as value
    """
    trees = {}
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    tree_files = []
    for ext in extensions:
        tree_files.extend(folder.glob(f'*{ext}'))
    
    if not tree_files:
        raise ValueError(f"No tree files found in {folder_path} with extensions {extensions}")
    
    print(f"Found {len(tree_files)} tree files")
    
    for tree_file in tree_files:
        try:
            tree = Tree(str(tree_file), format=1)
            trees[tree_file.name] = tree
            print(f"Loaded: {tree_file.name}")
        except Exception as e:
            print(f"Error loading {tree_file.name}: {e}")
    
    return trees

def get_splits(tree):
    """
    Get all splits (bipartitions) from a tree.
    
    Args:
        tree (Tree): Input tree
        
    Returns:
        set: Set of splits, where each split is a frozenset of leaf names
    """
    splits = set()
    all_leaves = set(leaf.name for leaf in tree.get_leaves())
    
    for node in tree.traverse():
        if not node.is_leaf() and not node.is_root():
            # Get leaves in this subtree
            subtree_leaves = set(leaf.name for leaf in node.get_leaves())
            # Skip trivial splits (single leaves or all leaves)
            if len(subtree_leaves) > 1 and len(subtree_leaves) < len(all_leaves):
                # Create split as frozenset (to make it hashable)
                split = frozenset(subtree_leaves)
                splits.add(split)
    
    return splits

def normalize_tree_topology(tree):
    """
    Normalize a tree by replacing leaf names with generic labels based on topology.
    
    Args:
        tree (Tree): Input tree
    
    Returns:
        tuple: (normalized_tree, mapping_dict)
    """
    # Create a copy of the tree
    normalized_tree = tree.copy()
    
    # Get leaves in a consistent order (e.g., by traversal order)
    leaves = []
    for leaf in normalized_tree.traverse():
        if leaf.is_leaf():
            leaves.append(leaf)
    
    # Sort leaves by their position in the tree (deterministic ordering)
    leaves.sort(key=lambda x: x.get_distance(normalized_tree))
    
    # Create mapping from original names to generic labels
    mapping = {}
    for i, leaf in enumerate(leaves):
        generic_name = f"T{i+1:03d}"
        mapping[leaf.name] = generic_name
        leaf.name = generic_name
    
    return normalized_tree, mapping

def get_topological_splits(tree):
    """
    Get all topological splits from a normalized tree.
    
    Args:
        tree (Tree): Normalized tree
        
    Returns:
        set: Set of topological splits
    """
    splits = set()
    total_leaves = len([leaf for leaf in tree.get_leaves()])
    
    for node in tree.traverse():
        if not node.is_leaf() and not node.is_root():
            # Get leaves in this subtree
            subtree_leaves = [leaf.name for leaf in node.get_leaves()]
            subtree_size = len(subtree_leaves)
            
            # Skip trivial splits
            if subtree_size > 1 and subtree_size < total_leaves:
                # Create a canonical representation of the split
                # Use the size and relative position instead of actual names
                split_signature = (subtree_size, total_leaves - subtree_size)
                splits.add(split_signature)
    
    return splits




def get_proportional_signature(node, tree):
    """
    Create signature based on proportional relationships.
    
    Args:
        node: Tree node to create signature for
        tree: Complete tree object
        
    Returns:
        tuple: (proportion, sister_signature, depth_level)
    """
    total_leaves = len([leaf for leaf in tree.get_leaves()])
    subtree_size = len([leaf for leaf in node.get_leaves()])
    
    # Get proportional size (what fraction of the tree this clade represents)
    proportion = subtree_size / total_leaves
    
    # Get sister clade proportions (what this clade is grouped with)
    parent = node.up
    if parent:
        sister_proportions = []
        for sibling in parent.children:
            if sibling != node:
                sister_size = len([leaf for leaf in sibling.get_leaves()])
                sister_proportions.append(sister_size / total_leaves)
        sister_signature = tuple(sorted(sister_proportions))
    else:
        sister_signature = ()
    
    # Add depth information for better discrimination
    depth_level = len(node.get_ancestors())
    
    # Round to avoid floating point matching issues
    proportion = round(proportion, 3)
    sister_signature = tuple(round(p, 3) for p in sister_signature)
    
    return (proportion, sister_signature, depth_level)


def generate_topological_consensus_tree(trees, threshold=0.5):
    """
    Generate a majority consensus tree based on topology/shape rather than specific taxa.
    UPDATED VERSION using proportional signatures.
    """
    tree_list = list(trees.values())
    
    if len(tree_list) == 0:
        raise ValueError("No trees provided for consensus")
    
    print(f"Generating topological consensus tree with threshold {threshold}...")
    
    # Normalize all trees and collect their topological patterns
    normalized_trees = []
    all_tree_sizes = []
    
    for tree_name, tree in trees.items():
        normalized_tree, mapping = normalize_tree_topology(tree)
        normalized_trees.append((tree_name, normalized_tree, mapping))
        num_leaves = len([leaf for leaf in tree.get_leaves()])
        all_tree_sizes.append(num_leaves)
        print(f"Tree {tree_name}: {num_leaves} leaves")
    
    # Group trees by size ranges instead of requiring exact matches
    from collections import defaultdict
    size_groups = defaultdict(list)
    
    for name, tree, mapping in normalized_trees:
        tree_size = len([leaf for leaf in tree.get_leaves()])
        # Group trees into size categories (allows some flexibility)
        size_category = (tree_size // 10) * 10  # Group by tens
        size_groups[size_category].append((name, tree, mapping, tree_size))
    
    # Use the largest group
    largest_group = max(size_groups.values(), key=len)
    print(f"Using {len(largest_group)} trees from size group for consensus")
    
    # Count topological split patterns using proportional signatures
    split_counts = defaultdict(int)
    split_branch_lengths = defaultdict(list)
    split_examples = defaultdict(list)  # Store examples for debugging
    
    for tree_name, normalized_tree, mapping, tree_size in largest_group:
        # Get proportional signatures for all internal nodes
        for node in normalized_tree.traverse():
            if not node.is_leaf() and not node.is_root():
                # Get proportional signature
                signature = get_proportional_signature(node, normalized_tree)
                
                # Only consider meaningful splits (not too small or too large)
                proportion = signature[0]
                if 0.1 <= proportion <= 0.9:  # Ignore very small or very large clades
                    split_counts[signature] += 1
                    
                    # Store branch length
                    branch_length = node.dist if hasattr(node, 'dist') else 0.0
                    split_branch_lengths[signature].append(branch_length)
                    
                    # Store example for debugging
                    split_examples[signature].append((tree_name, proportion, len(node.get_leaves())))
    
    # Filter splits by threshold
    num_trees = len(largest_group)
    consensus_splits = []
    
    print(f"\nTopological splits found:")
    for signature, count in sorted(split_counts.items(), key=lambda x: x[1], reverse=True):
        frequency = count / num_trees
        if frequency >= threshold:
            avg_branch_length = np.mean(split_branch_lengths[signature]) if split_branch_lengths[signature] else 0.0
            consensus_splits.append((signature, frequency, avg_branch_length))
            
            # Debug information
            proportion, sister_sig, depth = signature
            examples = split_examples[signature][:3]  # Show first 3 examples
            print(f"  Split proportion {proportion:.3f}, sisters {sister_sig}, depth {depth}")
            print(f"    Frequency: {frequency:.3f}, avg branch length: {avg_branch_length:.6f}")
            print(f"    Examples: {examples}")
    
    print(f"\nFound {len(consensus_splits)} consensus topological splits")
    
    if not consensus_splits:
        print("No consensus splits found, using first tree as template")
        return largest_group[0][1].copy()
    
    # Build consensus tree using proportional signatures
    consensus_tree = build_consensus_tree_from_proportional_splits(
        largest_group, consensus_splits, trees
    )
    
    return consensus_tree


def build_consensus_tree_from_proportional_splits(largest_group, consensus_splits, original_trees):
    """
    Build a consensus tree from proportional splits.
    
    Args:
        largest_group: List of (tree_name, normalized_tree, mapping, tree_size) tuples
        consensus_splits: List of (signature, frequency, branch_length) tuples
        original_trees: Dictionary of original trees
        
    Returns:
        Tree: Consensus tree
    """
    # Find the tree with the most consensus splits as template
    best_template = None
    best_match_count = 0
    
    for tree_name, normalized_tree, mapping, tree_size in largest_group:
        match_count = 0
        for signature, frequency, branch_length in consensus_splits:
            # Check if this tree contains a split matching this signature
            for node in normalized_tree.traverse():
                if not node.is_leaf() and not node.is_root():
                    node_signature = get_proportional_signature(node, normalized_tree)
                    if signatures_match(node_signature, signature):
                        match_count += 1
                        break
        
        if match_count > best_match_count:
            best_match_count = match_count
            best_template = (tree_name, normalized_tree, mapping, tree_size)
    
    if best_template is None:
        best_template = largest_group[0]
    
    print(f"Using {best_template[0]} as template (matches {best_match_count} consensus splits)")
    
    # Create consensus tree based on template
    consensus_tree = best_template[1].copy()
    
    # Update branch lengths with consensus values
    for node in consensus_tree.traverse():
        if not node.is_leaf() and not node.is_root():
            node_signature = get_proportional_signature(node, consensus_tree)
            
            # Find matching consensus split
            for signature, frequency, avg_branch_length in consensus_splits:
                if signatures_match(node_signature, signature):
                    node.dist = avg_branch_length
                    break
    
    # Map back to meaningful taxon names from original trees
    # Use the taxon names from the largest original tree
    largest_original_tree = max(original_trees.values(), 
                               key=lambda t: len([leaf for leaf in t.get_leaves()]))
    original_leaf_names = [leaf.name for leaf in largest_original_tree.get_leaves()]
    
    # Assign names to consensus tree leaves
    consensus_leaves = [leaf for leaf in consensus_tree.get_leaves()]
    for i, leaf in enumerate(consensus_leaves):
        if i < len(original_leaf_names):
            leaf.name = original_leaf_names[i]
        else:
            leaf.name = f"Taxa_{i+1}"
    
    return consensus_tree


def signatures_match(sig1, sig2, tolerance=0.05):
    """
    Check if two proportional signatures match within tolerance.
    
    Args:
        sig1, sig2: Signatures to compare (proportion, sister_signature, depth)
        tolerance: Tolerance for proportion matching
        
    Returns:
        bool: True if signatures match
    """
    prop1, sister1, depth1 = sig1
    prop2, sister2, depth2 = sig2
    
    # Check proportion match
    if abs(prop1 - prop2) > tolerance:
        return False
    
    # Check depth match (can be flexible)
    if abs(depth1 - depth2) > 1:
        return False
    
    # Check sister signature match (more flexible)
    if len(sister1) != len(sister2):
        return False
    
    if len(sister1) > 0:
        # Check if sister proportions are similar
        for s1, s2 in zip(sister1, sister2):
            if abs(s1 - s2) > tolerance:
                return False
    
    return True

def generate_majority_consensus_tree(trees, threshold=0.5, use_topology=False):
    """
    Generate a majority consensus tree from a collection of trees.
    
    Args:
        trees (dict): Dictionary of Tree objects
        threshold (float): Minimum frequency for a split to be included (default: 0.5 for majority)
        use_topology (bool): If True, use topological consensus instead of taxa-based
    
    Returns:
        Tree: Majority consensus tree
    """
    if use_topology:
        return generate_topological_consensus_tree(trees, threshold)
    
    tree_list = list(trees.values())
    
    if len(tree_list) == 0:
        raise ValueError("No trees provided for consensus")
    
    # Get all leaf names from the first tree
    leaf_names = set(leaf.name for leaf in tree_list[0].get_leaves())
    
    # Verify all trees have the same leaf set
    for i, tree in enumerate(tree_list[1:], 1):
        current_leaves = set(leaf.name for leaf in tree.get_leaves())
        if current_leaves != leaf_names:
            print(f"Warning: Tree {i} has different leaf set")
    
    print(f"Generating majority consensus tree with threshold {threshold}...")
    
    # Count split frequencies
    split_counts = defaultdict(int)
    split_branch_lengths = defaultdict(list)
    
    for tree_name, tree in trees.items():
        splits = get_splits(tree)
        for split in splits:
            split_counts[split] += 1
            
            # Find the node corresponding to this split and get its branch length
            for node in tree.traverse():
                if not node.is_leaf() and not node.is_root():
                    node_leaves = set(leaf.name for leaf in node.get_leaves())
                    if frozenset(node_leaves) == split:
                        branch_length = node.dist if hasattr(node, 'dist') else 0.0
                        split_branch_lengths[split].append(branch_length)
                        break
    
    # Filter splits by threshold
    num_trees = len(tree_list)
    consensus_splits = []
    
    for split, count in split_counts.items():
        frequency = count / num_trees
        if frequency >= threshold:
            avg_branch_length = np.mean(split_branch_lengths[split]) if split_branch_lengths[split] else 0.0
            consensus_splits.append((split, frequency, avg_branch_length))
            print(f"Split {list(split)[:3]}... frequency: {frequency:.3f}, avg branch length: {avg_branch_length:.6f}")
    
    print(f"Found {len(consensus_splits)} consensus splits")
    
    # Build consensus tree
    consensus_tree = build_consensus_tree_from_splits(leaf_names, consensus_splits)
    
    return consensus_tree

def build_consensus_tree_from_splits(leaf_names, consensus_splits):
    """
    Build a consensus tree from a set of splits.
    
    Args:
        leaf_names (set): Set of leaf names
        consensus_splits (list): List of tuples (split, frequency, branch_length)
    
    Returns:
        Tree: Consensus tree
    """
    # Start with a star tree (all leaves connected to root)
    consensus_tree = Tree()
    consensus_tree.name = "root"
    
    # Add all leaves to the root
    leaf_nodes = {}
    for leaf_name in leaf_names:
        leaf_node = Tree()
        leaf_node.name = leaf_name
        leaf_node.dist = 0.0
        consensus_tree.add_child(leaf_node)
        leaf_nodes[leaf_name] = leaf_node
    
    # Sort splits by frequency (highest first) to resolve conflicts
    consensus_splits.sort(key=lambda x: x[1], reverse=True)
    
    # Add splits to the tree
    for split, frequency, branch_length in consensus_splits:
        try:
            add_split_to_tree(consensus_tree, split, branch_length, leaf_nodes)
        except Exception as e:
            print(f"Warning: Could not add split {list(split)[:3]}...: {e}")
    
    return consensus_tree

def add_split_to_tree(tree, split, branch_length, leaf_nodes):
    """
    Add a split to an existing tree.
    
    Args:
        tree (Tree): Tree to modify
        split (frozenset): Split to add
        branch_length (float): Branch length for the split
        leaf_nodes (dict): Dictionary mapping leaf names to nodes
    """
    # Find the leaves that should be grouped together
    split_leaves = list(split)
    
    # Find the current parent nodes of these leaves
    current_parents = set()
    split_nodes = []
    
    for leaf_name in split_leaves:
        if leaf_name in leaf_nodes:
            leaf_node = leaf_nodes[leaf_name]
            current_parents.add(leaf_node.up)
            split_nodes.append(leaf_node)
    
    # If all leaves are already under the same parent, the split already exists
    if len(current_parents) == 1:
        return
    
    # Find the lowest common ancestor of all split leaves
    if len(split_nodes) < 2:
        return
    
    lca = split_nodes[0]
    for node in split_nodes[1:]:
        lca = lca.get_common_ancestor(node)
    
    # Check if we can create this split without conflicting with existing structure
    lca_leaves = set(leaf.name for leaf in lca.get_leaves())
    
    # If the LCA contains exactly the split leaves, the split already exists
    if lca_leaves == split:
        return
    
    # If the LCA contains more than the split leaves, we need to create a new internal node
    if len(lca_leaves) > len(split):
        # Create new internal node
        new_internal = Tree()
        new_internal.name = f"internal_{len(split)}"
        new_internal.dist = branch_length
        
        # Find which children of LCA should be moved to the new internal node
        children_to_move = []
        for child in lca.children:
            child_leaves = set(leaf.name for leaf in child.get_leaves())
            if child_leaves.issubset(split):
                children_to_move.append(child)
        
        # Only proceed if we have children to move and it makes sense
        if len(children_to_move) > 1:
            # Remove children from LCA and add them to new internal node
            for child in children_to_move:
                child.detach()
                new_internal.add_child(child)
            
            # Add new internal node to LCA
            lca.add_child(new_internal)

def calculate_rf_distances(trees, consensus_tree):
    """
    Calculate Robinson-Foulds distances with branch lengths from each tree to the consensus tree.
    
    Args:
        trees (dict): Dictionary of Tree objects
        consensus_tree (Tree): Consensus tree
    
    Returns:
        dict: Dictionary with filename as key and RF-length distance as value
    """
    rf_distances = {}
    
    print("Calculating RF-length distances to consensus tree...")
    
    for name, tree in trees.items():
        try:
            rf_result = tree.robinson_foulds(consensus_tree, unrooted_trees=True)
            rf_distance = rf_result[1]  # Weighted RF distance (includes branch lengths)
            rf_distances[name] = rf_distance
            print(f"{name}: RF-length distance = {rf_distance:.6f}")
        except Exception as e:
            print(f"Error calculating RF-length distance for {name}: {e}")
            rf_distances[name] = None
    
    return rf_distances

def save_results_to_csv(rf_distances, output_file):
    """
    Save RF-length distances to CSV file.
    
    Args:
        rf_distances (dict): Dictionary of RF-length distances
        output_file (str): Output CSV file path
    """
    
    
    try:
        df = pd.read_csv(output_file)
        values = [0] * len(df['filename'])
        df_filenames = [k.split(".")[0] for k in df['filename'].values]
        print(df_filenames)
        for tree_filename, distance in rf_distances.items():
            f = tree_filename.split(".")[0] + "_AA"
            if f in df_filenames:
                print(distance)
                values[df_filenames.index(f)] = distance
        
        print('aslkdhflkasdf')
        print(values)
        
        df['rf_length_distance'] = values
        print(df)
        df.to_csv(output_file, index=False)
        print(f"RF-length distances updated in {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error finding output file: {e}")
    
def print_statistics(rf_distances):
    """
    Print summary statistics of RF-length distances.
    
    Args:
        rf_distances (dict): Dictionary of RF-length distances
    """
    valid_distances = [d for d in rf_distances.values() if d is not None]
    
    if not valid_distances:
        print("No valid RF-length distances calculated")
        return
    
    print("\nRF-Length Distance Statistics:")
    print(f"Number of trees: {len(valid_distances)}")
    print(f"Mean RF-length distance: {np.mean(valid_distances):.6f}")
    print(f"Median RF-length distance: {np.median(valid_distances):.6f}")
    print(f"Standard deviation: {np.std(valid_distances):.6f}")
    print(f"Min RF-length distance: {min(valid_distances):.6f}")
    print(f"Max RF-length distance: {max(valid_distances):.6f}")

def main():
    #Sample argument: python src/model_gen_aa/treedist.py data/misc/orthomam_small -o data/model_gen/V1_sample_aa/protein_evolution_parameters_with_rates.csv --save-consensus consense.tree --threshold 0.6 --topology
    
    parser = argparse.ArgumentParser(
        description="Generate majority consensus tree and calculate RF-length distances"
    )
    parser.add_argument(
        "tree_folder",
        help="Path to folder containing phylogenetic tree files"
    )
    parser.add_argument(
        "-o", "--output",
        default="rf_length_distances.csv",
        help="Output CSV file (default: rf_length_distances.csv)"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=['.nwk', '.newick', '.tre', '.tree'],
        help="File extensions to consider as tree files"
    )
    parser.add_argument(
        "--save-consensus",
        help="Save consensus tree to file (optional)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum frequency for splits to be included in consensus (default: 0.5)"
    )
    parser.add_argument(
        "--topology",
        action="store_true",
        help="Use topological consensus (tree shape) instead of taxa-based consensus"
    )
    
    args = parser.parse_args()
    
    try:
        # Load trees
        print(f"Loading trees from {args.tree_folder}...")
        trees = load_trees_from_folder(args.tree_folder, tuple(args.extensions))
        
        if len(trees) < 2:
            raise ValueError("Need at least 2 trees to generate consensus")
        
        # Generate majority consensus tree
        print("\nGenerating majority consensus tree...")
        use_topology = getattr(args, 'topology', False)
        consensus_tree = generate_majority_consensus_tree(trees, args.threshold, use_topology)
        
        # Save consensus tree if requested
        if args.save_consensus:
            consensus_tree.write(outfile=args.save_consensus)
            print(f"Consensus tree saved to {args.save_consensus}")
        
        # Calculate RF-length distances
        print("\nCalculating RF-length distances...")
        rf_distances = calculate_rf_distances(trees, consensus_tree)
        
        # Print statistics
        print_statistics(rf_distances)
        
        # Save results
        print(f"\nSaving results to {args.output}...")
        save_results_to_csv(rf_distances, args.output)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()