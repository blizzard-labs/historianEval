import re
from typing import Optional, Tuple, List
import math
import dendropy
import dendropy.calculate
import dendropy.calculate.treemeasure

class TreeNode:
    """Simple tree node class for Newick parsing"""
    def __init__(self, name: str = "", branch_length: float = 0.0):
        self.name = name
        self.branch_length = branch_length
        self.children = []
        self.parent = None
    
    def add_child(self, child):
        """Add a child node"""
        child.parent = self
        self.children.append(child)
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (terminal node)"""
        return len(self.children) == 0
    
    def is_internal(self) -> bool:
        """Check if node is internal (has children)"""
        return len(self.children) > 0
    
    def get_leaf_count(self) -> int:
        """Get number of leaves descending from this node"""
        if self.is_leaf():
            return 1
        return sum(child.get_leaf_count() for child in self.children)
    
    def __repr__(self):
        return f"TreeNode(name='{self.name}', children={len(self.children)})"


def parse_newick(newick_string: str) -> TreeNode:
    """
    Parse a Newick format string into a tree structure
    
    Args:
        newick_string: Newick formatted phylogenetic tree
        
    Returns:
        TreeNode: Root node of the parsed tree
    """
    # Clean the input string
    newick = newick_string.strip().rstrip(';')
    
    # Stack to keep track of current nodes
    node_stack = []
    current_node = None
    i = 0
    
    while i < len(newick):
        char = newick[i]
        
        if char == '(':
            # Start of a new internal node
            new_node = TreeNode()
            if current_node:
                current_node.add_child(new_node)
            node_stack.append(new_node)
            current_node = new_node
            
        elif char == ')':
            # End of current internal node
            if node_stack:
                current_node = node_stack.pop()
            # Check for branch length after closing parenthesis
            i += 1
            branch_info = ""
            while i < len(newick) and newick[i] not in "(),;":
                branch_info += newick[i]
                i += 1
            i -= 1  # Adjust for the outer loop increment
            
            # Parse branch length if present
            if ':' in branch_info:
                try:
                    current_node.branch_length = float(branch_info.split(':')[1])
                except (ValueError, IndexError):
                    current_node.branch_length = 0.0
                    
        elif char == ',':
            # Sibling separator - go back to parent
            if node_stack:
                current_node = node_stack[-1]
                
        elif char not in " \t\n":
            # Start of a taxon name or branch length
            taxon_info = ""
            while i < len(newick) and newick[i] not in "(),;":
                taxon_info += newick[i]
                i += 1
            i -= 1  # Adjust for the outer loop increment
            
            # Parse taxon name and branch length
            if ':' in taxon_info:
                parts = taxon_info.split(':')
                name = parts[0]
                try:
                    branch_length = float(parts[1])
                except ValueError:
                    branch_length = 0.0
            else:
                name = taxon_info
                branch_length = 0.0
            
            # Create leaf node
            leaf_node = TreeNode(name, branch_length)
            if current_node:
                current_node.add_child(leaf_node)
            else:
                # Single taxon tree
                current_node = leaf_node
        
        i += 1
    
    # Return the root (should be the last node in stack or current_node)
    return node_stack[0] if node_stack else current_node

"""
Notes...
The Colless index measures tree balance by summing the absolute differences
between the number of leaves in left and right subtrees at each internal node.

Colless Index = Σ |L_i - R_i| for all internal nodes i
where L_i and R_i are the number of leaves in left and right subtrees

A perfectly balanced binary tree has Colless index = 0
Higher values indicate more imbalanced trees
"""


def calculate_colless_index(root: TreeNode) -> int:
    """
    Calculate the Colless imbalance index for a phylogenetic tree
    
    The Colless index is the sum of absolute differences between
    the number of leaves in left and right subtrees for each internal node.
    
    Args:
        root: Root node of the tree
        
    Returns:
        int: Colless imbalance index (0 = perfectly balanced)
    """
    def colless_recursive(node: TreeNode) -> int:
        """Recursively calculate Colless index"""
        if node.is_leaf():
            return 0
        
        # Get leaf counts for all children
        child_leaf_counts = [child.get_leaf_count() for child in node.children]
        
        # For binary trees, calculate |L - R|
        if len(child_leaf_counts) == 2:
            left_count, right_count = child_leaf_counts
            current_imbalance = abs(left_count - right_count)
        else:
            # For non-binary trees, use variance-based approach
            # or sum of pairwise differences
            current_imbalance = 0
            for i in range(len(child_leaf_counts)):
                for j in range(i + 1, len(child_leaf_counts)):
                    current_imbalance += abs(child_leaf_counts[i] - child_leaf_counts[j])
        
        # Add imbalance from children
        child_imbalance = sum(colless_recursive(child) for child in node.children)
        
        return current_imbalance + child_imbalance
    
    return colless_recursive(root)


def calculate_normalized_colless_index(root: TreeNode) -> float:
    """
    Calculate the normalized Colless index (Colless index / maximum possible)
    
    For a binary tree with n leaves, the maximum Colless index is:
    (n-1)(n-2)/2 for a completely unbalanced tree
    
    Args:
        root: Root node of the tree
        
    Returns:
        float: Normalized Colless index (0.0 = perfectly balanced, 1.0 = maximally unbalanced)
    """
    colless_index = calculate_colless_index(root)
    n_leaves = root.get_leaf_count()
    
    if n_leaves <= 2:
        return 0.0
    
    # Maximum possible Colless index for n leaves
    max_colless = (n_leaves - 1) * (n_leaves - 2) // 2
    
    return colless_index / max_colless if max_colless > 0 else 0.0

def collapse_unary_nodes(tree):
    #For dendropy trees
    changed = True
    while changed:
        changed = False
        for node in list(tree.postorder_node_iter()):
            if not node.is_leaf() and len(node.child_nodes()) == 1 and node != tree.seed_node:
                parent = node.parent_node
                child = node.child_nodes()[0]
                if parent:
                    parent.remove_child(node)
                    parent.add_child(child)
                    changed = True
    # Special case: root with one child
    root = tree.seed_node
    while len(root.child_nodes()) == 1:
        child = root.child_nodes()[0]
        tree.seed_node = child
        root = child

def resolve_polytomies(tree):
    """Resolve polytomies in dendropy tree to make it strictly bifurcating"""
    for node in tree.postorder_node_iter():
        if not node.is_leaf() and len(node.child_nodes()) > 2:
            # Convert polytomy to series of binary splits
            children = list(node.child_nodes())
            # Remove all children from the node
            for child in children:
                node.remove_child(child)
            
            # Add children back in binary fashion
            current_node = node
            for i in range(len(children) - 2):
                # Create new internal node
                new_internal = tree.node_factory()
                current_node.add_child(new_internal)
                current_node.add_child(children[i])
                current_node = new_internal
            
            # Add the last two children
            current_node.add_child(children[-2])
            current_node.add_child(children[-1])

def is_strictly_bifurcating(tree):
    """Check if tree is strictly bifurcating (all internal nodes have exactly 2 children)"""
    for node in tree.preorder_node_iter():
        if not node.is_leaf():
            if len(node.child_nodes()) != 2:
                return False
    return True

def analyze_tree_balance(newick_string: str) -> dict:
    """
    Comprehensive tree balance analysis
    
    Args:
        newick_string: Newick formatted tree
        
    Returns:
        dict: Analysis results including various balance metrics
    """
    # Parse the tree
    root = parse_newick(newick_string)
    tree = dendropy.Tree.get_from_string(newick_string, schema="newick")
    collapse_unary_nodes(tree)
    
    # Calculate metrics
    n_leaves = root.get_leaf_count()
    colless_index = calculate_colless_index(root)
    #normalized_colless = calculate_normalized_colless_index(root)
    
    if is_strictly_bifurcating(tree):
        normalized_colless = dendropy.calculate.treemeasure.colless_tree_imbalance(tree)
    else:
        tree_copy = dendropy.Tree(tree)
        resolve_polytomies(tree_copy)
        normalized_colless = dendropy.calculate.treemeasure.colless_tree_imbalance(tree_copy)
    
    # Calculate tree depth
    def get_max_depth(node: TreeNode, depth: int = 0) -> int:
        if node.is_leaf():
            return depth
        return max(get_max_depth(child, depth + 1) for child in node.children)
    
    max_depth = get_max_depth(root)
    
    # Calculate number of internal nodes
    def count_internal_nodes(node: TreeNode) -> int:
        if node.is_leaf():
            return 0
        return 1 + sum(count_internal_nodes(child) for child in node.children)
    
    n_internal = count_internal_nodes(root)
    
    # Determine balance category
    if normalized_colless < 0.2:
        balance_category = "Well-balanced"
    elif normalized_colless < 0.5:
        balance_category = "Moderately balanced"
    elif normalized_colless < 0.8:
        balance_category = "Imbalanced"
    else:
        balance_category = "Highly imbalanced"
    
    return {
        'newick_string': newick_string,
        'n_leaves': n_leaves,
        'n_internal_nodes': n_internal,
        'max_depth': max_depth,
        'colless_index': colless_index,
        'normalized_colless_index': normalized_colless,
        'balance_category': balance_category,
        'is_binary': all(len(node.children) <= 2 for node in get_all_nodes(root) if node.is_internal())
    }


def get_all_nodes(root: TreeNode) -> List[TreeNode]:
    """Get all nodes in the tree"""
    nodes = [root]
    for child in root.children:
        nodes.extend(get_all_nodes(child))
    return nodes


def compare_tree_balance(newick_trees: List[str]) -> List[dict]:
    """
    Compare balance metrics across multiple trees
    
    Args:
        newick_trees: List of Newick formatted trees
        
    Returns:
        List[dict]: Balance analysis for each tree, sorted by Colless index
    """
    results = []
    
    for i, tree in enumerate(newick_trees):
        try:
            analysis = analyze_tree_balance(tree)
            analysis['tree_id'] = f"Tree_{i+1}"
            results.append(analysis)
        except Exception as e:
            print(f"Error analyzing tree {i+1}: {e}")
    
    # Sort by Colless index (most balanced first)
    results.sort(key=lambda x: x['normalized_colless_index'])
    
    return results


# Example usage and testing
def main():
    """Test the Colless index calculator with various tree structures"""
    
    # Test trees with different balance properties
    test_trees = [
        # Perfectly balanced binary tree
        "((A:1,B:1):1,(C:1,D:1):1);",
        
        # Your original unbalanced tree
        "(((Taxon1:0.01,(Taxon2:0.2,Taxon3:0.3):0.01):0.21,Taxon4:0.23):0.44);",
        
        # Balanced version
        "((Taxon1:0.01,(Taxon2:0.2,Taxon3:0.3):0.01):0.21,Taxon4:0.23);",
        
        # Completely unbalanced (caterpillar tree)
        "(((A:1,B:1):1,C:1):1,D:1);",
        
        # Another balanced tree
        "((A:1,B:1):1,(C:1,D:1):1);",
        
        # 6-taxon examples
        "(((A:1,B:1):1,(C:1,D:1):1):1,(E:1,F:1):1);",  # Balanced
        "((((A:1,B:1):1,C:1):1,D:1):1,(E:1,F:1):1);",  # Semi-balanced
        "(((((A:1,B:1):1,C:1):1,D:1):1,E:1):1,F:1);",  # Unbalanced
    ]
    
    print("=== Colless Imbalance Index Analysis ===\n")
    
    # Analyze each tree
    results = compare_tree_balance(test_trees)
    
    # Print results
    print(f"{'Tree':<12} {'Leaves':<7} {'Colless':<8} {'Normalized':<11} {'Balance Category':<20} {'Tree Structure'}")
    print("-" * 95)
    
    for result in results:
        tree_structure = result['newick_string'][:50] + "..." if len(result['newick_string']) > 50 else result['newick_string']
        print(f"{result['tree_id']:<12} {result['n_leaves']:<7} {result['colless_index']:<8} "
              f"{result['normalized_colless_index']:<11.3f} {result['balance_category']:<20} {tree_structure}")
    
    print("\n=== Detailed Analysis of Your Trees ===\n")
    
    # Detailed analysis of your specific trees
    your_trees = [
        ("Original (Unbalanced)", "(((Taxon1:0.01,(Taxon2:0.2,Taxon3:0.3):0.01):0.21,Taxon4:0.23):0.44);"),
        ("Balanced Version", "((Taxon1:0.01,(Taxon2:0.2,Taxon3:0.3):0.01):0.21,Taxon4:0.23);")
    ]
    
    for name, tree in your_trees:
        print(f"{name}:")
        analysis = analyze_tree_balance(tree)
        
        print(f"  Newick: {tree}")
        print(f"  Number of leaves: {analysis['n_leaves']}")
        print(f"  Number of internal nodes: {analysis['n_internal_nodes']}")
        print(f"  Maximum depth: {analysis['max_depth']}")
        print(f"  Colless index: {analysis['colless_index']}")
        print(f"  Normalized Colless index: {analysis['normalized_colless_index']:.3f}")
        print(f"  Balance category: {analysis['balance_category']}")
        print(f"  Is binary tree: {analysis['is_binary']}")
        print()


if __name__ == "__main__":
    main()


# Additional utility functions for specific use cases

def quick_colless_index(newick_string: str) -> int:
    """
    Quick calculation of Colless index for immediate use
    
    Args:
        newick_string: Newick formatted tree
        
    Returns:
        int: Colless imbalance index
    """
    try:
        root = parse_newick(newick_string)
        return calculate_colless_index(root)
    except Exception as e:
        print(f"Error calculating Colless index: {e}")
        return -1


def is_tree_balanced(newick_string: str, threshold: float = 0.3) -> bool:
    """
    Simple check if a tree is reasonably balanced
    
    Args:
        newick_string: Newick formatted tree
        threshold: Normalized Colless threshold (default 0.3)
        
    Returns:
        bool: True if tree is considered balanced
    """
    try:
        root = parse_newick(newick_string)
        normalized_colless = calculate_normalized_colless_index(root)
        return normalized_colless <= threshold
    except Exception as e:
        print(f"Error checking tree balance: {e}")
        return False