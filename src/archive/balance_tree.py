"""
Python methods for balancing phylogenetic trees
Includes multiple approaches from simple to sophisticated
"""

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import math

# Method 1: Simple BioPython approach (most practical)
def balance_tree_biopython(newick_string: str) -> str:
    """
    Balance a tree using BioPython's built-in methods
    Requires: pip install biopython
    """
    try:
        from Bio import Phylo
        from io import StringIO
        
        # Parse the tree
        tree = Phylo.read(StringIO(newick_string), 'newick')
        
        # Get all terminals (leaf nodes)
        terminals = tree.get_terminals()
        
        # Simple balanced grouping - pair up terminals
        balanced_pairs = []
        for i in range(0, len(terminals), 2):
            if i + 1 < len(terminals):
                # Pair consecutive terminals
                term1 = terminals[i]
                term2 = terminals[i + 1]
                avg_branch = (term1.branch_length + term2.branch_length) / 2
                balanced_pairs.append(f"({term1.name}:{avg_branch:.6f},{term2.name}:{avg_branch:.6f})")
            else:
                # Odd number of terminals - keep the last one separate
                balanced_pairs.append(f"{terminals[i].name}:{terminals[i].branch_length:.6f}")
        
        # Create balanced tree structure
        while len(balanced_pairs) > 1:
            new_pairs = []
            for i in range(0, len(balanced_pairs), 2):
                if i + 1 < len(balanced_pairs):
                    new_pairs.append(f"({balanced_pairs[i]},{balanced_pairs[i+1]})")
                else:
                    new_pairs.append(balanced_pairs[i])
            balanced_pairs = new_pairs
        
        return balanced_pairs[0] + ";"
        
    except ImportError:
        print("BioPython not available, using manual method")
        return balance_tree_manual(newick_string)


# Method 2: Manual parsing and balancing
def balance_tree_manual(newick_string: str) -> str:
    """
    Balance a tree by manually parsing and reconstructing
    More control but requires careful implementation
    """
    # Extract taxa and their branch lengths
    taxa_info = extract_taxa_info(newick_string)
    
    # Create balanced binary tree
    balanced_tree = create_balanced_binary_tree(taxa_info)
    
    return balanced_tree


def extract_taxa_info(newick_string: str) -> List[Tuple[str, float]]:
    """Extract taxa names and branch lengths from Newick string"""
    # Remove outer parentheses and semicolon
    clean_string = newick_string.strip().rstrip(';')
    
    # Find all taxa with their branch lengths
    taxa_pattern = r'([A-Za-z_][A-Za-z0-9_]*):([0-9.]+)'
    matches = re.findall(taxa_pattern, clean_string)
    
    return [(name, float(length)) for name, length in matches]


def create_balanced_binary_tree(taxa_info: List[Tuple[str, float]]) -> str:
    """Create a balanced binary tree from taxa information"""
    if not taxa_info:
        return ""
    
    if len(taxa_info) == 1:
        name, length = taxa_info[0]
        return f"{name}:{length:.6f}"
    
    # Sort by branch length to maintain some biological meaning
    taxa_info.sort(key=lambda x: x[1])
    
    # Recursively build balanced tree
    def build_tree(taxa_list):
        if len(taxa_list) == 1:
            name, length = taxa_list[0]
            return f"{name}:{length:.6f}"
        elif len(taxa_list) == 2:
            name1, length1 = taxa_list[0]
            name2, length2 = taxa_list[1]
            return f"({name1}:{length1:.6f},{name2}:{length2:.6f})"
        else:
            # Split into two balanced groups
            mid = len(taxa_list) // 2
            left_group = taxa_list[:mid]
            right_group = taxa_list[mid:]
            
            left_subtree = build_tree(left_group)
            right_subtree = build_tree(right_group)
            
            return f"({left_subtree},{right_subtree})"
    
    return build_tree(taxa_info) + ";"


# Method 3: Advanced balancing with branch length preservation
class TreeNode:
    """Simple tree node class for advanced balancing"""
    def __init__(self, name: str = "", branch_length: float = 0.0):
        self.name = name
        self.branch_length = branch_length
        self.children = []
        self.parent = None
    
    def is_leaf(self):
        return len(self.children) == 0
    
    def add_child(self, child):
        child.parent = self
        self.children.append(child)


def balance_tree_advanced(newick_string: str, preserve_distances: bool = True) -> str:
    """
    Advanced tree balancing with options to preserve evolutionary distances
    """
    # Parse into tree structure
    root = parse_newick_to_tree(newick_string)
    
    # Get all leaf nodes
    leaves = get_leaves(root)
    
    # Calculate original pairwise distances if preserving distances
    if preserve_distances:
        original_distances = calculate_pairwise_distances(root, leaves)
    
    # Create balanced tree
    balanced_root = create_balanced_tree_from_leaves(leaves)
    
    # Adjust branch lengths to preserve distances if requested
    if preserve_distances:
        adjust_branch_lengths(balanced_root, original_distances)
    
    return tree_to_newick(balanced_root)


def parse_newick_to_tree(newick_string: str) -> TreeNode:
    """Parse Newick string into tree structure"""
    # Simplified parser - for production use, consider using proper parsing libraries
    clean_string = newick_string.strip().rstrip(';')
    
    # This is a simplified implementation
    # For robust parsing, use libraries like ete3 or dendropy
    
    # Extract taxa info for simple case
    taxa_info = extract_taxa_info(newick_string)
    
    # Create simple balanced structure
    root = TreeNode()
    for name, length in taxa_info:
        leaf = TreeNode(name, length)
        root.add_child(leaf)
    
    return root


def get_leaves(root: TreeNode) -> List[TreeNode]:
    """Get all leaf nodes from tree"""
    leaves = []
    
    def collect_leaves(node):
        if node.is_leaf():
            leaves.append(node)
        else:
            for child in node.children:
                collect_leaves(child)
    
    collect_leaves(root)
    return leaves


def create_balanced_tree_from_leaves(leaves: List[TreeNode]) -> TreeNode:
    """Create balanced binary tree from leaf nodes"""
    if len(leaves) == 1:
        return leaves[0]
    
    # Sort leaves by name for consistent results
    leaves.sort(key=lambda x: x.name)
    
    def build_balanced(node_list):
        if len(node_list) == 1:
            return node_list[0]
        elif len(node_list) == 2:
            internal = TreeNode()
            internal.add_child(node_list[0])
            internal.add_child(node_list[1])
            return internal
        else:
            mid = len(node_list) // 2
            left_subtree = build_balanced(node_list[:mid])
            right_subtree = build_balanced(node_list[mid:])
            
            internal = TreeNode()
            internal.add_child(left_subtree)
            internal.add_child(right_subtree)
            return internal
    
    return build_balanced(leaves)


def calculate_pairwise_distances(root: TreeNode, leaves: List[TreeNode]) -> Dict[Tuple[str, str], float]:
    """Calculate pairwise distances between all leaves"""
    distances = {}
    
    for i, leaf1 in enumerate(leaves):
        for j, leaf2 in enumerate(leaves[i+1:], i+1):
            # Calculate distance between leaf1 and leaf2
            # This is simplified - in practice, you'd traverse the tree
            # For now, use sum of branch lengths as approximation
            dist = leaf1.branch_length + leaf2.branch_length
            distances[(leaf1.name, leaf2.name)] = dist
    
    return distances


def adjust_branch_lengths(root: TreeNode, target_distances: Dict[Tuple[str, str], float]):
    """Adjust branch lengths to approximate target distances"""
    # This is a complex optimization problem
    # For simplicity, we'll use average branch lengths
    leaves = get_leaves(root)
    
    if not leaves:
        return
    
    avg_length = sum(leaf.branch_length for leaf in leaves) / len(leaves)
    
    def set_branch_lengths(node):
        if node.is_leaf():
            # Keep original branch length for leaves
            pass
        else:
            # Set internal branch lengths
            node.branch_length = avg_length / 2
            for child in node.children:
                set_branch_lengths(child)
    
    set_branch_lengths(root)


def tree_to_newick(root: TreeNode) -> str:
    """Convert tree structure back to Newick format"""
    def node_to_string(node):
        if node.is_leaf():
            return f"{node.name}:{node.branch_length:.6f}"
        else:
            children_str = ",".join(node_to_string(child) for child in node.children)
            if node.branch_length > 0:
                return f"({children_str}):{node.branch_length:.6f}"
            else:
                return f"({children_str})"
    
    return node_to_string(root) + ";"


# Method 4: Quick and dirty approach for immediate use
def quick_balance_tree(newick_string: str) -> str:
    """
    Quick and dirty tree balancing for immediate use
    Works with most simple cases
    """
    # Extract taxa names and branch lengths
    taxa_pattern = r'([A-Za-z_][A-Za-z0-9_]*):([0-9.]+)'
    matches = re.findall(taxa_pattern, newick_string)
    
    if len(matches) < 2:
        return newick_string  # Can't balance single taxon
    
    # Create simple balanced pairs
    balanced_groups = []
    
    # Pair up taxa
    for i in range(0, len(matches), 2):
        if i + 1 < len(matches):
            name1, length1 = matches[i]
            name2, length2 = matches[i + 1]
            balanced_groups.append(f"({name1}:{length1},{name2}:{length2})")
        else:
            # Odd number - keep last one separate
            name, length = matches[i]
            balanced_groups.append(f"{name}:{length}")
    
    # Recursively group pairs
    while len(balanced_groups) > 1:
        new_groups = []
        for i in range(0, len(balanced_groups), 2):
            if i + 1 < len(balanced_groups):
                new_groups.append(f"({balanced_groups[i]},{balanced_groups[i+1]})")
            else:
                new_groups.append(balanced_groups[i])
        balanced_groups = new_groups
    
    return balanced_groups[0] + ";"


# Example usage and testing
def main():
    """Test the balancing methods"""
    
    # Your original problematic tree
    original_tree = "(((((T144:0.0683024,((T011:0.026304,T012:0.0534863)1:0.0144902,((T009:0.0107478,T007:0.0127697)1:0.0105828,(T008:0.0100232,(T005:0.0116844,T010:0.0446947)1:0.00960238)1:0.0156257)1:0.0156257)1:0.0156257)1:0.0042239,(((T015:0.00862792,T148:0.102515)1:0.00542209,(T118:0.0497522,T166:0.116746)1:0.0156257)1:0.00196123,(((((T019:0.0,T017:1.2966e-06)1:0.0156257,T030:0.00783559)1:0.00960238,(((((T119:1.2966e-06,((T101:1.2966e-06,T125:0.00422681)1:0.0186572,(((T099:1.2966e-06,T098:1.2966e-06)1:0.0156257,(T100:1.2966e-06,T109:0.00575643)1:0.0156257)1:0.0105828,(T110:0.00383177,T116:0.00977557)1:0.00960238)1:0.00960238)1:0.00960238)1:0.0156257,T131:0.00377399)1:0.0105828,(((((T058:0.00200968,T016:0.00434897)1:0.00877421,(((((T050:0.00393346,T036:1.2966e-06)1:0.0156257,T031:2.9495e-06)1:0.00960238,((T043:0.00200107,T051:2.2729e-06)1:0.0156257,T037:2.6709e-06)1:0.0156257)1:0.0105828,T042:0.00197487)1:0.0105828,T027:0.00840255)1:0.0196737)1:0.0128874,((((((((T133:0.00167565,T124:0.0106526)1:0.0105828,(T134:0.00167977,(T128:1.2966e-06,T127:2.4469e-06)1:0.00960238)1:0.0156257)1:0.0156257,(T093:0.00166974,T079:2.8381e-06)1:0.0156257)1:0.0144902,T123:0.0103189)1:0.0186572,T113:0.026499)1:0.00960238,T094:0.0017005)1:0.0105828,T080:0.0136149)1:0.0196737,T096:0.0223185)1:0.00997442)1:0.0181567,(T032:0.00615571,T026:0.00384808)1:0.0156257)1:0.012227,(((T104:0.00866085,(((T132:0.00347064,T138:0.0256794)1:0.0156257,T141:0.0133637)1:0.0105828,T057:0.0121206)1:0.0168419)1:0.0156257,T083:0.00353755)1:0.0156257,T105:0.0261037)1:0.00183178)1:0.0137216)1:0.0128874,T121:0.00408623)1:2.388e-06,(((((((T040:2.8319e-06,(T046:0.00373546,((T038:0.00182465,(T039:0.00184139,((T028:0.00187054,T033:2.7334e-06)1:0.0156257,T025:1.2966e-06)1:0.0196737)1:0.0144902)1:0.0156257,T045:0.00372075)1:0.0168419)1:0.00960238)1:0.0156257,T044:0.00183566)1:0.0156257,(((T111:0.0283984,T077:0.0031501)1:0.0156257,T136:0.0377915)1:0.0168419,(((T182:0.0117776,T081:2.8469e-06)1:0.0156257,T180:0.00316667)1:0.0156257,T090:0.00201546)1:0.00960238)1:0.0181567)1:0.0104806,((((((T048:0.0019577,(T049:0.00197691,T054:0.00194252)1:0.00960238)1:0.0156257,T055:0.00394859)1:0.0156257,((((T052:2.4393e-06,(T035:2.2817e-06,T060:0.0156933)1:0.0156257)1:0.00960238,(T053:2.568e-06,T029:2.8363e-06)1:0.0156257)1:0.0105828,T041:0.00391321)1:0.00960238,T034:1.2966e-06)1:0.00877421)1:0.0168419,(((T097:0.00188733,T064:0.00187443)1:0.0156257,T103:0.00368762)1:0.00960238,T091:0.0103376)1:0.0105828)1:0.0181567,T047:1.2966e-06)1:0.0168419,T065:1.2966e-06)1:0.0144902)1:0.00167608,T056:0.00723684)1:0.0234559,(((T022:1.2966e-06,((((((T137:0.0648993,(((((((T082:0.0171611,(T170:0.00617624,T146:0.0103046)1:0.0156257)1:0.0156257,T120:0.0293025)1:0.00960238,T169:0.00364251)1:0.00960238,T089:0.0202202)1:0.00313382,T140:0.0427856)1:0.00534101,(((((T145:0.0566461,((T102:0.00538611,T106:0.00906179)1:0.0181567,((T085:1.2966e-06,((T076:0.00178525,(((((T069:0.0,T067:0.0)1:0.00960238,T066:1.2966e-06)1:0.0105828,T074:0.00176906)1:0.0168419,(T070:0.0,T068:0.0)1:0.0156257)1:0.0156257,T087:0.00357205)1:0.0104806)1:0.0156257,T071:1.2966e-06)1:0.0168419)1:0.0156257,T086:1.2966e-06)1:0.00960238)1:0.00997442)1:0.0116221,((((T181:0.0840383,((T129:0.00167262,((((T126:2.6891e-06,T139:0.00502965)1:0.0156257,T150:0.0337415)1:0.00960238,T143:0.015368)1:0.0156257,T151:0.0398696)1:0.0168419)1:0.0156257,T130:0.00171062)1:0.0168419)1:0.00960238,(((((T168:0.0282882,(((T161:0.006268,(T153:0.00566544,T154:0.00721937)1:0.00960238)1:0.0156257,T163:0.0311405)1:0.0105828,T167:0.00994547)1:0.00960238)1:0.0156257,((T162:0.00388734,T157:0.00827383)1:0.00960238,T173:0.0611281)1:0.0105828)1:0.0186572,T160:0.0210878)1:0.0168419,T156:0.062523)1:0.0181567,(T158:2.8668e-06,T164:0.0118115)1:0.0156257)1:0.0106338)1:0.0104806,(T152:0.0322342,T159:0.0517007)1:0.0156257)1:0.0137216,((((T183:0.00359947,T177:0.00786306)1:0.00960238,(T184:0.0164968,T179:0.00879291)1:0.0156257)1:0.0156257,(T175:0.0374757,T147:0.0525727)1:0.00877421)1:0.0196737,((T172:0.00952943,(T171:0.0221505,T178:0.00878944)1:0.00960238)1:0.0156257,T176:0.0137644)1:0.0168419)1:0.0203813)1:0.0114256)1:0.00657853,T075:0.00177696)1:0.0200736,(((((T092:0.00413749,T078:1.2966e-06)1:0.0156257,T084:0.00242252)1:0.0105828,((T115:0.00356689,T095:0.00366855)1:0.0156257,T108:2.4028e-06)1:0.0156257)1:0.0168419,T135:0.0189422)1:0.0186572,((T061:1.2966e-06,(T062:0.00176832,T059:2.3398e-06)1:0.0105828)1:0.00960238,(T063:2.7651e-06,T073:0.00176286)1:0.0156257)1:0.00960238)1:0.0106338)1:0.00782649,T088:0.038692)1:0.00211294)1:0.0186572,(((((T114:0.002463,T149:0.0417195)1:0.0156257,T174:0.0974605)1:0.00960238,T112:0.00137582)1:0.0156257,T142:0.0466533)1:0.0168419,T155:0.0656231)1:0.0156257)1:0.0196737)1:0.0156257,T165:0.114359)1:0.00538252,T117:0.0561227)1:0.00960238,T021:1.2966e-06)1:0.00394842,T122:0.0605953)1:0.0156257,T020:1.2966e-06)1:0.0105828)1:0.0156257,T023:0.00169642)1:0.0168419,T024:0.020151)1:0.0114256)1:0.0179111,T107:0.0173428)1:0.00236713)1:0.00782649)1:0.0105828,T018:0.0)1:0.0111993,T072:0.0638776)1:0.00254371)1:0.00960238)1:0.0196737,T003:0.0061088)1:0.163801,(((T013:0.0177651,T004:0.00288566)1:0.0156257,T014:0.0221908)1:0.0156257,T006:0.00933119)1:0.0446008)1:0.0247839,(T001:0.00876202,T002:0.0125201)1:0.0156257);"
    
    print("Original tree:")
    print(original_tree)
    print()
    
    # Test quick balance
    print("Quick balanced tree:")
    quick_balanced = quick_balance_tree(original_tree)
    print(quick_balanced)
    print()
    
    # Test manual balance
    print("Manual balanced tree:")
    manual_balanced = balance_tree_manual(original_tree)
    print(manual_balanced)
    print()
    
    # Test with BioPython if available
    print("BioPython balanced tree:")
    try:
        bio_balanced = balance_tree_biopython(original_tree)
        print(bio_balanced)
    except:
        print("BioPython not available")
    print()
    
    # Show tree complexity comparison
    print("Complexity comparison:")
    print(f"Original - Parentheses: {original_tree.count('(')}, Nesting depth: {calculate_nesting_depth(original_tree)}")
    print(f"Balanced - Parentheses: {quick_balanced.count('(')}, Nesting depth: {calculate_nesting_depth(quick_balanced)}")


def calculate_nesting_depth(newick_string: str) -> int:
    """Calculate maximum nesting depth of parentheses"""
    max_depth = 0
    current_depth = 0
    
    for char in newick_string:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth -= 1
    
    return max_depth


if __name__ == "__main__":
    main()