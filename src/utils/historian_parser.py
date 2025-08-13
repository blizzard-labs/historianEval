#!/usr/bin/env python3
"""
MCMC Trace Log Parser with Tree Extraction
Parses trace.log files from phylogenetic MCMC samplers and creates both trace files and NEXUS tree files.
"""

import re
import sys
from collections import OrderedDict

def convert_to_newick(nh_tree):
    """
    Convert New Hampshire format tree to standard Newick format.
    Removes internal node labels (node1, node2, etc.) while preserving ENV_ taxa labels.
    
    Args:
        nh_tree (str): New Hampshire format tree string
        
    Returns:
        str: Standard Newick format tree string
    """
    # Clean up the tree string
    tree = nh_tree.strip()
    
    # Remove any leading/trailing whitespace and formatting artifacts
    tree = re.sub(r'^\s*GF\s*NH\s*', '', tree)
    tree = tree.strip()
    
    # Remove internal node labels that match pattern "node" followed by numbers
    # This regex looks for patterns like ":0.123)node1:0.456" and converts to ":0.123):0.456"
    # or ")node1," to "),"
    tree = re.sub(r'\)node\d+([,:)])', r')\1', tree)
    
    # Also handle cases where node labels appear without parentheses before them
    # Pattern like "ENV_something:0.1,node1:0.2" -> "ENV_something:0.1,:0.2"
    tree = re.sub(r',node\d+([:,)])', r',\1', tree)
    tree = re.sub(r'\(node\d+([:,)])', r'(\1', tree)
    
    # Remove any remaining standalone internal node labels
    # This catches patterns like ")node123" and converts to ")"
    tree = re.sub(r'\)node\d+', ')', tree)
    
    # Clean up any double commas or other artifacts from node removal
    tree = re.sub(r',,+', ',', tree)
    tree = re.sub(r'\(,', '(', tree)
    tree = re.sub(r',\)', ')', tree)
    
    # Ensure proper semicolon termination
    if not tree.endswith(';'):
        tree += ';'
    
    return tree

def parse_trace_log(input_file, output_file):
    """
    Parse MCMC trace log and create a clean trace file.
    
    Args:
        input_file (str): Path to input trace.log file
        output_file (str): Path to output trace file
    """
    
    iterations = []
    all_parameters = OrderedDict()
    
    print(f"Parsing {input_file}...")
    
    with open(input_file, 'r') as f:
        current_iteration = None
        current_params = {}
        
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments that aren't Stockholm headers
            if not line or (line.startswith('#') and 'STOCKHOLM' not in line):
                continue
                
            # Check for Stockholm header indicating new iteration
            if line.startswith('# STOCKHOLM'):
                # If we have a previous iteration, save it
                if current_iteration is not None:
                    iterations.append((current_iteration, current_params.copy()))
                
                # Start new iteration
                current_iteration = len(iterations)
                current_params = {}
                continue
            
            # Parse parameter lines
            if '=' in line:
                # Handle lines like "likelihood indels substitutions mean_indel_length"
                if not '=' in line or line.count('=') > 10:  # Likely a header line
                    continue
                    
                # Split on whitespace and look for key=value pairs
                parts = line.split()
                for part in parts:
                    if '=' in part:
                        try:
                            key, value = part.split('=', 1)
                            # Clean up the key and value
                            key = key.strip()
                            value = value.strip()
                            
                            # Try to convert to float, keep as string if not possible
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                            
                            current_params[key] = value
                            
                            # Keep track of all parameters we've seen
                            if key not in all_parameters:
                                all_parameters[key] = []
                                
                        except ValueError:
                            continue
            
            # Also handle lines that might have parameter names and values separated
            elif line and not line.startswith('#'):
                # Look for numeric values that might be parameters
                parts = line.split()
                if len(parts) >= 2:
                    # Try to identify parameter patterns
                    for i, part in enumerate(parts):
                        try:
                            float(part)
                            # This could be a parameter value
                            # Use position-based naming if we can't identify the parameter
                            if i == 0 and 'likelihood' not in current_params:
                                current_params['likelihood'] = float(part)
                            elif 'posterior' in line.lower() and 'posterior' not in current_params:
                                current_params['posterior'] = float(part)
                        except ValueError:
                            continue
        
        # Don't forget the last iteration
        if current_iteration is not None:
            iterations.append((current_iteration, current_params))
    
    print(f"Found {len(iterations)} iterations")
    print(f"Parameters detected: {list(all_parameters.keys())}")
    
    # Create the output trace file
    with open(output_file, 'w') as f:
        # Write header
        if all_parameters:
            header = "iteration\t" + "\t".join(all_parameters.keys())
            f.write(header + "\n")
            
            # Write data
            for iteration_num, params in iterations:
                row = [str(iteration_num)]
                
                for param_name in all_parameters.keys():
                    value = params.get(param_name, 'NA')
                    row.append(str(value))
                
                f.write("\t".join(row) + "\n")
    
    print(f"Trace file written to {output_file}")
    
    return len(iterations), len(all_parameters)

def extract_trees(input_file, output_trees_file):
    """
    Extract phylogenetic trees from trace.log file and create NEXUS format trees file.
    Trees are converted from New Hampshire format to standard Newick format.
    
    Args:
        input_file (str): Path to input trace.log file
        output_trees_file (str): Path to output .trees file
    """
    
    print(f"Extracting trees from {input_file}...")
    
    trees = []
    taxa_labels = []
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split by Stockholm headers to get iterations
    iterations = content.split('# STOCKHOLM 1.0')
    if iterations and not iterations[0].strip():
        iterations = iterations[1:]  # Remove empty first element
    
    # Extract taxa labels from the first iteration
    if iterations:
        first_iteration = iterations[1] if len(iterations) > 1 else iterations[0]
        lines = first_iteration.strip().split('\n')
        
        curr_in_taxa_ln = False
        for idx, line in enumerate(lines):
            line = line.strip()
            # Skip comment lines, parameter lines, and tree lines
            if (line.startswith('#') or 
                line.startswith('tree ') or 
                line.startswith('Tree ') or
                line.startswith('//') or
                not line or
                line.replace('x', '').replace('-', '').replace('X', '').strip() == ''):
                curr_in_taxa_ln = False
                continue
            
            if line.startswith('node'):
                continue
            
            if lines[idx - 1].strip().startswith("#=GF NH"):
                curr_in_taxa_ln = True
            
            # Look for lines that start with sequence names (before sequence data)
            parts = line.split()
            if len(parts) >= 2 and curr_in_taxa_ln:
                potential_taxon = parts[0]
                
                if potential_taxon and potential_taxon not in taxa_labels:
                    taxa_labels.append(potential_taxon)
    
    print(f"Found {len(taxa_labels)} taxa: {taxa_labels}")
    
    # Extract trees from each iteration
    tree_count = 0
    for iteration_idx, iteration_content in enumerate(iterations):
        lines = iteration_content.strip().split('\n')
        
        # Look for tree lines (lines that start with "tree" and contain parentheses)
        for line in lines:
            line = line.strip()
            
            # Tree lines typically start with "tree" and contain Newick format
            if (line.startswith('#=GF NH') and '(' in line and ')' in line) or \
               (line.startswith('Tree ') and '(' in line and ')' in line):
                
                # Extract the tree string
                tree_string = ""
                if line.startswith('#=GF NH'):
                    # Extract everything after '#=GF NH'
                    tree_string = line.replace('#=GF NH', '').strip()
                elif '=' in line:
                    # Format is usually: tree STATE_0 = (newick_string);
                    tree_part = line.split('=', 1)[1].strip()
                    if tree_part.endswith(';'):
                        tree_part = tree_part[:-1]  # Remove trailing semicolon
                    tree_string = tree_part
                
                if tree_string:
                    # Convert from New Hampshire to standard Newick format
                    newick_tree = convert_to_newick(tree_string)
                    trees.append(newick_tree)
                    tree_count += 1
                    break
    
    print(f"Extracted {tree_count} trees")
    
    if not trees:
        print("No trees found in the trace file!")
        return 0
    
    # Write NEXUS format trees file
    with open(output_trees_file, 'w') as f:
        f.write("#NEXUS\n")
        f.write("\n")
        f.write("Begin taxa;\n")
        f.write(f"\tDimensions ntax={len(taxa_labels)};\n")
        f.write("\tTaxlabels\n")
        
        for taxon in taxa_labels:
            f.write(f"\t\t{taxon}\n")
        
        f.write("\t\t;\n")
        f.write("End;\n")
        f.write("Begin trees;\n")
        f.write("\tTranslate\n")
        
        # Write translation table
        for i, taxon in enumerate(taxa_labels, 1):
            separator = "," if i < len(taxa_labels) else ""
            f.write(f"\t\t{i} {taxon}{separator}\n")
        
        f.write("\t\t;\n")
        
        # Write trees (already in Newick format with semicolons)
        for i, tree in enumerate(trees):
            # Ensure the tree doesn't end with double semicolons
            clean_tree = tree.rstrip(';') + ';'
            f.write(f"tree STATE_{i} = {clean_tree}\n")
        
        f.write("End;\n")
    
    print(f"NEXUS trees file written to {output_trees_file}")
    print(f"Trees converted from New Hampshire to standard Newick format")
    print(f"Internal node labels (node1, node2, etc.) removed")
    return len(trees)

def parse_trace_log_advanced(input_file, output_file):
    """
    Advanced parser that tries to extract more parameter information.
    """
    print(f"Using advanced parsing for {input_file}...")
    
    iterations = []
    parameter_names = []
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # First pass: identify parameter names from header-like lines
    for i, line in enumerate(lines):
        line = line.strip()
        if 'likelihood' in line and 'indels' in line and 'substitutions' in line:
            # This looks like a parameter header
            parts = line.split()
            # Filter out non-parameter words
            potential_params = []
            for part in parts:
                if part and not part.startswith('#'):
                    potential_params.append(part)
            if potential_params:
                parameter_names = potential_params
                break
    
    if not parameter_names:
        # Fallback parameter names
        parameter_names = ['likelihood', 'prior', 'posterior']
    
    print(f"Parameter names identified: {parameter_names}")
    
    # Second pass: extract data
    current_iteration = -1
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('# STOCKHOLM'):
            current_iteration += 1
            continue
        
        # Look for lines with multiple numeric values
        if line and not line.startswith('#'):
            parts = line.split()
            numeric_parts = []
            
            for part in parts:
                try:
                    val = float(part)
                    numeric_parts.append(val)
                except ValueError:
                    continue
            
            # If we have enough numeric values, treat as parameter values
            if len(numeric_parts) >= len(parameter_names):
                iteration_data = [current_iteration] + numeric_parts[:len(parameter_names)]
                iterations.append(iteration_data)
    
    # Write output
    with open(output_file, 'w') as f:
        # Header
        header = "iter\t" + "\t".join(parameter_names)
        f.write(header + "\n")
        
        # Data
        for iteration_data in iterations:
            f.write("\t".join(map(str, iteration_data)) + "\n")
    
    print(f"Advanced parsing complete. {len(iterations)} iterations written to {output_file}")
    
    return len(iterations), len(parameter_names)

def main():
    if len(sys.argv) < 3:
        print("Usage: python mcmc_trace_parser.py <input_trace.log> <output_trace.txt> [--trees]")
        print("Example: python mcmc_trace_parser.py trace.log parsed_trace.txt")
        print("Example with trees: python mcmc_trace_parser.py trace.log parsed_trace.txt --trees")
        print("")
        print("Options:")
        print("  --trees    Also extract trees and create NEXUS .trees file for TreeStat2")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    extract_trees_flag = '--trees' in sys.argv
    
    # Determine trees output filename
    if extract_trees_flag:
        if output_file.endswith('.txt'):
            trees_file = output_file.replace('.txt', '.trees')
        else:
            trees_file = output_file + '.trees'
    
    try:
        # Parse trace parameters
        n_iterations, n_params = parse_trace_log(input_file, output_file)
        
        # If we didn't get much data, try the advanced parser
        if n_iterations < 10 or n_params < 3:
            print("Basic parser found limited data, trying advanced parser...")
            n_iterations, n_params = parse_trace_log_advanced(input_file, output_file)
        
        # Extract trees if requested
        n_trees = 0
        if extract_trees_flag:
            n_trees = extract_trees(input_file, trees_file)
        
        print(f"\nSummary:")
        print(f"  Iterations processed: {n_iterations}")
        print(f"  Parameters extracted: {n_params}")
        print(f"  Output trace file: {output_file}")
        
        if extract_trees_flag:
            print(f"  Trees extracted: {n_trees}")
            print(f"  Output trees file: {trees_file}")
            print(f"  Format: NEXUS (compatible with TreeStat2)")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()