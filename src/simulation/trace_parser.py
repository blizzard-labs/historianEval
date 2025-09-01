#!/usr/bin/env python3
"""
MCMC Trace Log Parser with Tree Extraction
Parses trace.log files from phylogenetic MCMC samplers and creates both trace files and NEXUS tree files.
"""

import re
import sys
import os
from collections import OrderedDict
from Bio import Phylo
from io import StringIO

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

def remove_internal_nodes_baliphy(input_file):
    out_trees = ""
    with open(input_file, 'r') as infile:
        for line in infile:
            cleaned = re.sub(r'\(A\d+:', '(', line)  # removes internal node name but keeps :
            cleaned = re.sub(r'A\d+:', '', cleaned)  # removes other occurrences
            cleaned = re.sub(r'A\d+\)', ')', cleaned)  # in case no branch length
            
            out_trees += cleaned
    
    return out_trees

def extract_sequences_historian(input_file, output_fasta_file):
    """
    Extract sequence alignments from Historian trace.log file and create FASTA format file.
    
    Args:
        input_file (str): Path to input trace.log file
        output_fasta_file (str): Path to output .fastas file
    """
    
    print(f"Extracting sequences from {input_file}...")
    
    sequence_iterations = []
    taxa_order = []
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split by Stockholm headers to get iterations
    iterations = content.split('# STOCKHOLM 1.0')
    if iterations and not iterations[0].strip():
        iterations = iterations[1:]  # Remove empty first element
    
    sequence_count = 0
    
    for iteration_idx, iteration_content in enumerate(iterations):
        lines = iteration_content.strip().split('\n')
        
        current_sequences = OrderedDict()
        in_alignment_section = False
        
        for idx, line in enumerate(lines):
            line = line.strip()
            
            # Skip comment lines, parameter lines, and tree lines
            if (line.startswith('#') or 
                line.startswith('tree ') or 
                line.startswith('Tree ') or
                line.startswith('//') or
                not line):
                continue
            
            # Skip lines that are just gaps/stars/placeholders
            if line.replace('x', '').replace('-', '').replace('X', '').replace('*', '').strip() == '':
                continue
            
            '''
            # Skip internal node lines
            if line.startswith('node'):
                continue
            '''
            
            if lines[idx - 1].strip().startswith("#=GF NH"):
                in_alignment_section = True
            
            # Look for sequence data lines
            parts = line.split()
            if len(parts) >= 2 and in_alignment_section:
                potential_taxon = parts[0]
                potential_sequence = parts[1] if len(parts) > 1 else ""
                
                #print('potential_taxon:', potential_taxon)
                #print('potential_sequence:', potential_sequence)
                
                # Check if this looks like sequence data (contains common sequence characters)
                if (potential_sequence and 
                    len(potential_sequence) > 10 and
                    any(char in potential_sequence for char in 'ATCGRYSWKMBDHVNX')):
                    
                    current_sequences[potential_taxon] = potential_sequence
                    
                    # Store taxa order from first iteration
                    if iteration_idx == 1 and potential_taxon not in taxa_order:
                        taxa_order.append(potential_taxon)
        
        if current_sequences:
            sequence_iterations.append(current_sequences)
            sequence_count += 1
    
    print(f"Found {len(taxa_order)} taxa: {taxa_order}")
    print(f"Extracted sequences from {sequence_count} iterations")
    
    if not sequence_iterations:
        print("No sequence data found in the trace file!")
        return 0
    
    # Write FASTA format file with all iterations
    with open(output_fasta_file, 'w') as f:
        for iter_idx, sequences in enumerate(sequence_iterations):
            f.write(f"iterations = {iter_idx}\n\n")
            
            # Write sequences in consistent order
            for taxon in taxa_order:
                if taxon in sequences:
                    f.write(f">{taxon}\n")
                    f.write(f"{sequences[taxon]}\n")
            
            f.write("\n\n\n")  # Separate iterations
    
    print(f"FASTA sequences file written to {output_fasta_file}")
    return len(sequence_iterations)

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


def configure_baliphy_trees(input_file, output_trees_file):
    trees = remove_internal_nodes_baliphy(input_file)
    tree = Phylo.read(StringIO(trees.strip().splitlines()[0]), 'newick')
    
    taxa = tree.get_terminals()
    print(f"Found {len(taxa)} taxa: {taxa}")
    
    # Write NEXUS format trees file
    with open(output_trees_file, 'w') as f:
        f.write("#NEXUS\n")
        f.write("\n")
        f.write("Begin taxa;\n")
        f.write(f"\tDimensions ntax={len(taxa)};\n")
        f.write("\tTaxlabels\n")
        
        for taxon in taxa:
            f.write(f"\t\t{taxon.name}\n")
        
        f.write("\t\t;\n")
        f.write("End;\n")
        f.write("Begin trees;\n")
        f.write("\tTranslate\n")
        
        # Write translation table
        for i, taxon in enumerate(taxa, 1):
            separator = "," if i < len(taxa) else ""
            f.write(f"\t\t{i} {taxon.name}{separator}\n")
        
        f.write("\t\t;\n")
        
        # Write trees (already in Newick format with semicolons)
        for i, tree in enumerate(trees.splitlines()):
            # Ensure the tree doesn't end with double semicolons
            clean_tree = tree.rstrip(';') + ';'
            f.write(f"tree STATE_{i} = {clean_tree}\n")
        
        f.write("End;\n")
    
    print(f"NEXUS trees file written to {output_trees_file}")
    print(f"Trees converted from New Hampshire to standard Newick format")
    print(f"Internal node labels (A1, A2, etc.) removed")
    return len(trees)


def main():
    if len(sys.argv) < 4:
        print("Usage: python mcmc_trace_parser.py <format> <input_trace.log> <output_trace.txt> [--trees] [--sequences]")
        print("Example: python mcmc_trace_parser.py historian trace.log parsed_trace.txt")
        print("Example with trees: python mcmc_trace_parser.py historian trace.log parsed_trace.txt --trees")
        print("Example with sequences: python mcmc_trace_parser.py historian trace.log parsed_trace.txt --sequences")
        print("Example with both: python mcmc_trace_parser.py historian trace.log parsed_trace.txt --trees --sequences")
        print("")
        print("Options:")
        print("  --trees       Also extract trees and create NEXUS .trees file for TreeStat2")
        print("  --sequences   Also extract posterior sequences and create FASTA .fastas file")
        sys.exit(1)
    
    format_type = sys.argv[1].lower()
    
    input_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        input_file = input_file.replace('.log', '.log.1')
    
    output_file = sys.argv[3]
    extract_trees_flag = '--trees' in sys.argv
    extract_sequences_flag = '--sequences' in sys.argv
    
    # Determine output filenames
    if extract_trees_flag:
        if output_file.endswith('.txt'):
            trees_file = output_file.replace('.txt', '.trees')
        elif not output_file.endswith('.trees'):
            trees_file = output_file + '.trees'
        else:
            trees_file = output_file
    
    if extract_sequences_flag:
        if output_file.endswith('.txt'):
            sequences_file = output_file.replace('.txt', '.fastas')
        else:
            sequences_file = os.path.splitext(output_file)[0] + '.fastas'
    
    if format_type == 'historian':
        print("Using Historian format parser...")
    
        try:
            '''
            # Parse trace parameters
            n_iterations, n_params = parse_trace_log(input_file, output_file)
            
            # If we didn't get much data, try the advanced parser
            if n_iterations < 10 or n_params < 3:
                print("Basic parser found limited data, trying advanced parser...")
            '''
            
            n_iterations, n_params = parse_trace_log_advanced(input_file, output_file)
            
            # Extract trees if requested
            n_trees = 0
            if extract_trees_flag:
                n_trees = extract_trees(input_file, trees_file)
            
            # Extract sequences if requested
            n_sequence_iterations = 0
            if extract_sequences_flag:
                n_sequence_iterations = extract_sequences_historian(input_file, sequences_file)
            
            print(f"\nSummary:")
            print(f"  Iterations processed: {n_iterations}")
            print(f"  Parameters extracted: {n_params}")
            print(f"  Output trace file: {output_file}")
            
            if extract_trees_flag:
                print(f"  Trees extracted: {n_trees}")
                print(f"  Output trees file: {trees_file}")
                print(f"  Format: NEXUS (compatible with TreeStat2)")
            
            if extract_sequences_flag:
                print(f"  Sequence iterations extracted: {n_sequence_iterations}")
                print(f"  Output sequences file: {sequences_file}")
                print(f"  Format: FASTA (posterior.fastas)")
            
        except FileNotFoundError:
            print(f"Error: Could not find input file '{input_file}'")
            sys.exit(1)
        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    elif format_type == 'baliphy':
        print("Using Baliphy format parser...")
        
        try:
            n_trees = 0
            if extract_trees_flag:
                n_trees = configure_baliphy_trees(input_file, trees_file)
            
            print(f"\nSummary:")
            if extract_trees_flag:
                print(f"  Trees extracted: {n_trees}")
                print(f"  Output trees file: {trees_file}")
                print(f"  Format: NEXUS (compatible with TreeStat2)")
            else:
                print("No --trees flag provided, nothing to do.")
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