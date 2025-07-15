#!/usr/bin/env python3
"""
Protein Evolution Simulator using Pyvolve
Simulates protein evolution with custom GTR matrix and indel parameters
Modified to read GTR parameters from TSV files
"""

import pyvolve
import numpy as np
import pandas as pd
from Bio import Phylo
from io import StringIO
import argparse
import sys
import os

def read_gtr_parameters_from_tsv(exchangeabilities_file, equilibriums_file):
    """
    Read GTR parameters from TSV files
    
    Parameters:
    - exchangeabilities_file: path to TSV file with exchangeabilities matrix
    - equilibriums_file: path to TSV file with equilibrium frequencies
    
    Returns:
    - equilibrium_freqs: list of 20 amino acid equilibrium frequencies
    - exchangeabilities: list of 190 exchangeability parameters (upper triangle)
    """
    
    # Read equilibrium frequencies
    print(f"Reading equilibrium frequencies from {equilibriums_file}")
    eq_df = pd.read_csv(equilibriums_file, sep='\t')
    
    # Check if the file has the expected columns
    if 'amino acid' not in eq_df.columns or 'equilibrium frequency' not in eq_df.columns:
        raise ValueError(f"Expected columns 'amino acid' and 'equilibrium frequency' in {equilibriums_file}")
    
    # Create mapping from amino acid to frequency
    eq_dict = dict(zip(eq_df['amino acid'], eq_df['equilibrium frequency']))
    
    # Pyvolve expects amino acids in alphabetical order
    alphabetical_order = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    
    # Extract frequencies in alphabetical order
    equilibrium_freqs = []
    for aa in alphabetical_order:
        if aa not in eq_dict:
            raise ValueError(f"Amino acid {aa} not found in equilibrium frequencies file")
        equilibrium_freqs.append(eq_dict[aa])
    
    # Validate frequencies sum to 1
    if not np.isclose(sum(equilibrium_freqs), 1.0, rtol=1e-3):
        print(f"Warning: equilibrium frequencies sum to {sum(equilibrium_freqs):.6f}, normalizing to 1.0")
        total = sum(equilibrium_freqs)
        equilibrium_freqs = [freq / total for freq in equilibrium_freqs]
    
    # Read exchangeabilities matrix
    print(f"Reading exchangeabilities matrix from {exchangeabilities_file}")
    ex_df = pd.read_csv(exchangeabilities_file, sep='\t', index_col=0)
    
    # Get the amino acids from the matrix (should be in alphabetical order based on your screenshot)
    matrix_amino_acids = list(ex_df.index)
    
    # Verify we have 20 amino acids
    if len(matrix_amino_acids) != 20:
        raise ValueError(f"Expected 20 amino acids in exchangeabilities matrix, got {len(matrix_amino_acids)}")
    
    # Extract upper triangle of exchangeabilities matrix in alphabetical order
    # Pyvolve expects amino acids in alphabetical order
    exchangeabilities = []
    
    # Since your TSV files are already in alphabetical order, we can use them directly
    for i in range(20):
        for j in range(i+1, 20):
            # Get amino acids in alphabetical order
            aa_i = matrix_amino_acids[i]
            aa_j = matrix_amino_acids[j]
            
            # Get the exchangeability value from the matrix
            value = ex_df.iloc[i, j]
            exchangeabilities.append(value)
    
    print(f"Successfully read {len(equilibrium_freqs)} equilibrium frequencies")
    print(f"Successfully read {len(exchangeabilities)} exchangeability parameters")
    
    return equilibrium_freqs, exchangeabilities

def create_gtr_model(equilibrium_freqs, exchangeabilities, gamma_shape=None, prop_invariant=None):
    """
    Create a custom GTR model for protein evolution using pyvolve
    
    The correct approach is to use the 'custom' model type and provide the rate matrix
    """
    
    # Validate inputs
    if len(equilibrium_freqs) != 20:
        raise ValueError("equilibrium_freqs must have exactly 20 values (one for each amino acid)")
    
    if len(exchangeabilities) != 190:
        raise ValueError("exchangeabilities must have exactly 190 values (upper triangle of 20x20 matrix)")
    
    if not np.isclose(sum(equilibrium_freqs), 1.0):
        raise ValueError("equilibrium_freqs must sum to 1.0")
    
    # Construct the Q matrix from GTR parameters
    Q = construct_q_matrix(equilibrium_freqs, exchangeabilities)
    
    # Ensure Q is a numpy array and convert to list of lists for pyvolve
    Q = np.array(Q)
    Q_list = Q.tolist() 
    
    # Ensure equilibrium_freqs is a list (not numpy array)
    equilibrium_freqs = list(equilibrium_freqs)
    
    # Create model parameters dictionary for custom model
    model_params = {
        'matrix': Q,  # Pass as list of lists #!check with numpy lists
        'state_freqs': equilibrium_freqs
    }
    
    # Add rate variation parameters if provided
    if gamma_shape is not None:
        model_params['alpha'] = gamma_shape
    
    if prop_invariant is not None:
        model_params['pinv'] = prop_invariant
    
    # Create a custom model using the 'custom' model type
    # We need to specify the amino acid alphabet
    amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    
    # Create the model using the custom type
    model = pyvolve.Model("custom", model_params, alphabet=amino_acids)
    
    return model

def construct_q_matrix(equilibrium_freqs, exchangeabilities):
    """
    Construct a Q matrix from GTR parameters
    
    Parameters:
    - equilibrium_freqs: list of 20 amino acid equilibrium frequencies
    - exchangeabilities: list of 190 exchangeability parameters (upper triangle)
    
    Returns:
    - Q: 20x20 rate matrix
    """
    
    # Initialize Q matrix
    Q = np.zeros((20, 20))
    
    # Fill the upper triangle with exchangeabilities * equilibrium frequencies
    idx = 0
    for i in range(20):
        for j in range(i+1, 20):
            # Q[i][j] = exchangeability * equilibrium_freq[j]
            Q[i][j] = exchangeabilities[idx] * equilibrium_freqs[j]
            # Q[j][i] = exchangeability * equilibrium_freq[i] (symmetric)
            Q[j][i] = exchangeabilities[idx] * equilibrium_freqs[i]
            idx += 1
    
    # Set diagonal elements so each row sums to 0
    for i in range(20):
        Q[i][i] = -np.sum(Q[i])
    
    return Q

def create_indel_model(insertion_rate, deletion_rate, mean_insert_length, mean_delete_length):
    """
    Create an indel model with geometric length distributions
    
    Parameters:
    - insertion_rate: rate of insertions per site per unit time
    - deletion_rate: rate of deletions per site per unit time
    - mean_insert_length: mean length of insertions (geometric distribution)
    - mean_delete_length: mean length of deletions (geometric distribution)
    """
    
    # Convert mean lengths to geometric distribution parameters
    # For geometric distribution: p = 1/mean
    insert_prob = 1.0 / mean_insert_length
    delete_prob = 1.0 / mean_delete_length
    
    # Create indel model
    indel_model = pyvolve.Model("indel", 
                               {'insertion_rate': insertion_rate,
                                'deletion_rate': deletion_rate,
                                'insertion_length': insert_prob,
                                'deletion_length': delete_prob})
    
    return indel_model

def create_guide_tree(num_taxa, tree_string=None):
    """
    Create a guide tree for simulation
    
    Parameters:
    - num_taxa: number of taxa/sequences
    - tree_string: optional Newick tree string, if None creates a balanced tree
    """
    
    if tree_string:
        # Parse provided tree string
        tree_io = StringIO(tree_string)
        tree = Phylo.read(tree_io, "newick")
    else:
        # Create a simple balanced tree
        if num_taxa == 2:
            tree_string = "(seq1:0.1,seq2:0.1);"
        elif num_taxa == 3:
            tree_string = "((seq1:0.1,seq2:0.1):0.1,seq3:0.1);"
        elif num_taxa == 4:
            tree_string = "((seq1:0.1,seq2:0.1):0.1,(seq3:0.1,seq4:0.1):0.1);"
        else:
            # For more taxa, create a more complex balanced tree
            tree_string = f"({','.join([f'seq{i}:0.1' for i in range(1, num_taxa+1)])});"
        
        tree_io = StringIO(tree_string)
        tree = Phylo.read(tree_io, "newick")
    
    return tree

def simulate_protein_evolution(num_sequences, sequence_length, 
                             equilibrium_freqs, exchangeabilities,
                             gamma_shape=None, prop_invariant=None,
                             insertion_rate=0.01, deletion_rate=0.01,
                             mean_insert_length=2.0, mean_delete_length=2.0,
                             guide_tree=None, output_file=None):
    """
    Main function to simulate protein evolution
    
    Parameters:
    - num_sequences: number of sequences/taxa to simulate
    - sequence_length: length of root sequence
    - equilibrium_freqs: amino acid equilibrium frequencies (20 values)
    - exchangeabilities: GTR exchangeability parameters (190 values)
    - gamma_shape: gamma distribution shape parameter
    - prop_invariant: proportion of invariant sites
    - insertion_rate: rate of insertions
    - deletion_rate: rate of deletions
    - mean_insert_length: mean length of insertions
    - mean_delete_length: mean length of deletions
    - guide_tree: phylogenetic tree (Newick string or None for default)
    - output_file: output file name (None for stdout)
    """
    
    print(f"Simulating evolution of {num_sequences} protein sequences...")
    print(f"Root sequence length: {sequence_length}")
    print(f"Insertion rate: {insertion_rate}, Deletion rate: {deletion_rate}")
    print(f"Mean insert length: {mean_insert_length}, Mean delete length: {mean_delete_length}")
    
    # Create the substitution model
    print("Creating GTR substitution model...")
    subst_model = create_gtr_model(equilibrium_freqs, exchangeabilities, 
                                  gamma_shape, prop_invariant)
    
    # Create the indel model
    print("Creating indel model...")
    indel_model = create_indel_model(insertion_rate, deletion_rate, 
                                   mean_insert_length, mean_delete_length)
    
    # Create the guide tree
    print("Creating guide tree...")
    tree = create_guide_tree(num_sequences, guide_tree)
    
    # Create the partition (combining substitution and indel models)
    partition = pyvolve.Partition(models=[subst_model, indel_model], 
                                 root_sequence_length=sequence_length)
    
    # Create the evolver
    evolver = pyvolve.Evolver(partitions=partition, tree=tree)
    
    # Run the simulation
    print("Running evolution simulation...")
    evolver()
    
    # Get the results
    sequences = evolver.get_sequences()
    
    # Output results
    if output_file:
        print(f"Writing results to {output_file}...")
        with open(output_file, 'w') as f:
            for seq_id, sequence in sequences.items():
                f.write(f">{seq_id}\n{sequence}\n")
    else:
        print("\nSimulated sequences:")
        for seq_id, sequence in sequences.items():
            print(f">{seq_id}")
            print(sequence)
    
    return sequences

def get_default_amino_acid_freqs():
    """Return default amino acid frequencies in alphabetical order (A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y)"""
    return [0.087, 0.033, 0.047, 0.050, 0.040, 0.089, 0.034, 0.037, 0.080, 0.080, 0.015, 0.040, 0.050, 0.038, 0.041, 0.070, 0.058, 0.067, 0.013, 0.032]

def get_default_exchangeabilities():
    """Return default GTR exchangeabilities (190 values for upper triangle)"""
    # This is a simplified example - in practice, you'd use empirically derived values
    return [1.0] * 190

def main():
    parser = argparse.ArgumentParser(description='Simulate protein evolution with pyvolve')
    parser.add_argument('--num-sequences', type=int, default=4, 
                       help='Number of sequences to simulate')
    parser.add_argument('--sequence-length', type=int, default=100,
                       help='Length of root sequence')
    parser.add_argument('--exchangeabilities-file', type=str, default=None,
                       help='Path to TSV file with GTR exchangeabilities matrix')
    parser.add_argument('--equilibriums-file', type=str, default=None,
                       help='Path to TSV file with equilibrium frequencies')
    parser.add_argument('--gamma-shape', type=float, default=None,
                       help='Gamma distribution shape parameter')
    parser.add_argument('--prop-invariant', type=float, default=None,
                       help='Proportion of invariant sites')
    parser.add_argument('--insertion-rate', type=float, default=0.01,
                       help='Insertion rate per site per unit time')
    parser.add_argument('--deletion-rate', type=float, default=0.01,
                       help='Deletion rate per site per unit time')
    parser.add_argument('--mean-insert-length', type=float, default=2.0,
                       help='Mean insertion length (geometric distribution)')
    parser.add_argument('--mean-delete-length', type=float, default=2.0,
                       help='Mean deletion length (geometric distribution)')
    parser.add_argument('--tree', type=str, default=None,
                       help='Newick tree string (optional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file name (optional)')
    
    args = parser.parse_args()
    
    # Determine how to get GTR parameters
    if args.exchangeabilities_file and args.equilibriums_file:
        # Read from TSV files
        if not os.path.exists(args.exchangeabilities_file):
            print(f"Error: Exchangeabilities file {args.exchangeabilities_file} not found")
            sys.exit(1)
        if not os.path.exists(args.equilibriums_file):
            print(f"Error: Equilibriums file {args.equilibriums_file} not found")
            sys.exit(1)
        
        equilibrium_freqs, exchangeabilities = read_gtr_parameters_from_tsv(
            args.exchangeabilities_file, args.equilibriums_file)
    else:
        # Use default parameters
        print("Using default GTR parameters (specify --exchangeabilities-file and --equilibriums-file to use custom parameters)")
        equilibrium_freqs = get_default_amino_acid_freqs()
        exchangeabilities = get_default_exchangeabilities()
    
    try:
        sequences = simulate_protein_evolution(
            num_sequences=args.num_sequences,
            sequence_length=args.sequence_length,
            equilibrium_freqs=equilibrium_freqs,
            exchangeabilities=exchangeabilities,
            gamma_shape=args.gamma_shape,
            prop_invariant=args.prop_invariant,
            insertion_rate=args.insertion_rate,
            deletion_rate=args.deletion_rate,
            mean_insert_length=args.mean_insert_length,
            mean_delete_length=args.mean_delete_length,
            guide_tree=args.tree,
            output_file=args.output
        )
        
        print(f"\nSimulation completed successfully!")
        print(f"Generated {len(sequences)} sequences")
        
    except Exception as e:
        print(f"Error during simulation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Example usage:
"""
# Using your TSV files
python evolve.py \
    --exchangeabilities-file GTR_exchangeabilities.tsv \
    --equilibriums-file GTR_equilibriums.tsv \
    --num-sequences 5 \
    --sequence-length 200

# Advanced usage with custom parameters
python evolve.py \
    --exchangeabilities-file GTR_exchangeabilities.tsv \
    --equilibriums-file GTR_equilibriums.tsv \
    --num-sequences 6 \
    --sequence-length 150 \
    --gamma-shape 0.5 \
    --prop-invariant 0.1 \
    --insertion-rate 0.02 \
    --deletion-rate 0.015 \
    --mean-insert-length 3.0 \
    --mean-delete-length 2.5 \
    --tree "((seq1:0.1,seq2:0.1):0.05,(seq3:0.1,seq4:0.1):0.05);" \
    --output evolved_sequences.fasta

# Using default parameters (fallback)
python evolve.py --num-sequences 4 --sequence-length 100
"""