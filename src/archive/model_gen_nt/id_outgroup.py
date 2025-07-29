#!/usr/bin/env python3
"""
Scalable script to identify the outgroup in large multiple sequence alignments.
Uses memory-efficient algorithms and optimizations for large datasets.
"""

import argparse
import numpy as np
from Bio import SeqIO
import multiprocessing as mp
from functools import partial
import time
import sys
from collections import Counter
import random

def hamming_distance_vectorized(seq1, seq2):
    """Vectorized Hamming distance calculation."""
    seq1_arr = np.frombuffer(seq1.encode('ascii'), dtype=np.uint8)
    seq2_arr = np.frombuffer(seq2.encode('ascii'), dtype=np.uint8)
    
    # Exclude gaps from both sequences
    valid_positions = (seq1_arr != ord('-')) & (seq2_arr != ord('-'))
    return np.sum(seq1_arr[valid_positions] != seq2_arr[valid_positions])

def calculate_distances_chunk(args):
    """Calculate distances for a chunk of sequence pairs."""
    sequences, seq_ids, start_idx, end_idx = args
    
    chunk_distances = {}
    for i in range(start_idx, end_idx):
        seq_id = seq_ids[i]
        seq = sequences[seq_id]
        distances = []
        
        for j, other_id in enumerate(seq_ids):
            if i != j:
                other_seq = sequences[other_id]
                dist = hamming_distance_vectorized(seq, other_seq)
                distances.append(dist)
        
        chunk_distances[seq_id] = np.mean(distances)
    
    return chunk_distances

def sampling_based_outgroup(sequences, sample_size=1000, random_seed=42):
    """
    Find outgroup using random sampling for very large datasets.
    """
    random.seed(random_seed)
    seq_ids = list(sequences.keys())
    
    if len(seq_ids) <= sample_size:
        return None  # Use full method
    
    print(f"Using sampling approach with {sample_size} sequences")
    sampled_ids = random.sample(seq_ids, sample_size)
    sampled_sequences = {sid: sequences[sid] for sid in sampled_ids}
    
    return find_outgroup_memory_efficient(sampled_sequences, use_multiprocessing=True)

def find_outgroup_memory_efficient(sequences, use_multiprocessing=True, chunk_size=None):
    """
    Memory-efficient outgroup finder that doesn't store full distance matrix.
    """
    seq_ids = list(sequences.keys())
    n_sequences = len(seq_ids)
    
    print(f"Processing {n_sequences} sequences...")
    
    if chunk_size is None:
        chunk_size = max(1, n_sequences // mp.cpu_count()) if use_multiprocessing else n_sequences
    
    # Calculate average distances without storing full matrix
    if use_multiprocessing and n_sequences > 100:
        print("Using multiprocessing...")
        
        # Create chunks for parallel processing
        chunks = []
        for i in range(0, n_sequences, chunk_size):
            end_idx = min(i + chunk_size, n_sequences)
            chunks.append((sequences, seq_ids, i, end_idx))
        
        # Process chunks in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            chunk_results = pool.map(calculate_distances_chunk, chunks)
        
        # Combine results
        avg_distances = {}
        for chunk_result in chunk_results:
            avg_distances.update(chunk_result)
    
    else:
        print("Using single-threaded processing...")
        avg_distances = {}
        
        for i, seq_id in enumerate(seq_ids):
            if i % 100 == 0:
                print(f"Processed {i}/{n_sequences} sequences", end='\r')
            
            seq = str(sequences[seq_id])
            distances = []
            
            for j, other_id in enumerate(seq_ids):
                if i != j:
                    other_seq = str(sequences[other_id])
                    dist = hamming_distance_vectorized(seq, other_seq)
                    distances.append(dist)
            
            avg_distances[seq_id] = np.mean(distances)
        
        print()  # New line after progress
    
    # Find outgroup
    outgroup_id = max(avg_distances, key=avg_distances.get)
    outgroup_distance = avg_distances[outgroup_id]
    
    return outgroup_id, outgroup_distance, avg_distances

def quick_outgroup_estimation(sequences, n_comparisons=1000):
    """
    Ultra-fast outgroup estimation using random pairwise comparisons.
    Suitable for initial screening of very large datasets.
    """
    seq_ids = list(sequences.keys())
    n_sequences = len(seq_ids)
    
    if n_sequences < 10:
        return None  # Too small for sampling
    
    print(f"Quick estimation using {n_comparisons} random comparisons...")
    
    # Track cumulative distances for each sequence
    distance_sums = {seq_id: 0.0 for seq_id in seq_ids}
    comparison_counts = {seq_id: 0 for seq_id in seq_ids}
    
    # Perform random pairwise comparisons
    for _ in range(n_comparisons):
        i, j = random.sample(range(n_sequences), 2)
        seq1_id, seq2_id = seq_ids[i], seq_ids[j]
        
        seq1 = str(sequences[seq1_id])
        seq2 = str(sequences[seq2_id])
        
        dist = hamming_distance_vectorized(seq1, seq2)
        
        distance_sums[seq1_id] += dist
        distance_sums[seq2_id] += dist
        comparison_counts[seq1_id] += 1
        comparison_counts[seq2_id] += 1
    
    # Calculate average distances
    avg_distances = {}
    for seq_id in seq_ids:
        if comparison_counts[seq_id] > 0:
            avg_distances[seq_id] = distance_sums[seq_id] / comparison_counts[seq_id]
        else:
            avg_distances[seq_id] = 0.0
    
    outgroup_id = max(avg_distances, key=avg_distances.get)
    outgroup_distance = avg_distances[outgroup_id]
    
    return outgroup_id, outgroup_distance, avg_distances

def consensus_based_distance(sequences):
    """
    Calculate distances using consensus sequence method.
    More efficient for large datasets with many similar sequences.
    """
    seq_ids = list(sequences.keys())
    alignment_length = len(str(sequences[seq_ids[0]]))
    
    print("Calculating consensus sequence...")
    
    # Build consensus sequence
    consensus = []
    for pos in range(alignment_length):
        chars = [str(sequences[seq_id])[pos] for seq_id in seq_ids if str(sequences[seq_id])[pos] != '-']
        if chars:
            consensus_char = Counter(chars).most_common(1)[0][0]
            consensus.append(consensus_char)
        else:
            consensus.append('-')
    
    consensus_seq = ''.join(consensus)
    
    # Calculate distances to consensus
    consensus_distances = {}
    for seq_id in seq_ids:
        seq = str(sequences[seq_id])
        dist = hamming_distance_vectorized(seq, consensus_seq)
        consensus_distances[seq_id] = dist
    
    outgroup_id = max(consensus_distances, key=consensus_distances.get)
    outgroup_distance = consensus_distances[outgroup_id]
    
    return outgroup_id, outgroup_distance, consensus_distances

def main():
    parser = argparse.ArgumentParser(
        description="Scalable outgroup finder for large sequence alignments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Methods:
    full      - Calculate all pairwise distances (default, slow for >1000 seqs)
    sampling  - Random sampling approach (fast, good for >10000 seqs)
    quick     - Ultra-fast estimation using random comparisons
    consensus - Distance to consensus sequence (fast, assumes similar sequences)

Examples:
    python outgroup_finder.py alignment.fasta
    python outgroup_finder.py large_alignment.fasta --method sampling
    python outgroup_finder.py huge_alignment.fasta --method quick
    python outgroup_finder.py alignment.fasta --method consensus --threads 8
        """
    )
    
    parser.add_argument('alignment', help='Path to alignment file')
    parser.add_argument('-f', '--format', default='fasta',
                       choices=['fasta', 'phylip', 'clustal', 'nexus', 'stockholm'],
                       help='Alignment file format (default: fasta)')
    parser.add_argument('-m', '--method', default='auto',
                       choices=['full', 'sampling', 'quick', 'consensus', 'auto'],
                       help='Method to use (default: auto)')
    parser.add_argument('-s', '--sample-size', type=int, default=1000,
                       help='Sample size for sampling method (default: 1000)')
    parser.add_argument('-c', '--comparisons', type=int, default=10000,
                       help='Number of comparisons for quick method (default: 10000)')
    parser.add_argument('--no-multiprocessing', action='store_true',
                       help='Disable multiprocessing')
    parser.add_argument('-o', '--output', help='Output file for results')
    parser.add_argument('--benchmark', action='store_true',
                       help='Show timing information')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Read sequences
    print("Loading sequences...")
    sequences = {}
    try:
        for record in SeqIO.parse(args.alignment, args.format):
            sequences[record.id] = str(record.seq)
    except Exception as e:
        print(f"Error reading alignment: {e}")
        sys.exit(1)
    
    n_sequences = len(sequences)
    print(f"Loaded {n_sequences} sequences")
    
    if n_sequences < 3:
        print("Need at least 3 sequences to identify outgroup")
        sys.exit(1)
    
    # Choose method automatically if requested
    if args.method == 'auto':
        if n_sequences <= 500:
            method = 'full'
        elif n_sequences <= 5000:
            method = 'sampling'
        elif n_sequences <= 20000:
            method = 'quick'
        else:
            method = 'consensus'
        print(f"Auto-selected method: {method}")
    else:
        method = args.method
    
    # Apply selected method
    if method == 'full':
        result = find_outgroup_memory_efficient(
            sequences, 
            use_multiprocessing=not args.no_multiprocessing
        )
    elif method == 'sampling':
        result = sampling_based_outgroup(sequences, args.sample_size)
        if result is None:
            result = find_outgroup_memory_efficient(sequences)
    elif method == 'quick':
        result = quick_outgroup_estimation(sequences, args.comparisons)
    elif method == 'consensus':
        result = consensus_based_distance(sequences)
    
    outgroup_id, outgroup_distance, all_distances = result
    
    # Print results
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("SCALABLE OUTGROUP ANALYSIS RESULTS")
    print("="*60)
    print(f"Method used: {method}")
    print(f"Sequences analyzed: {n_sequences}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Identified outgroup: {outgroup_id}")
    print(f"Distance metric: {outgroup_distance:.2f}")
    
    # Show top candidates
    print(f"\nTop 10 most divergent sequences:")
    print("-" * 45)
    sorted_distances = sorted(all_distances.items(), key=lambda x: x[1], reverse=True)
    
    for i, (seq_id, dist) in enumerate(sorted_distances[:10], 1):
        marker = " <- OUTGROUP" if seq_id == outgroup_id else ""
        print(f"{i:2d}. {seq_id:25s} {dist:8.2f}{marker}")
    
    if args.benchmark:
        sequences_per_second = n_sequences / elapsed_time
        print(f"\nPerformance: {sequences_per_second:.1f} sequences/second")
    
    # Save results if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write("Scalable Outgroup Analysis Results\n")
                f.write("=" * 35 + "\n\n")
                f.write(f"Method: {method}\n")
                f.write(f"Sequences: {n_sequences}\n")
                f.write(f"Processing time: {elapsed_time:.2f}s\n")
                f.write(f"Outgroup: {outgroup_id}\n")
                f.write(f"Distance: {outgroup_distance:.2f}\n\n")
                f.write("All sequences (ranked by distance):\n")
                
                for i, (seq_id, dist) in enumerate(sorted_distances, 1):
                    marker = " <- OUTGROUP" if seq_id == outgroup_id else ""
                    f.write(f"{i:3d}. {seq_id:25s} {dist:8.2f}{marker}\n")
            
            print(f"\nResults saved to: {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()