#!/usr/bin/env python3
"""
Phylogenetic Data Partitioner

This script splits a folder containing alignments and trees into balanced partitions
while keeping complementary alignment (.fasta) and tree (.tree) files paired together.

Usage:
    python partition_phylo_data.py <main_folder> <num_partitions>

Expected folder structure:
main_folder/
├── alignments/
│   ├── file1.fasta
│   ├── file2.fasta
│   └── ...
└── trees/
    ├── file1.tree
    ├── file2.tree
    └── ...
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Set


def find_paired_files(alignments_dir: Path, trees_dir: Path) -> List[str]:
    """
    Find files that have both alignment (.fasta) and tree (.tree) versions.
    
    Args:
        alignments_dir: Path to alignments folder
        trees_dir: Path to trees folder
        
    Returns:
        List of base filenames (without extensions) that have both files
    """
    if not alignments_dir.exists():
        raise FileNotFoundError(f"Alignments directory not found: {alignments_dir}")
    
    if not trees_dir.exists():
        raise FileNotFoundError(f"Trees directory not found: {trees_dir}")
    
    # Get base names of alignment files (without .fasta extension)
    alignment_files = set()
    for file in alignments_dir.glob("*.fasta"):
        alignment_files.add(file.stem)
    
    # Get base names of tree files (without .tree extension)
    tree_files = set()
    for file in trees_dir.glob("*.tree"):
        tree_files.add(file.stem)
    
    # Find intersection (files that have both alignment and tree)
    paired_files = list(alignment_files & tree_files)
    
    if not paired_files:
        raise ValueError("No paired files found! Make sure alignment and tree files have matching base names.")
    
    # Report any unpaired files
    alignment_only = alignment_files - tree_files
    tree_only = tree_files - alignment_files
    
    if alignment_only:
        print(f"Warning: {len(alignment_only)} alignment files have no corresponding tree file:")
        for file in sorted(alignment_only):
            print(f"  - {file}.fasta")
    
    if tree_only:
        print(f"Warning: {len(tree_only)} tree files have no corresponding alignment file:")
        for file in sorted(tree_only):
            print(f"  - {file}.tree")
    
    return sorted(paired_files)


def create_partitions(paired_files: List[str], num_partitions: int) -> List[List[str]]:
    """
    Distribute files evenly across partitions.
    
    Args:
        paired_files: List of base filenames
        num_partitions: Number of partitions to create
        
    Returns:
        List of lists, where each inner list contains filenames for one partition
    """
    if num_partitions <= 0:
        raise ValueError("Number of partitions must be positive")
    
    if num_partitions > len(paired_files):
        raise ValueError(f"Cannot create {num_partitions} partitions from {len(paired_files)} file pairs")
    
    partitions = [[] for _ in range(num_partitions)]
    
    # Distribute files round-robin style for even distribution
    for i, filename in enumerate(paired_files):
        partition_idx = i % num_partitions
        partitions[partition_idx].append(filename)
    
    return partitions


def create_partition_directories(main_folder: Path, num_partitions: int) -> List[Path]:
    """
    Create partition directories with alignments and trees subfolders.
    
    Args:
        main_folder: Path to main folder
        num_partitions: Number of partitions
        
    Returns:
        List of partition directory paths
    """
    partition_dirs = []
    
    for i in range(num_partitions):
        partition_name = f"partition_{i+1:03d}"  # e.g., partition_001, partition_002
        partition_dir = main_folder / partition_name
        
        # Create partition directory and subdirectories
        partition_dir.mkdir(exist_ok=True)
        (partition_dir / "alignments").mkdir(exist_ok=True)
        (partition_dir / "trees").mkdir(exist_ok=True)
        
        partition_dirs.append(partition_dir)
    
    return partition_dirs


def copy_files_to_partitions(main_folder: Path, partitions: List[List[str]], 
                           partition_dirs: List[Path]) -> None:
    """
    Copy alignment and tree files to their respective partitions.
    
    Args:
        main_folder: Path to main folder containing original alignments and trees
        partitions: List of file lists for each partition
        partition_dirs: List of partition directory paths
    """
    alignments_dir = main_folder / "alignments"
    trees_dir = main_folder / "trees"
    
    for partition_idx, (file_list, partition_dir) in enumerate(zip(partitions, partition_dirs)):
        print(f"\nCopying files to partition {partition_idx + 1}:")
        
        for filename in file_list:
            # Copy alignment file
            src_alignment = alignments_dir / f"{filename}.fasta"
            dst_alignment = partition_dir / "alignments" / f"{filename}.fasta"
            shutil.copy2(src_alignment, dst_alignment)
            
            # Copy tree file
            src_tree = trees_dir / f"{filename}.tree"
            dst_tree = partition_dir / "trees" / f"{filename}.tree"
            shutil.copy2(src_tree, dst_tree)
            
            print(f"  - {filename} (alignment + tree)")


def print_summary(partitions: List[List[str]], total_files: int) -> None:
    """Print summary of the partitioning operation."""
    print(f"\n{'='*60}")
    print("PARTITIONING SUMMARY")
    print(f"{'='*60}")
    print(f"Total file pairs processed: {total_files}")
    print(f"Number of partitions created: {len(partitions)}")
    print()
    
    for i, partition in enumerate(partitions):
        print(f"Partition {i+1:3d}: {len(partition):4d} file pairs")
    
    # Show distribution balance
    sizes = [len(p) for p in partitions]
    min_size, max_size = min(sizes), max(sizes)
    print(f"\nDistribution: {min_size} to {max_size} files per partition")
    if max_size - min_size <= 1:
        print("✓ Well balanced distribution")
    else:
        print("⚠ Uneven distribution (expected with this number of partitions)")


def main():
    parser = argparse.ArgumentParser(
        description="Split phylogenetic data into balanced partitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python partition_phylo_data.py /path/to/data 5

This will create 5 partitions: partition_001, partition_002, ..., partition_005
Each partition will contain alignments/ and trees/ subfolders with paired files.
        """
    )
    
    parser.add_argument("main_folder", type=str, 
                       help="Path to folder containing 'alignments' and 'trees' subfolders")
    parser.add_argument("num_partitions", type=int, 
                       help="Number of partitions to create")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without actually copying files")
    
    args = parser.parse_args()
    
    try:
        main_folder = Path(args.main_folder)
        
        if not main_folder.exists():
            raise FileNotFoundError(f"Main folder not found: {main_folder}")
        
        print(f"Processing folder: {main_folder}")
        print(f"Creating {args.num_partitions} partitions...")
        
        # Find paired files
        alignments_dir = main_folder / "alignments"
        trees_dir = main_folder / "trees"
        paired_files = find_paired_files(alignments_dir, trees_dir)
        
        print(f"\nFound {len(paired_files)} paired alignment/tree files")
        
        # Create partitions
        partitions = create_partitions(paired_files, args.num_partitions)
        
        if args.dry_run:
            print("\n--- DRY RUN MODE ---")
            print_summary(partitions, len(paired_files))
            print("\nFiles would be copied to:")
            for i in range(args.num_partitions):
                partition_name = f"partition_{i+1:03d}"
                print(f"  {main_folder / partition_name}")
            return
        
        # Create partition directories
        partition_dirs = create_partition_directories(main_folder, args.num_partitions)
        
        # Copy files
        copy_files_to_partitions(main_folder, partitions, partition_dirs)
        
        # Print summary
        print_summary(partitions, len(paired_files))
        
        print(f"\n✓ Successfully created {args.num_partitions} partitions in:")
        print(f"  {main_folder}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()