#!/usr/bin/env python3
"""
MCMC Trace File Merger

This script merges two MCMC trace files:
- treetrace.log: Contains tree topology and phylogenetic parameters
- parsed_trace.log: Contains likelihood and substitution parameters

The output is saved as trace.log with all columns combined.
"""

import pandas as pd
import sys
import os

def read_trace_file(filename, sep='\t'):
    """
    Read a trace file and return a pandas DataFrame.
    Handles both tab and space-separated files.
    """
    try:
        # First, try reading with the specified separator
        df = pd.read_csv(filename, sep=sep)
        return df
    except Exception as e:
        print(f"Error reading {filename} with separator '{sep}': {e}")
        # Try with whitespace separator as fallback
        try:
            df = pd.read_csv(filename, sep=r'\s+', engine='python')
            return df
        except Exception as e2:
            print(f"Error reading {filename} with whitespace separator: {e2}")
            return None

def merge_trace_files(treetrace_file, parsed_trace_file, output_file='trace.log'):
    """
    Merge two MCMC trace files horizontally by matching state/iter numbers.
    
    Parameters:
    - treetrace_file: Path to the tree trace file
    - parsed_trace_file: Path to the parsed trace file  
    - output_file: Output filename (default: 'trace.log')
    """
    
    print(f"Reading {treetrace_file}...")
    treetrace_df = read_trace_file(treetrace_file)
    if treetrace_df is None:
        print(f"Failed to read {treetrace_file}")
        return False
    
    print(f"Reading {parsed_trace_file}...")
    parsed_df = read_trace_file(parsed_trace_file)
    if parsed_df is None:
        print(f"Failed to read {parsed_trace_file}")
        return False
    
    print(f"Tree trace shape: {treetrace_df.shape}")
    print(f"Parsed trace shape: {parsed_df.shape}")
    
    print(f"Tree trace columns: {list(treetrace_df.columns)}")
    print(f"Parsed trace columns: {list(parsed_df.columns)}")
    
    # Standardize the iteration column names for merging
    tree_merge_col = None
    parsed_merge_col = None
    
    # For indels
    indels_col = "indels" if "indels" in parsed_df.columns else "#indels" if "#indels" in parsed_df.columns else None

    # For substitutions
    substs_col = "substitutions" if "substitutions" in parsed_df.columns else "#substs" if "#substs" in parsed_df.columns else None

    # Select columns (add other columns as needed)
    selected_cols = ["iter", 'likelihood']
    if indels_col:
        selected_cols.append(indels_col)
    if substs_col:
        selected_cols.append(substs_col)

    parsed_df = parsed_df[selected_cols]
    treetrace_df = treetrace_df[['state', 'CCD1 RF distance', 'CCD1 information content (log(p))', 'Colless tree-imbalance', 'Tree Length', 'Tree Topology']]
    
    
    # Find the iteration column in each file
    if 'state' in treetrace_df.columns:
        tree_merge_col = 'state'
    elif 'iter' in treetrace_df.columns:
        tree_merge_col = 'iter'
    
    if 'iter' in parsed_df.columns:
        parsed_merge_col = 'iter'
    elif 'state' in parsed_df.columns:
        parsed_merge_col = 'state'
    
    if tree_merge_col is None or parsed_merge_col is None:
        print(f"Error: Cannot find iteration columns. Tree: {tree_merge_col}, Parsed: {parsed_merge_col}")
        return False
    
    print(f"Merging on: tree file '{tree_merge_col}' = parsed file '{parsed_merge_col}'")
    
    # If column names are different, rename one to match
    if tree_merge_col != parsed_merge_col:
        if tree_merge_col == 'state':
            treetrace_df = treetrace_df.rename(columns={'state': 'iter'})
            tree_merge_col = 'iter'
            print("Renamed 'state' to 'iter' in tree trace file")
        else:
            parsed_df = parsed_df.rename(columns={'state': 'iter'})
            parsed_merge_col = 'iter'
            print("Renamed 'state' to 'iter' in parsed trace file")
    
    # Perform the horizontal merge (join by iteration number)
    try:
        print("Merging files horizontally by iteration number...")
        
        merged_df = pd.merge(parsed_df, treetrace_df, on=tree_merge_col, how='inner')
        print(f"Merged trace shape: {merged_df.shape}")
        
        # Save the merged file
        merged_df.to_csv(output_file, sep='\t', index=False)
        print(f"Successfully saved merged trace to: {output_file}")
        
        # Print summary statistics
        print("\n--- Summary ---")
        print(f"Total rows in merged file: {len(merged_df)}")
        print(f"Total columns in merged file: {len(merged_df.columns)}")
        print(f"Columns from parsed trace: {len(parsed_df.columns)}")
        print(f"Columns from tree trace: {len(treetrace_df.columns) - 1}")  # -1 because merge column is shared
        print(f"Sample of first few iteration numbers: {list(merged_df[tree_merge_col].head(10))}")
        
        return True
        
    except Exception as e:
        print(f"Error during merging: {e}")
        return False

def main():
    """Main function to run the trace file merger."""
    if len(sys.argv) < 4:
        print("Usage: python clean_treestat.py <treetrace> <parsed_trace> <output_trace>")
    
    # Default file names based on your images
    treetrace_file = sys.argv[1]
    parsed_trace_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Check if files exist
    if not os.path.exists(treetrace_file):
        print(f"Error: {treetrace_file} not found!")
        print("Please make sure the file exists in the current directory.")
        return
    
    if not os.path.exists(parsed_trace_file):
        print(f"Error: {parsed_trace_file} not found!")
        print("Please make sure the file exists in the current directory.")
        return
    
    print("MCMC Trace File Merger (Horizontal Join)")
    print("=" * 50)
    
    # Perform the merge
    success = merge_trace_files(treetrace_file, parsed_trace_file, output_file)
    
    if success:
        print("\n✓ Merge completed successfully!")
        print(f"Output saved as: {output_file}")
    else:
        print("\n✗ Merge failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
