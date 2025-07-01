#!/usr/bin/env python3
"""
Maximum Likelihood Phylogenetic Tree Generator using BioPython PAML Interface

This script generates maximum likelihood phylogenetic trees from sequence alignments
using BioPython's PAML interface (baseml for DNA, codeml for codons, and external
tools for initial tree estimation).

Requirements:
- BioPython
- PAML package installed and available in PATH
- Optional: FastTree or RAxML for initial tree estimation

Usage:
    python ml_tree_generator.py --input_dir alignments/ --output_dir trees/
"""

import os
import sys
import argparse
import subprocess
import shutil
import tempfile
from pathlib import Path
from Bio import AlignIO, SeqIO
from Bio.Phylo.PAML import baseml, codeml
from Bio.Align import MultipleSeqAlignment
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PAMLTreeGenerator:
    def __init__(self, input_dir, output_dir, analysis_type='baseml', genetic_code=1, 
                 runmode=0, model=0, threads=1):
        """
        Initialize the PAML ML tree generator.
        
        Args:
            input_dir (str): Directory containing alignment files
            output_dir (str): Directory to save output trees
            analysis_type (str): Type of analysis ('baseml' for DNA, 'codeml' for codons)
            genetic_code (int): Genetic code table (for codeml)
            runmode (int): PAML runmode (0=user tree, 1=simultaneous estimation)
            model (int): Substitution model
            threads (int): Number of CPU threads to use
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.analysis_type = analysis_type.lower()
        self.genetic_code = genetic_code
        self.runmode = runmode
        self.model = model
        self.threads = threads
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported alignment formats
        self.supported_formats = {
            '.fasta': 'fasta',
            '.fas': 'fasta',
            '.fa': 'fasta',
            '.phy': 'phylip-relaxed',
            '.phylip': 'phylip-relaxed',
            '.nex': 'nexus',
            '.nexus': 'nexus'
        }
        
    def check_dependencies(self):
        """Check if required tools are installed."""
        required_tools = []
        
        if self.analysis_type == 'baseml':
            required_tools.append('baseml')
        elif self.analysis_type == 'codeml':
            required_tools.append('codeml')
            
        # Check for tree estimation tools
        tree_tools = ['fasttree', 'FastTree', 'raxml', 'iqtree']
        
        missing_tools = []
        available_tree_tool = None
        
        for tool in required_tools:
            if not shutil.which(tool):
                missing_tools.append(tool)
                
        for tool in tree_tools:
            if shutil.which(tool):
                available_tree_tool = tool
                break
                
        if missing_tools:
            logger.error(f"Missing required tools: {', '.join(missing_tools)}")
            logger.error("Please install PAML package")
            return False, None
            
        if not available_tree_tool:
            logger.warning("No tree estimation tool found. Will use simple distance-based trees.")
            
        logger.info(f"Found PAML tools. Tree estimation tool: {available_tree_tool}")
        return True, available_tree_tool
        
    def detect_sequence_type(self, alignment):
        """
        Detect if sequences are DNA, RNA, or protein.
        
        Args:
            alignment: BioPython alignment object
            
        Returns:
            str: 'dna', 'rna', or 'protein'
        """
        seq_sample = str(alignment[0].seq).upper().replace('-', '').replace('N', '')
        
        if len(seq_sample) == 0:
            return 'dna'  # default
            
        # Count nucleotide characters
        nucleotide_chars = set('ATCGU')
        nucleotide_count = sum(1 for c in seq_sample if c in nucleotide_chars)
        nucleotide_ratio = nucleotide_count / len(seq_sample)
        
        if nucleotide_ratio > 0.85:
            if 'U' in seq_sample:
                return 'rna'
            else:
                return 'dna'
        else:
            return 'protein'
            
    def is_codon_alignment(self, alignment):
        """
        Check if alignment is suitable for codon analysis.
        
        Args:
            alignment: BioPython alignment object
            
        Returns:
            bool: True if appears to be codon alignment
        """
        # Check if alignment length is multiple of 3
        if len(alignment[0]) % 3 != 0:
            return False
            
        # Check if sequences contain only valid nucleotides
        seq_type = self.detect_sequence_type(alignment)
        if seq_type not in ['dna', 'rna']:
            return False
            
        # Additional checks could be added here (e.g., check for stop codons)
        return True
        
    def generate_initial_tree(self, alignment_file, tree_tool=None):
        """
        Generate an initial tree using external tools.
        
        Args:
            alignment_file (Path): Path to alignment file
            tree_tool (str): Tree estimation tool to use
            
        Returns:
            str: Path to generated tree file or None
        """
        if not tree_tool:
            return None
            
        output_tree = self.output_dir / f"{alignment_file.stem}_initial.tre"
        
        try:
            if tree_tool.lower() in ['fasttree', 'FastTree']:
                cmd = [tree_tool, '-nt', str(alignment_file)]
                with open(output_tree, 'w') as f:
                    result = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, text=True)
                    
            elif tree_tool.lower() == 'iqtree':
                temp_prefix = self.output_dir / f"{alignment_file.stem}_temp"
                cmd = [
                    'iqtree', '-s', str(alignment_file), '-pre', str(temp_prefix),
                    '-m', 'GTR+G', '-quiet', '-nt', '1'
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Move the tree file
                iqtree_output = Path(f"{temp_prefix}.treefile")
                if iqtree_output.exists():
                    shutil.move(str(iqtree_output), str(output_tree))
                    # Clean up other IQ-TREE files
                    for ext in ['.iqtree', '.log', '.bionj', '.mldist', '.model.gz']:
                        temp_file = Path(f"{temp_prefix}{ext}")
                        if temp_file.exists():
                            temp_file.unlink()
                            
            if output_tree.exists() and output_tree.stat().st_size > 0:
                logger.info(f"Generated initial tree: {output_tree}")
                return str(output_tree)
            else:
                logger.warning(f"Failed to generate initial tree for {alignment_file}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating initial tree: {e}")
            return None
            
    def prepare_phylip_alignment(self, alignment_file):
        """
        Convert alignment to PHYLIP format required by PAML.
        
        Args:
            alignment_file (Path): Input alignment file
            
        Returns:
            str: Path to PHYLIP formatted file
        """
        try:
            # Determine input format
            file_ext = alignment_file.suffix.lower()
            if file_ext not in self.supported_formats:
                return None
                
            format_name = self.supported_formats[file_ext]
            alignment = AlignIO.read(alignment_file, format_name)
            
            # Create PHYLIP output file
            phylip_file = self.output_dir / f"{alignment_file.stem}.phy"
            
            # Write in PHYLIP format with proper spacing
            with open(phylip_file, 'w') as f:
                f.write(f"  {len(alignment)}  {len(alignment[0])}\n")
                for record in alignment:
                    # PAML requires specific formatting
                    seq_id = record.id[:10].ljust(10)  # Limit to 10 chars, pad with spaces
                    f.write(f"{seq_id}  {str(record.seq)}\n")
                    
            return str(phylip_file)
            
        except Exception as e:
            logger.error(f"Error preparing PHYLIP alignment: {e}")
            return None
            
    def run_baseml(self, alignment_file, tree_file=None):
        """
        Run BASEML for DNA sequence analysis.
        
        Args:
            alignment_file (str): Path to PHYLIP alignment file
            tree_file (str): Path to tree file (optional)
            
        Returns:
            bool: True if successful
        """
        try:
            # Create working directory for this analysis
            work_dir = self.output_dir / f"{Path(alignment_file).stem}_baseml"
            work_dir.mkdir(exist_ok=True)
            
            # Initialize baseml
            bml = baseml.Baseml()
            
            # Set alignment file
            bml.alignment = alignment_file
            
            # Set tree file if provided
            if tree_file and Path(tree_file).exists():
                bml.tree = tree_file
                bml.set_options(runmode=0)  # User tree
            else:
                bml.set_options(runmode=1)  # Simultaneous estimation
                
            # Set working directory
            bml.working_dir = str(work_dir)
            
            # Configure baseml options
            bml.set_options(
                model=self.model,  # 0=JC69, 1=K80, 2=F81, 3=F84, 4=HKY85, 5=T92, 6=TN93, 7=REV
                alpha=0.5,         # Alpha parameter for gamma distribution
                ncatG=4,           # Number of categories for gamma
                clock=0,           # No molecular clock
                fix_alpha=0,       # Estimate alpha
                fix_kappa=0,       # Estimate kappa
                get_SE=1,          # Get standard errors
                RateAncestor=0,    # Don't infer ancestral sequences
                Small_Diff=0.5e-6, # Small difference for optimization
            )
            
            # Run baseml
            logger.info(f"Running BASEML for {Path(alignment_file).name}")
            results = bml.run()
            
            # Copy results to main output directory
            main_output = self.output_dir / f"{Path(alignment_file).stem}_baseml"
            
            # Copy tree file if it exists
            tree_output = work_dir / "2base.t"
            if tree_output.exists():
                shutil.copy(str(tree_output), str(main_output) + ".tre")
                
            # Copy results file
            results_file = work_dir / "2base.out"
            if results_file.exists():
                shutil.copy(str(results_file), str(main_output) + ".out")
                
            logger.info(f"BASEML completed for {Path(alignment_file).name}")
            return True
            
        except Exception as e:
            logger.error(f"Error running BASEML: {e}")
            return False
            
    def run_codeml(self, alignment_file, tree_file=None):
        """
        Run CODEML for codon sequence analysis.
        
        Args:
            alignment_file (str): Path to PHYLIP alignment file
            tree_file (str): Path to tree file (optional)
            
        Returns:
            bool: True if successful
        """
        try:
            # Create working directory for this analysis
            work_dir = self.output_dir / f"{Path(alignment_file).stem}_codeml"
            work_dir.mkdir(exist_ok=True)
            
            # Initialize codeml
            cml = codeml.Codeml()
            
            # Set alignment file
            cml.alignment = alignment_file
            
            # Set tree file if provided
            if tree_file and Path(tree_file).exists():
                cml.tree = tree_file
                cml.set_options(runmode=0)  # User tree
            else:
                cml.set_options(runmode=1)  # Simultaneous estimation
                
            # Set working directory
            cml.working_dir = str(work_dir)
            
            # Configure codeml options
            cml.set_options(
                seqtype=1,         # Codon sequences
                model=self.model,  # 0=one ratio, 1=free ratio, 2=2 or more ratios
                NSsites=[0],       # Sites models
                icode=self.genetic_code,  # Genetic code
                fix_kappa=0,       # Estimate kappa
                kappa=2,           # Initial kappa value
                fix_omega=0,       # Estimate omega
                omega=0.4,         # Initial omega value
                fix_alpha=1,       # Fix alpha
                alpha=0,           # Alpha value
                ncatG=4,           # Number of categories for gamma
                clock=0,           # No molecular clock
                get_SE=1,          # Get standard errors
                RateAncestor=0,    # Don't infer ancestral sequences
                Small_Diff=0.5e-6, # Small difference for optimization
            )
            
            # Run codeml
            logger.info(f"Running CODEML for {Path(alignment_file).name}")
            results = cml.run()
            
            # Copy results to main output directory
            main_output = self.output_dir / f"{Path(alignment_file).stem}_codeml"
            
            # Copy tree file if it exists
            tree_output = work_dir / "2ML.t"
            if tree_output.exists():
                shutil.copy(str(tree_output), str(main_output) + ".tre")
                
            # Copy results file
            results_file = work_dir / "2ML.out"
            if results_file.exists():
                shutil.copy(str(results_file), str(main_output) + ".out")
                
            logger.info(f"CODEML completed for {Path(alignment_file).name}")
            return True
            
        except Exception as e:
            logger.error(f"Error running CODEML: {e}")
            return False
            
    def validate_alignment(self, alignment_file):
        """
        Validate and load alignment file.
        
        Args:
            alignment_file (Path): Path to alignment file
            
        Returns:
            MultipleSeqAlignment or None: Loaded alignment or None if invalid
        """
        try:
            # Determine format from file extension
            file_ext = alignment_file.suffix.lower()
            if file_ext not in self.supported_formats:
                logger.warning(f"Unsupported file format: {alignment_file}")
                return None
                
            format_name = self.supported_formats[file_ext]
            alignment = AlignIO.read(alignment_file, format_name)
            
            # Basic validation
            if len(alignment) < 3:
                logger.warning(f"Alignment has fewer than 3 sequences: {alignment_file}")
                return None
                
            if len(alignment[0]) < 10:
                logger.warning(f"Alignment is too short: {alignment_file}")
                return None
                
            # Check for analysis type compatibility
            seq_type = self.detect_sequence_type(alignment)
            
            if self.analysis_type == 'codeml':
                if not self.is_codon_alignment(alignment):
                    logger.warning(f"Alignment not suitable for codon analysis: {alignment_file}")
                    return None
            elif self.analysis_type == 'baseml':
                if seq_type == 'protein':
                    logger.warning(f"Protein alignment not suitable for BASEML: {alignment_file}")
                    return None
                    
            logger.info(f"Loaded alignment: {alignment_file} ({len(alignment)} sequences, {len(alignment[0])} sites, {seq_type})")
            return alignment
            
        except Exception as e:
            logger.error(f"Error loading alignment {alignment_file}: {e}")
            return None
            
    def process_alignments(self):
        """Process all alignment files in the input directory."""
        if not self.input_dir.exists():
            logger.error(f"Input directory does not exist: {self.input_dir}")
            return
            
        # Check dependencies
        deps_ok, tree_tool = self.check_dependencies()
        if not deps_ok:
            return
            
        # Find all alignment files
        alignment_files = []
        for ext in self.supported_formats.keys():
            alignment_files.extend(self.input_dir.glob(f'*{ext}'))
            
        if not alignment_files:
            logger.warning("No alignment files found in input directory")
            return
            
        logger.info(f"Found {len(alignment_files)} alignment files")
        
        successful = 0
        failed = 0
        
        for alignment_file in alignment_files:
            logger.info(f"Processing: {alignment_file.name}")
            
            # Validate alignment
            alignment = self.validate_alignment(alignment_file)
            if alignment is None:
                failed += 1
                continue
                
            # Prepare PHYLIP format alignment
            phylip_file = self.prepare_phylip_alignment(alignment_file)
            if phylip_file is None:
                failed += 1
                continue
                
            # Generate initial tree if possible
            tree_file = None
            if tree_tool and self.runmode == 0:
                tree_file = self.generate_initial_tree(alignment_file, tree_tool)
                
            # Run PAML analysis
            try:
                if self.analysis_type == 'baseml':
                    success = self.run_baseml(phylip_file, tree_file)
                elif self.analysis_type == 'codeml':
                    success = self.run_codeml(phylip_file, tree_file)
                else:
                    logger.error(f"Unknown analysis type: {self.analysis_type}")
                    success = False
                    
                if success:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing {alignment_file.name}: {e}")
                failed += 1
                
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
        # Summary of output files
        if successful > 0:
            logger.info(f"Output files saved in: {self.output_dir}")
            logger.info("Look for files with extensions: .tre (trees), .out (detailed results)")


def main():
    """Main function to parse arguments and run the tree generator."""
    parser = argparse.ArgumentParser(
        description='Generate maximum likelihood phylogenetic trees using BioPython PAML interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic DNA analysis with BASEML
    python ml_tree_generator.py -i alignments/ -o trees/ -a baseml
    
    # Codon analysis with CODEML
    python ml_tree_generator.py -i alignments/ -o trees/ -a codeml
    
    # Simultaneous tree and parameter estimation
    python ml_tree_generator.py -i alignments/ -o trees/ -a baseml --runmode 1
    
    # Use specific substitution model
    python ml_tree_generator.py -i alignments/ -o trees/ -a baseml --model 4
        """
    )
    
    parser.add_argument('-i', '--input_dir', required=True,
                        help='Directory containing alignment files')
    parser.add_argument('-o', '--output_dir', required=True,
                        help='Directory to save output trees and results')
    parser.add_argument('-a', '--analysis_type', choices=['baseml', 'codeml'], default='baseml',
                        help='Type of PAML analysis (default: baseml)')
    parser.add_argument('--genetic_code', type=int, default=1,
                        help='Genetic code table for CODEML (default: 1 = standard)')
    parser.add_argument('--runmode', type=int, choices=[0, 1], default=0,
                        help='PAML runmode: 0=user tree, 1=simultaneous estimation (default: 0)')
    parser.add_argument('--model', type=int, default=0,
                        help='Substitution model (default: 0)')
    parser.add_argument('-t', '--threads', type=int, default=1,
                        help='Number of CPU threads for initial tree estimation (default: 1)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run tree generator
    generator = PAMLTreeGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        analysis_type=args.analysis_type,
        genetic_code=args.genetic_code,
        runmode=args.runmode,
        model=args.model,
        threads=args.threads
    )
    
    generator.process_alignments()


if __name__ == '__main__':
    main()