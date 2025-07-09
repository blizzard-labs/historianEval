#!/usr/bin/env python3
"""
TreeFam Gene Family Downloader

This script downloads FASTA alignments and Newick trees for gene families
from TreeFam database and related resources.

Note: TreeFam is no longer actively maintained, but archived data may still
be accessible. This script also includes alternatives like Ensembl Gene Trees.
"""

import os
import sys
import requests
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import logging
import re
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TreeFamDownloader:
    def __init__(self, output_dir: str = "treefam_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.fasta_dir = self.output_dir / "alignments"
        self.tree_dir = self.output_dir / "trees"
        self.fasta_dir.mkdir(exist_ok=True)
        self.tree_dir.mkdir(exist_ok=True)
        
        # TreeFam URLs (archived versions)
        self.treefam_base = "http://www.treefam.org"
        self.treefam_archive = "https://treefam.genomics.org.cn"
        
        # Alternative: Ensembl Gene Trees (more reliable)
        self.ensembl_rest = "https://rest.ensembl.org"
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TreeFam-Downloader/1.0 (bioinformatics-tool)'
        })
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
    
    def rate_limit(self):
        """Simple rate limiting to be respectful to servers"""
        time.sleep(self.request_delay)
    
    def validate_fasta_content(self, content: str) -> bool:
        """Validate that content is a proper FASTA file"""
        if not content or not content.strip():
            return False
        
        lines = content.strip().split('\n')
        if not lines:
            return False
        
        # Must start with a header
        if not lines[0].startswith('>'):
            return False
        
        # Check for at least one sequence line
        has_sequence = False
        for line in lines[1:]:
            if line.strip() and not line.startswith('>'):
                has_sequence = True
                break
        
        return has_sequence
    
    def validate_newick_content(self, content: str) -> bool:
        """Validate that content is a proper Newick tree"""
        if not content or not content.strip():
            return False
        
        content = content.strip()
        
        # Basic Newick format checks
        if not content.startswith('('):
            return False
        
        if not content.endswith(';'):
            return False
        
        # Check balanced parentheses
        open_count = 0
        for char in content:
            if char == '(':
                open_count += 1
            elif char == ')':
                open_count -= 1
                if open_count < 0:
                    return False
        
        # Should end with balanced parentheses
        if open_count != 0:
            return False
        
        # Should contain at least one leaf (some text between parentheses or commas)
        # This is a simple check - a proper tree should have identifiers
        if not re.search(r'[a-zA-Z0-9_]', content):
            return False
        
        return True
    
    def download_treefam_family(self, family_id: str) -> Dict[str, bool]:
        """Download alignment and tree for a TreeFam family ID"""
        results = {"alignment": False, "tree": False}
        
        try:
            # Try multiple TreeFam endpoints with different URL formats
            treefam_urls = [
                "http://www.treefam.org",
                "https://treefam.genomics.org.cn"
            ]
            
            for base_url in treefam_urls:
                try:
                    # First check if family exists
                    family_url = f"{base_url}/family/{family_id}"
                    family_response = self.session.get(family_url, timeout=30)
                    
                    if family_response.status_code != 200:
                        logger.warning(f"Family {family_id} not found at {base_url}")
                        continue
                    
                    # Try different alignment URL formats
                    alignment_urls = [
                        f"{base_url}/family/{family_id}/alignment",
                        f"{base_url}/family/{family_id}/alignment/fasta",
                        f"{base_url}/family/{family_id}.fasta"
                    ]
                    
                    for alignment_url in alignment_urls:
                        try:
                            logger.debug(f"Trying alignment URL: {alignment_url}")
                            alignment_response = self.session.get(alignment_url, timeout=30)
                            
                            if alignment_response.status_code == 200:
                                content = alignment_response.text
                                if self.validate_fasta_content(content):
                                    alignment_file = self.fasta_dir / f"{family_id}_treefam.fasta"
                                    with open(alignment_file, 'w') as f:
                                        f.write(content)
                                    results["alignment"] = True
                                    logger.info(f"Downloaded alignment for {family_id} from TreeFam")
                                    break
                                else:
                                    logger.debug(f"Invalid FASTA content for {family_id} from {alignment_url}")
                            else:
                                logger.debug(f"Failed to download alignment for {family_id} from {alignment_url}: HTTP {alignment_response.status_code}")
                        except requests.exceptions.RequestException as e:
                            logger.debug(f"Request failed for {alignment_url}: {e}")
                            continue
                        except Exception as e:
                            logger.debug(f"Unexpected error for {alignment_url}: {e}")
                            continue
                    
                    self.rate_limit()
                    
                    # Try different tree URL formats
                    tree_urls = [
                        f"{base_url}/family/{family_id}/tree",
                        f"{base_url}/family/{family_id}/tree/newick",
                        f"{base_url}/family/{family_id}.nh"
                    ]
                    
                    for tree_url in tree_urls:
                        try:
                            logger.debug(f"Trying tree URL: {tree_url}")
                            tree_response = self.session.get(tree_url, timeout=30)
                            
                            if tree_response.status_code == 200:
                                content = tree_response.text
                                if self.validate_newick_content(content):
                                    tree_file = self.tree_dir / f"{family_id}_treefam.newick"
                                    with open(tree_file, 'w') as f:
                                        f.write(content)
                                    results["tree"] = True
                                    logger.info(f"Downloaded tree for {family_id} from TreeFam")
                                    break
                                else:
                                    logger.debug(f"Invalid Newick content for {family_id} from {tree_url}")
                            else:
                                logger.debug(f"Failed to download tree for {family_id} from {tree_url}: HTTP {tree_response.status_code}")
                        except requests.exceptions.RequestException as e:
                            logger.debug(f"Request failed for {tree_url}: {e}")
                            continue
                        except Exception as e:
                            logger.debug(f"Unexpected error for {tree_url}: {e}")
                            continue
                    
                    # If we got at least one file, break from trying other URLs
                    if results["alignment"] or results["tree"]:
                        break
                        
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Failed to connect to {base_url}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error downloading TreeFam family {family_id}: {e}")
        
        return results
    
    def download_treefam_by_gene(self, gene_id: str) -> Dict[str, bool]:
        """Download TreeFam data by gene ID (requires family lookup)"""
        results = {"alignment": False, "tree": False}
        
        try:
            # Try to find TreeFam family ID for this gene
            search_urls = [
                f"{self.treefam_archive}/gene/{gene_id}",
                f"{self.treefam_base}/gene/{gene_id}"
            ]
            
            family_id = None
            for search_url in search_urls:
                try:
                    response = self.session.get(search_url, timeout=30)
                    if response.status_code == 200:
                        # Parse HTML to find family ID
                        content = response.text
                        family_match = re.search(r'family/([^/\s"]+)', content)
                        if family_match:
                            family_id = family_match.group(1)
                            logger.info(f"Found TreeFam family {family_id} for gene {gene_id}")
                            break
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Failed to search for gene {gene_id} at {search_url}: {e}")
                    continue
            
            if family_id:
                results = self.download_treefam_family(family_id)
                
                # Rename files to include gene ID
                if results["alignment"]:
                    old_file = self.fasta_dir / f"{family_id}_treefam.fasta"
                    new_file = self.fasta_dir / f"{gene_id}_{family_id}_treefam.fasta"
                    if old_file.exists():
                        old_file.rename(new_file)
                
                if results["tree"]:
                    old_file = self.tree_dir / f"{family_id}_treefam.newick"
                    new_file = self.tree_dir / f"{gene_id}_{family_id}_treefam.newick"
                    if old_file.exists():
                        old_file.rename(new_file)
            else:
                logger.warning(f"Could not find TreeFam family for gene {gene_id}")
            
        except Exception as e:
            logger.error(f"Error finding TreeFam family for gene {gene_id}: {e}")
        
        return results
    
    def download_ensembl_alternative(self, gene_id: str) -> Dict[str, bool]:
        """Download from Ensembl Gene Trees as alternative to TreeFam"""
        results = {"alignment": False, "tree": False}
        
        try:
            # Get gene tree from Ensembl
            tree_url = f"{self.ensembl_rest}/genetree/id/{gene_id}"
            
            # Download alignment
            alignment_params = {"content-type": "text/x-fasta", "aligned": "1"}
            try:
                alignment_response = self.session.get(tree_url, params=alignment_params, timeout=30)
                
                if alignment_response.status_code == 200:
                    content = alignment_response.text
                    if self.validate_fasta_content(content):
                        alignment_file = self.fasta_dir / f"{gene_id}_ensembl.fasta"
                        with open(alignment_file, 'w') as f:
                            f.write(content)
                        results["alignment"] = True
                        logger.info(f"Downloaded alignment for {gene_id} from Ensembl")
                    else:
                        logger.debug(f"Invalid FASTA content for {gene_id} from Ensembl")
                else:
                    logger.debug(f"Failed to download alignment for {gene_id} from Ensembl: HTTP {alignment_response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.debug(f"Failed to download alignment for {gene_id} from Ensembl: {e}")
            
            self.rate_limit()
            
            # Download tree
            tree_params = {"content-type": "text/x-nh"}
            try:
                tree_response = self.session.get(tree_url, params=tree_params, timeout=30)
                
                if tree_response.status_code == 200:
                    content = tree_response.text
                    if self.validate_newick_content(content):
                        tree_file = self.tree_dir / f"{gene_id}_ensembl.newick"
                        with open(tree_file, 'w') as f:
                            f.write(content)
                        results["tree"] = True
                        logger.info(f"Downloaded tree for {gene_id} from Ensembl")
                    else:
                        logger.debug(f"Invalid Newick content for {gene_id} from Ensembl")
                else:
                    logger.debug(f"Failed to download tree for {gene_id} from Ensembl: HTTP {tree_response.status_code}")
            except requests.exceptions.RequestException as e:
                logger.debug(f"Failed to download tree for {gene_id} from Ensembl: {e}")
            
        except Exception as e:
            logger.error(f"Error downloading from Ensembl for {gene_id}: {e}")
        
        return results
    
    def download_genes(self, gene_list: List[str], use_ensembl_fallback: bool = True) -> Dict[str, Dict[str, bool]]:
        """Download alignments and trees for a list of genes"""
        results = {}
        
        for gene in gene_list:
            logger.info(f"Processing gene: {gene}")
            
            # Try TreeFam first
            gene_results = self.download_treefam_by_gene(gene)
            
            # If TreeFam failed and fallback is enabled, try Ensembl
            if use_ensembl_fallback and not (gene_results["alignment"] or gene_results["tree"]):
                logger.info(f"TreeFam failed for {gene}, trying Ensembl as fallback")
                ensembl_results = self.download_ensembl_alternative(gene)
                
                # Merge results
                if ensembl_results["alignment"]:
                    gene_results["alignment"] = True
                if ensembl_results["tree"]:
                    gene_results["tree"] = True
            
            results[gene] = gene_results
            self.rate_limit()
        
        return results
    
    def download_families(self, family_list: List[str]) -> Dict[str, Dict[str, bool]]:
        """Download alignments and trees for a list of TreeFam family IDs"""
        results = {}
        
        for family_id in family_list:
            logger.info(f"Processing TreeFam family: {family_id}")
            results[family_id] = self.download_treefam_family(family_id)
            self.rate_limit()
        
        return results
    
    def print_summary(self, results: Dict[str, Dict[str, bool]]):
        """Print summary of download results"""
        total_genes = len(results)
        successful_alignments = sum(1 for r in results.values() if r["alignment"])
        successful_trees = sum(1 for r in results.values() if r["tree"])
        
        print(f"\n{'='*50}")
        print(f"DOWNLOAD SUMMARY")
        print(f"{'='*50}")
        print(f"Total genes/families processed: {total_genes}")
        print(f"Successful alignments: {successful_alignments}/{total_genes}")
        print(f"Successful trees: {successful_trees}/{total_genes}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*50}")
        
        # Print detailed results
        alignment_only = [gene for gene, result in results.items() 
                         if result["alignment"] and not result["tree"]]
        tree_only = [gene for gene, result in results.items() 
                    if result["tree"] and not result["alignment"]]
        both = [gene for gene, result in results.items() 
               if result["alignment"] and result["tree"]]
        neither = [gene for gene, result in results.items() 
                  if not result["alignment"] and not result["tree"]]
        
        if both:
            print(f"Both alignment and tree: {', '.join(both)}")
        if alignment_only:
            print(f"Alignment only: {', '.join(alignment_only)}")
        if tree_only:
            print(f"Tree only: {', '.join(tree_only)}")
        if neither:
            print(f"Failed downloads: {', '.join(neither)}")

def main():
    parser = argparse.ArgumentParser(description="Download gene family alignments and trees from TreeFam")
    parser.add_argument("--genes", nargs="+", help="List of gene IDs")
    parser.add_argument("--families", nargs="+", help="List of TreeFam family IDs")
    parser.add_argument("--gene-file", help="File containing gene IDs (one per line)")
    parser.add_argument("--family-file", help="File containing TreeFam family IDs (one per line)")
    parser.add_argument("--output-dir", default="treefam_data", help="Output directory")
    parser.add_argument("--no-ensembl-fallback", action="store_true", 
                       help="Don't use Ensembl as fallback if TreeFam fails")
    parser.add_argument("--delay", type=float, default=1.0, 
                       help="Delay between requests in seconds")
    parser.add_argument("--verbose", action="store_true", 
                       help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Collect gene/family lists
    genes = []
    families = []
    
    if args.genes:
        genes.extend(args.genes)
    
    if args.families:
        families.extend(args.families)
    
    if args.gene_file:
        with open(args.gene_file, 'r') as f:
            genes.extend(line.strip() for line in f if line.strip())
    
    if args.family_file:
        with open(args.family_file, 'r') as f:
            families.extend(line.strip() for line in f if line.strip())
    
    if not genes and not families:
        print("Please provide genes or families to download")
        print("Example usage:")
        print("  python treefam_downloader.py --genes ENSG00000139618 ENSG00000141510")
        print("  python treefam_downloader.py --families TF101001 TF101002")
        print("  python treefam_downloader.py --gene-file genes.txt")
        sys.exit(1)
    
    # Initialize downloader
    downloader = TreeFamDownloader(output_dir=args.output_dir)
    downloader.request_delay = args.delay
    
    # Download data
    all_results = {}
    
    if genes:
        logger.info(f"Downloading data for {len(genes)} genes")
        gene_results = downloader.download_genes(genes, not args.no_ensembl_fallback)
        all_results.update(gene_results)
    
    if families:
        logger.info(f"Downloading data for {len(families)} TreeFam families")
        family_results = downloader.download_families(families)
        all_results.update(family_results)
    
    # Print summary
    downloader.print_summary(all_results)

if __name__ == "__main__":
    main()