import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from typing import Optional, Dict, List
import time

class TreeFamWeb:
    """Web scraping interface for TreeFam database"""
    
    def __init__(self, base_url: str = "https://www.treefam.org"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Python TreeFam Client)'
        })
    
    def search_families(self, query: str) -> List[Dict]:
        """Search for gene families"""
        search_url = f"{self.base_url}/search"
        params = {'q': query}
        
        try:
            response = self.session.get(search_url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Parse search results (adapt based on actual HTML structure)
            for result in soup.find_all('div', class_='search-result'):
                family_id = result.find('a', href=re.compile(r'/family/'))
                if family_id:
                    family_id = family_id.get('href').split('/')[-1]
                    description = result.find('span', class_='description')
                    description = description.text.strip() if description else ""
                    
                    results.append({
                        'family_id': family_id,
                        'description': description,
                        'url': f"{self.base_url}/family/{family_id}"
                    })
            
            return results
            
        except requests.RequestException as e:
            print(f"Error searching families: {e}")
            return []
    
    def get_family_details(self, family_id: str) -> Dict:
        """Get detailed information about a family"""
        family_url = f"{self.base_url}/family/{family_id}"
        
        try:
            response = self.session.get(family_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract family information
            family_info = {
                'family_id': family_id,
                'url': family_url,
                'description': '',
                'size': 0,
                'genes': []
            }
            
            # Find description
            desc_elem = soup.find('div', class_='family-description')
            if desc_elem:
                family_info['description'] = desc_elem.text.strip()
            
            # Find gene count
            size_elem = soup.find('span', class_='family-size')
            if size_elem:
                family_info['size'] = int(re.search(r'\d+', size_elem.text).group())
            
            # Extract gene list
            gene_table = soup.find('table', class_='gene-table')
            if gene_table:
                for row in gene_table.find_all('tr')[1:]:  # Skip header
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        gene_info = {
                            'gene_id': cells[0].text.strip(),
                            'gene_name': cells[1].text.strip(),
                            'species': cells[2].text.strip()
                        }
                        family_info['genes'].append(gene_info)
            
            return family_info
            
        except requests.RequestException as e:
            print(f"Error getting family details: {e}")
            return {}
    
    def get_phylogenetic_tree(self, family_id: str, format: str = 'newick') -> Optional[str]:
        """Download phylogenetic tree for a family"""
        tree_url = f"{self.base_url}/family/{family_id}/tree"
        params = {'format': format}
        
        try:
            response = self.session.get(tree_url, params=params)
            response.raise_for_status()
            
            if format == 'newick':
                return response.text.strip()
            else:
                return response.content
                
        except requests.RequestException as e:
            print(f"Error getting tree: {e}")
            return None
    
    def get_sequence_alignment(self, family_id: str, format: str = 'fasta') -> Optional[str]:
        """Download sequence alignment for a family"""
        alignment_url = f"{self.base_url}/family/{family_id}/alignment"
        params = {'format': format}
        
        try:
            response = self.session.get(alignment_url, params=params)
            response.raise_for_status()
            
            return response.text
            
        except requests.RequestException as e:
            print(f"Error getting alignment: {e}")
            return None
    
    def batch_family_info(self, family_ids: List[str], delay: float = 1.0) -> List[Dict]:
        """Get information for multiple families with rate limiting"""
        results = []
        
        for family_id in family_ids:
            family_info = self.get_family_details(family_id)
            if family_info:
                results.append(family_info)
            
            # Rate limiting
            time.sleep(delay)
        
        return results

# Example usage
if __name__ == "__main__":
    web_client = TreeFamWeb()
    
    # Search for BRCA families
    print("Searching for BRCA families...")
    brca_families = web_client.search_families("BRCA")
    
    for family in brca_families[:3]:  # Limit to first 3
        print(f"\nFamily: {family['family_id']}")
        print(f"Description: {family['description']}")
        
        # Get detailed information
        details = web_client.get_family_details(family['family_id'])
        if details:
            print(f"Size: {details['size']} genes")
            print(f"First 3 genes: {details['genes'][:3]}")
            
        # Get phylogenetic tree
        tree = web_client.get_phylogenetic_tree(family['family_id'])
        if tree:
            print(f"Tree (first 100 chars): {tree[:100]}...")
        
        time.sleep(1)  # Rate limiting