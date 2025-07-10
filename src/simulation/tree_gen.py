import dendropy
from dendropy.simulate import treesim
from Bio import Phylo
import random
import subprocess
import pandas as pd
from enum import Enum
import math

class BirthDeathModel(Enum):
    """Birth-death model types"""
    BCSTDCST = "constant_constant"      # Birth Constant, Death Constant
    BEXPDCST = "exponential_constant"   # Birth Exponential, Death Constant
    BLINDCST = "linear_constant"        # Birth Linear, Death Constant
    BCSTDEXP = "constant_exponential"   # Birth Constant, Death Exponential
    BEXPDEXP = "exponential_exponential" # Birth Exponential, Death Exponential
    BLINDEXP = "linear_exponential"     # Birth Linear, Death Exponential
    BCSTDLIN = "constant_linear"        # Birth Constant, Death Linear
    BEXPDLIN = "exponential_linear"     # Birth Exponential, Death Linear
    BLINDLIN = "linear_linear"          # Birth Linear, Death Linear

class CoalescentModel(Enum):
    """Coalescent model types"""
    COALCST = "constant"        # Constant effective population size
    COALEXP = "exponential"     # Exponential population growth
    COALLIN = "linear"          # Linear population growth
    COALSTEP = "step"           # Step function population change
    COALLOG = "logistic"        # Logistic population growth

class HybridTreeGenerator:
    def __init__(self, sampled_file_csv):
        self.parameters = pd.read_csv(sampled_file_csv)
        self.dict = self.parameters.iloc[0].to_dict()
        #Dictionary contains: Filename, n_sequences_tips, alignment_length, gamma_shape, prop_invariant, insertion_rate, deletion_rate, mean_insertion_length, mean_deletion_length, best_BD_speciation_rate, best_BD_extinction_rate, best_coal_growth_rate, best_coal_eff_pop_size, bd_weight, best bd model [one-hot encoded], best coal model [one-hot encoded], tree_length, crown_age
        
        self.bd_models = ['BCSTDCST', 'BEXPDCST', 'BLINDCST', 'BCSTDEXP', 'BEXPDEXP', 'BLINDEXP',
                          'BCSTDLIN', 'BEXPDLIN', 'BLINDLIN']
        self.coal_models = ['COALCST', 'COALEXP', 'COALLIN', 'COALSTEP', 'COALLOG']
        
        
        for key in self.dict.keys():
            if ('best_' + key in self.bd_models and self.dict['best_' + key] == 1):
                self.best_bd_model = key
            elif ('best_' + key in self.coal_models and self.dict['best_' + key] == 1):
                self.best_coal_model = key
                  
        self.boundary_weight = self.dict['bd_weight']
        
    
    def birth_rate_function(self, t: float, model: BirthDeathModel, params: Dict) -> float:
        """Calculate birth rate at time t for different BD models"""
        if model in [BirthDeathModel.BCSTDCST, BirthDeathModel.BCSTDEXP, BirthDeathModel.BCSTDLIN]:
            # Constant birth rate
            return params.get('lambda0', 0.5)
        
        elif model in [BirthDeathModel.BEXPDCST, BirthDeathModel.BEXPDEXP, BirthDeathModel.BEXPDLIN]:
            # Exponential birth rate: lambda(t) = lambda0 * exp(alpha * t)
            lambda0 = params.get('lambda0', 0.5)
            alpha = params.get('alpha', 0.1)
            return lambda0 * math.exp(alpha * t)
        
        elif model in [BirthDeathModel.BLINDCST, BirthDeathModel.BLINDEXP, BirthDeathModel.BLINDLIN]:
            # Linear birth rate: lambda(t) = lambda0 + alpha * t
            lambda0 = params.get('lambda0', 0.5)
            alpha = params.get('alpha', 0.1)
            return max(0, lambda0 + alpha * t)
        
        return 0.5  # Default
    
    def death_rate_function(self, t: float, model: BirthDeathModel, params: Dict) -> float:
        """Calculate death rate at time t for different BD models"""
        if model in [BirthDeathModel.BCSTDCST, BirthDeathModel.BEXPDCST, BirthDeathModel.BLINDCST]:
            # Constant death rate
            return params.get('mu0', 0.1)
        
        elif model in [BirthDeathModel.BCSTDEXP, BirthDeathModel.BEXPDEXP, BirthDeathModel.BLINDEXP]:
            # Exponential death rate: mu(t) = mu0 * exp(beta * t)
            mu0 = params.get('mu0', 0.1)
            beta = params.get('beta', 0.05)
            return mu0 * math.exp(beta * t)
        
        elif model in [BirthDeathModel.BCSTDLIN, BirthDeathModel.BEXPDLIN, BirthDeathModel.BLINDLIN]:
            # Linear death rate: mu(t) = mu0 + beta * t
            mu0 = params.get('mu0', 0.1)
            beta = params.get('beta', 0.05)
            return max(0, mu0 + beta * t)
        
        return 0.1  # Default
    
    def population_size_function(self, t: float, model: CoalescentModel, params: Dict) -> float:
        """Calculate effective population size at time t for different coalescent models"""
        if model == CoalescentModel.COALCST:
            # Constant population size
            return params.get('N0', 1.0)
        
        elif model == CoalescentModel.COALEXP:
            # Exponential population growth: N(t) = N0 * exp(r * t)
            N0 = params.get('N0', 1.0)
            r = params.get('r', 0.1)
            return N0 * math.exp(r * t)
        
        elif model == CoalescentModel.COALLIN:
            # Linear population growth: N(t) = N0 + r * t
            N0 = params.get('N0', 1.0)
            r = params.get('r', 0.1)
            return max(0.01, N0 + r * t)  # Prevent zero population
        
        elif model == CoalescentModel.COALSTEP:
            # Step function population change
            N0 = params.get('N0', 1.0)
            N1 = params.get('N1', 2.0)
            t_change = params.get('t_change', 0.5)
            return N1 if t > t_change else N0
        
        elif model == CoalescentModel.COALLOG:
            # Logistic population growth: N(t) = K / (1 + ((K-N0)/N0) * exp(-r*t))
            N0 = params.get('N0', 1.0)
            K = params.get('K', 10.0)  # Carrying capacity
            r = params.get('r', 0.1)
            if K == N0:
                return N0
            return K / (1 + ((K - N0) / N0) * math.exp(-r * t))
        
        return 1.0  # Default
        
    def generate_bd_subtree(self, ntaxa: int, birth_rate: float, death_rate: float, time_limit: float = None):
        pass
    
    def generate_coal_subtree(self, ntaxa: int, growth_rate: float, eff_pop_size: float, time_limit: float = None):
        pass
        
            
'''

def generate_trees(clade):
    num_topologies = default.num_topologies
    config_data = default.model_classes[clade]
    
    for i in range(num_topologies):
        ntax = random.randint(config_data['ntax'][0], config_data['ntax'][1])
        if (clade == 'vertebrate'):
            tree = treesim.birth_death_tree(birth_rate=config_data['topology_params']['birth_rate'], 
                                            death_rate= config_data['topology_params']['death_rate'],
                                            num_extant_tips=ntax, repeat_until_success=True)
            
            fp = "data/" + clade + "/treeTop_" + str(i) + ".tree"
            tree.write_to_path(fp, schema = "newick")
            
            clean_tree(fp, config_data['sequence_length'])   


def clean_tree(tree_path, seq_length = 999):
    with open(tree_path, 'r') as f:
        tree_str = f.read().strip()
    
    tree_str = tree_str.replace('[&R]', '['+str(seq_length)+']')
    
    i = len(tree_str) - 1
    while (tree_str[i] != ":"):
        tree_str = tree_str[0:i]
        i-=1
        
    tree_str = tree_str[0:len(tree_str) - 1] + ";"
    
    with open(tree_path, 'w') as f:
        f.write(tree_str)


def simulate_evol(clade):
    config_data = default.model_classes[clade]
    
    cmd = [
        "./tools/indel-seq-gen",
        "--matrix", config_data['substitution_model'],
        "-e DNA_out", "< data/vertebrate/treeTop_0.tree"
    ]
    
    print(cmd)
    subprocess.run(cmd)


#./indel-seq-gen --matrix HKY --outfile DNA_out < simple_nuc.tree

generate_trees('vertebrate')
#simulate_evol('vertebrate')
'''

