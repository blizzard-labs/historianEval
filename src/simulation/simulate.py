import dendropy
from dendropy.simulate import treesim
from Bio import Phylo
import random
from Historian_eval.config import default
import subprocess


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
