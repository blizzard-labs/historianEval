#!/usr/bin/env python3

import os
import sys
import pandas as pd
import subprocess

class evolSimulator:
    def __init__(self, parameters_file, consensus_tree_file, tag='none'):
        self.parameters_file = parameters_file
        self.consensus_tree_file = consensus_tree_file
        
        # Load and cleanup csv
        self.params = pd.read_csv(self.parameters_file)
        self.params.insert(loc=0, column='sequence_name', value=['seq_' + str(k) for k in (self.params.index + 1)])
        self.params.to_csv(self.parameters_file, index=False)
        
        self.size = len(self.params.index)
        
        # Generate output folder
        base_dir = 'data/simulation'
        if (tag.strip().lower() != 'none'):
            os.mkdir(os.path.join(base_dir, tag))
        else:
            contents = os.listdir(base_dir)
            c= 0
            for folder in contents:
                if os.path.isdir(folder) and folder.startswith('experiment_'): c+=1
            self.output_folder = os.path.join(base_dir, 'experiment_' + str(c+1))
            
        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)
        
        #Extract all parameters from parameters_file into a dictionary and clean.
        #Class variables: parameters dataframe, indel-seq-gen path, historian path, baliphy-path
        pass
    
    def generate_treetop(self):
        #Write a python script that given a majority consensus tree and a threshold RF distance, 
        #run SPR steps on the consensus tree until it crosses the threshold. The script should do this multiple
        #times, generating multiple variant trees to an output folderwhen provided with a list of threshold RF 
        #distances.
        
        try:
            cmd = [
                "python", 
                "src/simulation/tree_gen.py",
                self.consensus_tree_file,
                self.output_folder,
                ",".join(str(rf) for rf in self.params['rf_length_distance'].tolist()),
                "--replicates", str(5),
            ]
            
            subprocess.run(cmd, check=True)
            print('Guide trees generated successfully')
        except subprocess.CalledProcessError as e:
            print(f'Error generating topologies: {e}')
            
    def runIndelSeqGen(self):
        #Write a script that given a csv of parameters and a corresponding folder of trees, simulates evolution
        #on them with indel-seq-gen
        pass
    
    def runSoftware(self):
        #Run historian/baliphy on raw sequences (save wall clock time)
        pass
    
    def evaluateResults(self):
        #MCMC mixing, rfl distance, align metrics, evolutionary parameters, spfn
        pass
    
def main():
    print('Begun script...')
    model_folder = 'data/model_gen/mamX10k'
    
    es = evolSimulator(os.path.join(model_folder, 'simulated_phylo_parameters.csv'), 
                       os.path.join(model_folder, 'consensus.tree'))

    es.generate_treetop()
if __name__ == '__main__':
    main()