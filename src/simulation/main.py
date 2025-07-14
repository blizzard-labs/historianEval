#!/usr/bin/env python3

import os
import sys
import re
import pandas as pd
import subprocess
import random

class evolSimulator:
    def __init__(self, parameters_file, consensus_tree_file, tag='none'):
        self.parameters_file = parameters_file
        self.consensus_tree_file = consensus_tree_file
        
        # Load and cleanup csv
        self.params = pd.read_csv(self.parameters_file)
        if 'sequence_name' not in self.params.columns:
            self.params.insert(loc=0, column='sequence_name', value=['seq_' + str(k) for k in (self.params.index + 1)])
            self.params.to_csv(self.parameters_file, index=False)
        
        self.size = len(self.params.index)
        
        # Generate output folder
        base_dir = 'data/simulation'
        if (tag.strip().lower() != 'none'):
            self.output_folder = os.path.join(base_dir, tag)
        else:
            contents = os.listdir(base_dir)
            c= 0
            for folder in contents:
                print(folder)
                if os.path.isdir(os.path.join(base_dir, folder)) and 'experiment_' in folder: 
                    c+=1
                    print(c)
            self.output_folder = os.path.join(base_dir, 'experiment_' + str(c+1))
            
        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)
        
        #Extract all parameters from parameters_file into a dictionary and clean.
        #Class variables: parameters dataframe, indel-seq-gen path, historian path, baliphy-path
        pass
    
    def generate_treetop(self):
        try:
            cmd = [
                "python", 
                "src/simulation/tree_gen_nni.py",
                self.consensus_tree_file,
                self.output_folder,
                ",".join(str(rf) for rf in self.params['rf_length_distance'].tolist()),
                "--replicates", str(1),
                "--max-iterations", str(1200),
            ]
            
            subprocess.run(cmd, check=True)
            print('Guide trees generated successfully')
        except subprocess.CalledProcessError as e:
            print(f'Error generating topologies: {e}')
           
    def generate_random_sequence(frequencies_filename, length, output_file):
        frequencies = {}
        
        try:
            with open(frequencies_filename, 'r') as f:
                lines = f.readlines()
                
                # Skip header line
                for line in lines[1:]:
                    line = line.strip()
                    if line:  # Skip empty lines
                        parts = line.split('\t')
                        if len(parts) >= 3:  # index, amino_acid, frequency
                            amino_acid = parts[1]
                            frequency = float(parts[2])
                            frequencies[amino_acid] = frequency
                            
        except FileNotFoundError:
            print(f"Error: File '{frequencies_filename}' not found.")
            sys.exit(1)
        except ValueError as e:
            print(f"Error parsing frequency values: {e}")
            sys.exit(1)
        
        weighted_list = []
        scale_factor = 10000
    
        for amino_acid, frequency in frequencies.items():
            count = int(frequency * scale_factor)
            weighted_list.extend([amino_acid] * count)
        
        seq = ''.join(random.choices(weighted_list, k=length))
        
        with open(output_file, 'w') as f:
            f.write(seq + '\n')
        
        return seq

    
    def runIndelSeqGen(self, seq_num):
        #Clean up output folders and extensions
        for folder in os.listdir(self.output_folder):
            if os.path.isdir(os.path.join(self.output_folder, folder)):
                for tree_file in os.listdir(os.path.join(self.output_folder, folder)):
                    if tree_file.endswith('.nwk'): 
                        n_name = os.path.join(self.output_folder, folder, tree_file.replace('.nwk', '.tree'))
                        os.rename(os.path.join(self.output_folder, folder, tree_file), 
                                n_name)
                        
                        #Modifying structure of the tree file to be compatible with indel-seq-gen
                        with open(n_name, 'r') as f:
                            content = f.read()
                           
                        m_content = content.split(')')
                        del m_content[-1]
                        mcontent = ')'.join(m_content) + ';'
                        
                        mcontent = re.sub(r'\)\d+:', r'):', content)
                        
                        with open(n_name, 'w') as f:
                            f.write(mcontent)
                            f.write('')
                        
        
        target_folder = os.path.join(self.output_folder, 'seq_' + str(seq_num))
        
        insert_rate = self.params['insertion_rate'].iloc[seq_num - 1]
        delete_rate = self.params['deletion_rate'].iloc[seq_num - 1]
        
        mean_insert_length = self.params['mean_insertion_length'].iloc[seq_num - 1]
        mean_delete_length = self.params['mean_deletion_length'].iloc[seq_num - 1]
        
        self.generate_random_sequence(int(self.params['n_sequences'].iloc[seq_num - 1]),
                                      os.path.join(target_folder, 'rootseq.fasta'))
        
        for file in os.listdir(target_folder): 
            try:
                tree_path = os.path.join(target_folder, file)
                
                cmd = [
                    "./tools/indel-seq-gen",
                    "--matrix", "JTT",
                    "--outfile", os.path.join(target_folder, file.replace('.tree', '')),
                    "--alpha", str(self.params['gamma_shape'].iloc[seq_num - 1]),
                    "--invar", str(self.params['prop_invariant'].iloc[seq_num - 1]),
                    "--outfile_format", "f",
                    
                ]
                
                with open(tree_path, 'rb') as tree_f:
                    subprocess.run(cmd, stdin=tree_f, check=True)
                print(f'Indel-seq-gen ran successfully on {file}')
            except subprocess.CalledProcessError as e:
                print(f'Error running indel-seq-gen on {file}: {e}')
        
    
    def runSoftware(self):
        #Run historian/baliphy on raw sequences (save wall clock time)
        pass
    
    def evaluateResults(self):
        #MCMC mixing, rfl distance, align metrics, evolutionary parameters, spfn
        pass
    
def main():
    print('Begun script...')
    if len(sys.argv) < 3:
        print('Usage: python src/simulation/main.py <parameters_file> <consensus_tree> [tag] ')
    
    
    #Pipeline: generate guide trees --> run indel-seq-gen --> organize output files --> run historian/baliphy on raw sequences --> evaluate results
    
    parameters = sys.argv[1]
    consensus = sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else 'none'
    
    es = evolSimulator(parameters, consensus, label)

    #es.generate_treetop()
    es.runIndelSeqGen(1)  # Example for sequence number 1, can be looped for all sequences
    
    

if __name__ == '__main__':
    main()