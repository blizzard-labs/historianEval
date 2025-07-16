#!/usr/bin/env python3

import os
import sys
import re
import pandas as pd
import subprocess
import random
import time

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
           
    def generate_random_sequence(self, frequencies_filename, length, output_file):
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

    def geometric_pmf(self, k, p):
        """Calculate the probability mass function of a geometric distribution."""
        return (1 - p) ** (k - 1) * p
    
    def generate_idl_strings(self, insertion_rate, mean_insertion_length,
                            deletion_rate, mean_deletion_length,
                            max_length=20, precision=9):
        p_ins = 1.0 / mean_insertion_length if mean_insertion_length > 0 else 0
        p_del = 1.0 / mean_deletion_length if mean_deletion_length > 0 else 0
        
        insertion_probs = []
        deletion_probs = []
        
        for length in range(1, max_length + 1):
            ins_prob = self.geometric_pmf(length, p_ins)
            del_prob = self.geometric_pmf(length, p_del)
            
            #Scaling by overall rates
            ins_prob *= insertion_rate
            del_prob *= deletion_rate
            
            insertion_probs.append(str(round(ins_prob, precision)))
            deletion_probs.append(str(round(del_prob, precision)))

        return ','.join(insertion_probs), ','.join(deletion_probs)

    
    def prep_guide_tree(self, tree_file, seq_num):
        target_folder = os.path.join(self.output_folder, 'seq_' + str(seq_num))
        n_name = tree_file.replace('.nwk', '.tree')
        
        os.rename(os.path.join(target_folder, tree_file),
                  os.path.join(target_folder, n_name))
        
        #Modifying structure of the tree file to be compatible with indel-seq-gen
        with open(os.path.join(target_folder, n_name), 'r') as f:
            content = f.read()
        
        mcontent = re.sub(r'\)\d+:', r'):', content)
        #TODO: Add sequence length, indel dist to the tree file
        
        #seq_length = self.params['alignment_length'].iloc[seq_num - 1]
        seq_length = random.randint(100, 250) #! Random sequence length for now... (stack overflows for large sequences)
        
        insert_rate = self.params['insertion_rate'].iloc[seq_num - 1]
        delete_rate = self.params['deletion_rate'].iloc[seq_num - 1]
        mean_insert_length = self.params['mean_insertion_length'].iloc[seq_num - 1]
        mean_delete_length = self.params['mean_deletion_length'].iloc[seq_num - 1]
        
        max_gap = 20 #! Maximum gap length is set to 20, must be changed in the future
        ins_idl, del_idl = self.generate_idl_strings(insert_rate, mean_insert_length, 
                                                    delete_rate, mean_delete_length,
                                                    max_length=max_gap) 
        
        #Write the IDL strings to files
        ins_idl_f = os.path.join(target_folder, n_name.replace('.tree', '_ins_idl')) + '.txt'
        del_idl_f = os.path.join(target_folder, n_name.replace('.tree', '_del_idl')) + '.txt'
        
        with open(ins_idl_f, 'w') as f:
            f.write(ins_idl + '\n')
        with open(del_idl_f, 'w') as f:
            f.write(del_idl + '\n')
        
        with open(os.path.join(target_folder, n_name), 'w') as f:
            f.write('[' + str(seq_length) + ']')
            f.write('{' + str(max_gap) + ',' + str(insert_rate) + '/' + str(delete_rate) + 
                    ',' + str(n_name.replace('.tree', '_ins_idl.txt')) + '}') #+ '/' +str(n_name.replace('.tree', '_del_idl.txt')) + '}')
            f.write(mcontent + '\n')
        
        return os.path.join(target_folder, n_name), n_name

    
    def runIndelSeqGen(self):
        #Clean up output folders and extensions
        for idx, f in enumerate(os.listdir(self.output_folder)):
            folder = 'seq_' + str(idx + 1)
            if os.path.isdir(os.path.join(self.output_folder, folder)):
                tree_file = os.listdir(os.path.join(self.output_folder, folder))[0]
                tree_path = os.path.join(self.output_folder, folder, tree_file)
                
                tree_path, tree_file = self.prep_guide_tree(tree_file, idx + 1)
                ''' #? Useless code, but useful for future reference
                self.generate_random_sequence('data/custom_gtr/GTR_equilibriums.tsv',
                                            int(self.params['n_sequences'].iloc[seq_num - 1]),
                                            os.path.join(target_folder, 'rootseq.root'))
                '''
                
                invariant_rate = self.params['prop_invariant'].iloc[idx] if self.params['prop_invariant'].iloc[idx] > 0 else 0.0
                
                cmd = [
                    "../../../../tools/indel-seq-gen",
                    "--matrix", "JTT",
                    "--outfile", 'sim', #Prefix to all output files
                    "--alpha", str(self.params['gamma_shape'].iloc[idx]),
                    "--invar", str(invariant_rate),
                    "--outfile_format", "f"
                ]
                
                with open(tree_path, 'rb') as guide_f:
                    try:
                        subprocess.run(cmd, cwd= tree_path.replace(tree_file, ''),stdin=guide_f, check=True)
                        print(f'Indel-seq-gen ran successfully on {tree_path}')
                    except subprocess.CalledProcessError as e:
                        print(f'Error running indel-seq-gen on {tree_path}: {e}')
        
    
    def runSoftwareSequence(self, sequence_folder, iterations=1000):
        if os.path.exists(os.path.join(sequence_folder, 'sim.seq')):
            os.rename(os.path.join(sequence_folder, 'sim.seq'),
                    os.path.join(sequence_folder, 'raw_seq.fasta'))
        
        os.makedirs(os.path.join(sequence_folder, 'historian'), exist_ok=True)
        os.makedirs(os.path.join(sequence_folder, 'baliphy'), exist_ok=True)
        
        #track wall-clock time, mcmc mixing
        historian_cmd = [
            "./tools/historian",
            "reconstruct",
            "-seqs", os.path.join(sequence_folder, 'raw_seq.fasta'),
            "-v3",
            "-mcmc",
            #"-samples", str(iterations)
        ] #Redirect stdout and stderr to a log file
        
        baliphy_cmd = [
            "bali-phy",
            os.path.join(sequence_folder, 'raw_seq.fasta'),
            "-A", "Amino-Acids",
            "-i", str(iterations),
            "-n", os.path.join(sequence_folder, 'baliphy/results'),
        ]
        
        '''
        start = time.time()
        try:
            print('Running historian...')
            log_path = os.path.join(sequence_folder, 'historian', 'historian.log')
            with open(log_path, 'w') as log_f:
                #subprocess.run(historian_cmd, check=True)
                subprocess.run(historian_cmd, stdout=log_f, stderr=log_f, check=True)
            print(f'Historian ran successfully on {sequence_folder}')
        except subprocess.CalledProcessError as e:
            print(f'Error running historian on {sequence_folder}: {e}')
        elapsed_h = start - time.time()
        '''
        
        start = time.time()
        try:
            print('Running baliphy...')
            baliphy_log_path = os.path.join(sequence_folder, 'baliphy', 'baliphy.log')
            with open(baliphy_log_path, 'w') as baliphy_log_f:
                subprocess.run(baliphy_cmd, check=True)
                #subprocess.run(baliphy_cmd, stdout=baliphy_log_f, stderr=baliphy_log_f, check=True)
            print(f'Baliphy ran successfully on {sequence_folder}')
        except subprocess.CalledProcessError as e:
            print(f'Error running baliphy on {sequence_folder}: {e}')
        elapsed_b = start - time.time()
        
        return elapsed_h, elapsed_b
    
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
    
    results_h = {}
    results_b = {}
    
    es = evolSimulator(parameters, consensus, label)

    #es.generate_treetop()
    #es.runIndelSeqGen()  # Example for sequence number 1, can be looped for all sequences
    results_h['wall_clock_time'], results_b['wall_clock_time'] = es.runSoftwareSequence(os.path.join(es.output_folder, 'seq_2')) 
    

if __name__ == '__main__':
    main()