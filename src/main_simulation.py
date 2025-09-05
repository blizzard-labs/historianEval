#!/usr/bin/env python3

import os
import sys
import re
import pandas as pd
import subprocess
import random
import time
from Bio import SeqIO
import concurrent.futures

class evolSimulator:
    def __init__(self, parameters_file, consensus_tree_file="none", tag='none'):
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
    
    def process_seq(self, seq_folder):
        seq_results = {}
        seq_results['path'] = seq_folder
        seq_results['elapsed_historian'], seq_results['elapsed_baliphy'] = self.runSoftwareSequence(seq_folder)
        return seq_results

    def generate_treetop_with_params(self, max_iterations=1000):
        for idx, row in self.params.iterrows():
            seq_folder = os.path.join(self.output_folder, 'seq_' + str(idx + 1))
            os.mkdir(seq_folder)
            
            best_model = None
            for col in self.params.columns:
                if (col.startswith('best_B') and not col.startswith('best_BD')) and row[col] == 1:
                    best_model = col.replace('best_', '')
                    break
            
            try:
                cmd = [
                    "python",
                    "src/simulation/tree_gen_bd.py",
                    "--birth_rate", str(row['best_BD_speciation_rate']),
                    "--death_rate", str(row['best_BD_extinction_rate']),
                    "--bd_model", best_model,
                    "--birth_alpha", str(row['best_BD_speciation_alpha']),
                    "--death_alpha", str(row['best_BD_extinction_alpha']),
                    "--target_colless", str(row['normalized_colless_index']),
                    "--target_gamma", str(row['gamma']),
                    "--num_taxa", str(row['n_sequences_tips']),
                    "--crown_age", str(row['crown_age']),
                    "--max_iterations", str(max_iterations),
                    "--output", os.path.join(seq_folder, 'guide.tree')
                ]
                
                subprocess.run(cmd, check=True)
                print(f'Guide tree generated for sequence {idx+1}!')
            except subprocess.CalledProcessError as e:
                print(f'Error generating topology for sequence {idx+1}: {e}')
        print('Generated all guide tree topologies!')
        
    def generate_treetop_with_distance(self):
        try:
            cmd = [
                "python", 
                "src/simulation/tree_gen_spr.py",
                self.consensus_tree_file,
                self.output_folder,
                ",".join(str(int(size)) for size in self.params['n_sequences'].tolist()),
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
                            max_length=25, precision=10, format='indelible'):
        p_ins = 1.0 / mean_insertion_length if mean_insertion_length > 0 else 0
        p_del = 1.0 / mean_deletion_length if mean_deletion_length > 0 else 0
        
        insertion_probs = []
        deletion_probs = []
        #! USE numpy geometric library
        for length in range(1, max_length + 1):
            ins_prob = self.geometric_pmf(length, p_ins)
            del_prob = self.geometric_pmf(length, p_del)
            
            #Scaling by overall rates
            ins_prob *= insertion_rate
            del_prob *= deletion_rate
            
            insertion_probs.append(str(round(ins_prob, precision)))
            deletion_probs.append(str(round(del_prob, precision)))

        if format == 'indelible':
            return '\n'.join(insertion_probs), '\n'.join(deletion_probs)
        else:
            return ','.join(insertion_probs), ','.join(deletion_probs)

    
    def prep_guide_tree(self, tree_file, seq_num, format="indelible"):
        target_folder = os.path.join(self.output_folder, 'seq_' + str(seq_num))
        n_name = "control.txt" if format == "indelible" else tree_file.replace('.nwk', '.txt').replace('.tree', '.txt')
        
        os.rename(os.path.join(target_folder, tree_file),
                  os.path.join(target_folder, n_name))
        
        with open(os.path.join(target_folder, n_name), 'r') as f:
            content = f.read()
        
        #Modifying structure of the tree file to be compatible with indel-seq-gen
        mcontent = re.sub(r'\)\d+:', r'):', content)
        
        seq_length = self.params['alignment_length'].iloc[seq_num - 1]
        
        insert_rate = self.params['insertion_rate'].iloc[seq_num - 1]
        delete_rate = self.params['deletion_rate'].iloc[seq_num - 1]
        mean_insert_length = self.params['mean_insertion_length'].iloc[seq_num - 1]
        mean_delete_length = self.params['mean_deletion_length'].iloc[seq_num - 1]
        
        max_gap = 20 #! Maximum gap length is set to 20, can be changed in the future
        ins_idl, del_idl = self.generate_idl_strings(insert_rate, mean_insert_length, 
                                                    delete_rate, mean_delete_length,
                                                    max_length=max_gap, format=format) 
        
        ins_q_val = 1 / (mean_insert_length)
        del_q_val = 1 / (mean_delete_length)
        
        if format == "indelible":
            with open(os.path.join(target_folder, n_name), 'w') as f:
                f.write(f'[TYPE] AMINOACID 1\n\n[MODEL] modelname\n')
                f.write(f'   [submodel] LG\n')
                f.write(f"   [rates] {self.params['prop_invariant'].iloc[seq_num - 1]} {self.params['gamma_shape'].iloc[seq_num - 1]} 0\n")
                f.write(f'   [insertmodel] NB {ins_q_val} 1\n') #Negative binomial distribution simplifies to a geometric distribution
                f.write(f'   [deletemodel] NB {del_q_val} 1\n')
                f.write(f'   [insertrate] {insert_rate}\n')
                f.write(f'   [deleterate] {delete_rate}\n\n')
                f.write('[TREE] treename '+ mcontent+'\n')
                f.write(f'[PARTITIONS] partitionname\n')
                f.write(f'  [treename modelname {seq_length}]\n\n')
                f.write(f'[EVOLVE] partitionname 1 simulated  ')
                
        else:
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
                
                tree_path, tree_file = self.prep_guide_tree(tree_file, idx + 1, "isg")
                ''' #? Useless code
                self.generate_random_sequence('data/custom_gtr/GTR_equilibriums.tsv',
                                            int(self.params['n_sequences'].iloc[seq_num - 1]),
                                            os.path.join(target_folder, 'rootseq.root'))
                '''
                
                invariant_rate = self.params['prop_invariant'].iloc[idx] if self.params['prop_invariant'].iloc[idx] > 0 else 0.0
                
                cmd = [
                    "../../../../tools/indel-seq-gen",
                    "--matrix", "LG",
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
        
    def runIndelible(self):
        for idx, f in enumerate(os.listdir(self.output_folder)):
            folder = 'seq_' + str(idx + 1)
            if os.path.isdir(os.path.join(self.output_folder, folder)):
                tree_file = os.listdir(os.path.join(self.output_folder, folder))[0]
                tree_path = os.path.join(self.output_folder, folder, tree_file)

                tree_path, tree_file = self.prep_guide_tree(tree_file, idx+1)
                cmd = [
                    "../../../../tools/indelible"
                ]
                
                try: 
                    subprocess.run(cmd, cwd= tree_path.replace(tree_file, ''), check=True)
                    print(f'Indelible ran successfully on {tree_path}')
                except subprocess.CalledProcessError as e:
                    print(f'Error running indelible on {tree_path}: {e}')
        
        print(f'Completed running indelible on all sequences')
        
    def cleanupSimFolders(self):
        for seq in os.listdir(self.output_folder):
            seq_folder = os.path.join(self.output_folder, seq)
            if os.path.isdir(seq_folder):
                os.makedirs(os.path.join(seq_folder, 'control_files'), exist_ok=True)
                
                # Move control files
                os.rename(os.path.join(seq_folder, 'control.txt'), os.path.join(seq_folder, 'control.txt').replace('control.txt', 'control_files/control.txt'))
                os.rename(os.path.join(seq_folder, 'LOG.txt'), os.path.join(seq_folder, 'LOG.txt').replace('LOG.txt', 'control_files/LOG.txt'))
                
                # Rename the raw sequence file
                os.rename(os.path.join(seq_folder, 'simulated.fas'), os.path.join(seq_folder, 'sequences.fasta'))
                
                # Convert the simulated TRUE file to fasta format
                records = SeqIO.parse(os.path.join(seq_folder, 'simulated_TRUE.phy'), 'phylip-relaxed')
                SeqIO.write(records, os.path.join(seq_folder, 'alignment.fasta'), 'fasta')
                os.rename(os.path.join(seq_folder, 'simulated_TRUE.phy'), os.path.join(seq_folder, 'simulated_TRUE.phy').replace('simulated_TRUE.phy', 'control_files/simulated_TRUE.phy'))
                
                # Extract the guide tree and save as a newick file
                with open(os.path.join(seq_folder, 'trees.txt'), 'r') as f:
                    tree_string = f.read().strip().split('\n')[-1].split()[-1].strip()
                
                with open(os.path.join(seq_folder, 'guide.tree'), 'w') as f:
                    f.write(tree_string + '\n')
                
                # Move the guide tree to control files
                os.rename(os.path.join(seq_folder, 'trees.txt'), os.path.join(seq_folder, 'trees.txt').replace('trees.txt', 'control_files/trees.txt'))
                
    
    def configure_historian_control_file(self, sequence_folder, matrix_path="data/matrices/lg.json"):
        indelible_control_path = os.path.join(sequence_folder, 'control_files/control.txt')
        with open(indelible_control_path, 'r') as f:
            contents = f.read()
        
        key_params = {}
        
        for line in contents.strip().split('\n'):
            parts = line.strip().split()
            
            if parts:
                if parts[0] == '[rates]' and len(parts) == 4:
                    key_params['prop_invariant'] = float(parts[1])
                    key_params['gamma_shape'] = float(parts[2])
                
                if parts[0] == '[insertrate]' and len(parts) == 2:
                    key_params['insertrate'] = float(parts[1])
                if parts[0] == '[deleterate]' and len(parts) == 2:
                    key_params['deleterate'] = float(parts[1])

                if parts[0] == '[insertmodel]' and len(parts) == 4:
                    key_params['insertext'] = 1 - float(parts[2])
                if parts[0] == '[deletemodel]' and len(parts) == 4:
                    key_params['deleteext'] = 1 - float(parts[2])
        
        with open(matrix_path, 'r') as f:
            matrix_contents = f.read()
        
        def tr(value):
            return 1/(1 - value)
        
        #Calculate weighted average for indel extension
        key_params['avg_length'] = (tr(key_params["insertext"]) * key_params['insertrate'] + tr(key_params["deleteext"]) * key_params['deleterate']) / (key_params["insertrate"] + key_params["deleterate"])
        
        key_params["indelrate"] = (key_params["insertrate"] + key_params["deleterate"]) / 2
        key_params["indelext"] = 1 - 1/key_params['avg_length']
                
        with open(os.path.join(sequence_folder, 'historian', 'lg.json'), 'w') as f:
            for line in matrix_contents.split('\n'):
                parse_line = line.strip().split()
                if parse_line:
                    if "insrate" in parse_line[0] and len(parse_line) == 2:
                        f.write(f'  "insrate": {key_params["indelrate"]},\n')
                    elif "delrate" in parse_line[0] and len(parse_line) == 2:
                        f.write(f'  "delrate": {key_params["indelrate"]},\n')
                    elif "insextprob" in parse_line[0] and len(parse_line) == 2:
                        f.write(f'  "insextprob": {key_params["indelext"]},\n')
                    elif "delextprob" in parse_line[0] and len(parse_line) == 2:
                        f.write(f'  "delextprob": {key_params["indelext"]},\n')
                    else:
                        f.write(line + '\n')
        
        return key_params, os.path.join(sequence_folder, 'historian', 'lg.json')
        
    def runSoftwareSequence(self, sequence_folder, iter_cap_per_seq=100000, timeout_hours=2, conda_env="phylo"):
        #Fixed number of iterations
        #Running Measure: total wall-clock, avg. time/iteration, convergence/iteration
        #Final Measure: SP, TC, RF, RFL, etc.
        
        #Fixed in sampling: LG08 Parameters,
        
        os.makedirs(os.path.join(sequence_folder, 'historian'), exist_ok=True)
        os.makedirs(os.path.join(sequence_folder, 'baliphy'), exist_ok=True)

        n_seqs = sum(1 for _ in SeqIO.parse(os.path.join(sequence_folder, 'sequences.fasta'), 'fasta'))
        key_params, matrix_path = self.configure_historian_control_file(sequence_folder)

        #./tools/historian reconstruct -seqs tools/testArena/seq_1/sequences.fasta -mcmc -model data/matrices/lg.json -gamma 5 -shape 1.8572344303576052 -samples 15 -trace tools/testArena/seq_1/historian/trace.log -v5
        #Indelible uses 5 gamma categories by default, so we set it to 5 here
        historian_cmd = [
            "./tools/historian",
            "reconstruct",
            "-seqs", os.path.join(sequence_folder, 'sequences.fasta'),
            "-mcmc",
            "-model", matrix_path,
            "-gamma", str(5),
            "-shape", str(key_params['gamma_shape']),
            "-samples", str(iter_cap_per_seq),
            "-trace", os.path.join(sequence_folder, 'historian/trace.log'),
            "-v5"
        ]
        
        #python historian_parser.py trace5.log parsed_trace.log --trees
        historian_parser_cmd = [
            "python",
            "src/simulation/trace_parser.py",
            os.path.join(sequence_folder, 'historian/trace.log'),
            os.path.join(sequence_folder, 'historian/parsed_trace.log'),
            "--trees",
            "--sequences"
        ]
                
        # bali-phy tools/testArena/seq_3/sequences.fasta -A Amino-Acids -n tools/testArena/seq_3/baliphy -S 'lg08 +> Rates.gamma(5,alpha=1.515285985794507) ' -I "rs07(rate=0.22083670052,mean_length=3.54825557248)"
        #Currently not including invariant sites to maintain consistency with historian
        baliphy_cmd = [
            #"conda run",
            #"-n", conda_env,
            "bali-phy",
            os.path.join(sequence_folder, "sequences.fasta"),
            "-A", "Amino-Acids",
            "-i", str(2 * iter_cap_per_seq * n_seqs),
            "-n", os.path.join(sequence_folder, "baliphy"),
            "-S", "lg08 +> f(lg08_freq) +> Rates.gamma(5, alpha=" + str(key_params["gamma_shape"]) + ")",
            '-I', 'rs07(rate=' + str(key_params['indelrate'] * 2) + ', mean_length=' + str(key_params['avg_length']) + ')'
        ]

        print('|\t [COMMAND]' + ' '.join(historian_cmd))
        print('|\t [COMMAND]'+ ' '.join(baliphy_cmd))
        
        def run_historian():
            start = time.time()
            log_path = os.path.join(sequence_folder, 'historian', 'historian.log')
            try:
                with open(log_path, 'w') as log_f:
                    subprocess.run(historian_cmd, stdout=log_f, stderr=log_f, check=True, timeout=timeout_hours*60*60)
                # Rename trace file if needed
                trace1 = os.path.join(sequence_folder, 'historian', 'trace.log.1')
                trace = os.path.join(sequence_folder, 'historian', 'trace.log')
                if os.path.exists(trace1):
                    os.rename(trace1, trace)
                print(f'Historian ran successfully on {sequence_folder}')
            except subprocess.TimeoutExpired:
                print(f'Historian timed out after {timeout_hours} hours on {sequence_folder}')
            except subprocess.CalledProcessError as e:
                print(f'Error running historian on {sequence_folder}: {e}')
            elapsed = time.time() - start
            # Parse historian trace
            try:
                subprocess.run(historian_parser_cmd, check=True)
                print(f'Historian trace parsed successfully on {sequence_folder}')
            except subprocess.CalledProcessError as e:
                print(f'Error parsing historian trace on {sequence_folder}: {e}')
            return elapsed

        def run_baliphy():
            start = time.time()
            log_path = os.path.join(sequence_folder, 'baliphy', 'baliphy.log')
            try:
                with open(log_path, 'w') as log_f:
                    subprocess.run(baliphy_cmd, stdout=log_f, stderr=log_f, check=True, timeout=timeout_hours*60*60)
                print(f'Baliphy ran successfully on {sequence_folder}')
            except subprocess.TimeoutExpired:
                print(f'Baliphy timed out after {timeout_hours} hours on {sequence_folder}')
            except subprocess.CalledProcessError as e:
                print(f'Error running Baliphy on {sequence_folder}: {e}')
            elapsed = time.time() - start
            return elapsed

        # Run historian and baliphy in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_h = executor.submit(run_historian)
            future_b = executor.submit(run_baliphy)
            elapsed_h = future_h.result()
            elapsed_b = future_b.result()
            #elapsed_b = 0 # For testing without bali-phy 

        return elapsed_h, elapsed_b


    def runBenchmark(self, sequence_folders=[]):
        if len(sequence_folders) == 0:
            for folder in os.listdir('data/simulation'):
                if folder.endswith('e2'): #! Restricting to Experiment 2
                    for seq_folder in os.listdir(os.path.join('data/simulation', folder)):
                        if os.path.isdir(os.path.join('data/simulation', folder, seq_folder)):
                            sequence_folders.append(os.path.join('data/simulation', folder, seq_folder))

        results = []

        def process_seq(seq_folder):
            seq_results = {}
            seq_results['path'] = seq_folder
            seq_results['elapsed_historian'], seq_results['elapsed_baliphy'] = self.runSoftwareSequence(seq_folder)
            return seq_results

        
        # Process two sequences at a time, with progress monitoring
        print(f"Starting benchmark for {len(sequence_folders)} sequences...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
            future_to_seq = {executor.submit(self.process_seq, seq_folder): seq_folder for seq_folder in sequence_folders}
            completed = 0
            for future in concurrent.futures.as_completed(future_to_seq):
                seq_folder = future_to_seq[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    print(f"[Progress] Completed {completed}/{len(sequence_folders)}: {seq_folder}")
                except Exception as exc:
                    print(f"[Error] Sequence {seq_folder} generated an exception: {exc}")

        with open(os.path.join(self.output_folder, 'benchmark_results.csv'), 'w') as f:
            f.write('path,elapsed_historian,elapsed_baliphy\n')
            for res in results:
                f.write(f"{res['path']},{res['elapsed_historian']},{res['elapsed_baliphy']}\n")

        return results
                
            

def main():
    print('Begun script...')
    if len(sys.argv) < 3:
        print('Usage: python src/simulation/main.py <parameters_file> <tag> [consensus_tree]')
        print("Example: python src/main_simulation.py data/model_gen/SCOPtype1/experiment1_parameters.csv SCOPt1e1")
    
    
    #Pipeline: generate guide trees --> run indel-seq-gen --> organize output files --> run historian/baliphy on raw sequences --> evaluate results
    
    parameters = sys.argv[1]
    label = sys.argv[2]
    consensus = sys.argv[3] if len(sys.argv) > 3 else 'none'
    
    es = evolSimulator(parameters, tag=label, consensus_tree_file=consensus)
    
    start = time.time()
    #*PFAM SOCP TYPES PIPELINE
    
    '''
    es.generate_treetop_with_params(max_iterations=1000)
    print(f'Generated tree topologies- ELAPSED TIME: {time.time() - start}============================')
    es.runIndelible()
    print(f'Ran Indelible- ELAPSED TIME: {time.time() - start}============================')
    
    es.cleanupSimFolders()
    print(f'Cleaned up simulation folders- ELAPSED TIME: {time.time() - start}============================')
    '''
    
    es.runBenchmark()
    print(f'Ran benchmark- ELAPSED TIME: {time.time() - start}============================')
    
    print('COMPLEETTEEEETETETETE!!!!')

if __name__ == '__main__':
    main()