#!/usr/bin/env python3

import os
import sys
import subprocess
import logging
import sys
from pathlib import Path
from Bio import Phylo, AlignIO, SeqIO
import utils.general as utils
import model_gen_aa.clean_table
import pandas as pd
from io import StringIO
import time

class modelConstructor:
    def __init__(self, operate, label, alignment_folder, tree_folder="none", temp_folder="data/model_gen", output_folder="models", params_file="none", log_file="none", log=True):
        self.operate = operate.strip().lower()
        self.label = label
        self.alignment_folder = alignment_folder
        self.tree_folder = tree_folder
        self.temp_folder = temp_folder + "/" + label
        self.output_folder = output_folder
        self.output_file = f"{self.output_folder}/{label}.json"
        self.params_file = params_file
        self.params_wr_file = params_file.replace(".csv", "_with_rates.csv")
        self.params_wc_file = params_file.replace(".csv", "_cleaned.csv")
        
        os.makedirs(self.temp_folder, exist_ok=True)
        if (self.tree_folder == "none"):
            os.makedirs(self.temp_folder + "/trees", exist_ok=True)
            self.tree_folder = self.temp_folder + "/trees"
        
        if log_file == "none":
            self.log_file = os.path.join("data/logs", f"{label}_model_gen.log")
        
        if log:
            #Setting up logging
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s [%(levelname)s] %(message)s',
                handlers=[
                    logging.FileHandler("output.log"),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            sys.stdout = utils.StreamToLogger(logging.getLogger(), logging.INFO)
            sys.stderr = utils.StreamToLogger(logging.getLogger(), logging.ERROR) 
        
    def cleanup_trees(self):
        for file in os.listdir(self.tree_folder):
            if os.path.isfile(os.path.join(self.tree_folder, file)) and (file.endswith(".nhx") or file.endswith(".newick")):
                utils.strip_metadata(os.path.join(self.tree_folder, file))
            if os.path.isfile(os.path.join(self.tree_folder, file)) and file.endswith(".rootree"):
                path_obj = Path(os.path.join(self.tree_folder, file))
                new_file_path = path_obj.with_suffix(".tree")
                path_obj.rename(new_file_path)
    
    def cleanup_pfam_data(self, min_species=3):
        self.cleanup_trees()
        
        for file in os.listdir(self.alignment_folder):
            if os.path.isfile(os.path.join(self.alignment_folder, file)) and file.endswith(".seed"):
                alignment = AlignIO.read(os.path.join(self.alignment_folder, file), "stockholm")
                AlignIO.write(alignment, os.path.join(self.alignment_folder, file).replace(".seed", ".fasta"), "fasta")
                os.remove(os.path.join(self.alignment_folder, file))
        
        for filename in os.listdir(self.alignment_folder):
            if filename.endswith(".fasta"):
                filepath = os.path.join(self.alignment_folder, filename)
                records = list(SeqIO.parse(filepath, "fasta"))
                
                if len(records) < min_species:
                    os.remove(filepath)
                    os.remove(os.path.join(self.tree_folder, filename.replace(".fasta", ".tree")))

    def cleanup_modeltest_trees(self):
        modeltest_folder = os.path.join(self.temp_folder, "temp_modeltest")
        try:
            existant_files = []
            for file in os.listdir(self.tree_folder):
                if os.path.isfile(os.path.join(self.tree_folder, file)):
                    existant_files.append(file.split(".")[0])
            
            for file in os.listdir(modeltest_folder):
                if file.endswith(".tree") and file.split(".")[0] not in existant_files:
                    os.rename(os.path.join(modeltest_folder, file), os.path.join(self.tree_folder, file))
            
            self.cleanup_trees()
            print(f"Trees cleaned and moved to {self.tree_folder}.")
        except Exception as e:
            print(f"Error cleaning up trees: {e}")
            raise
        
    def extract_substitution_params(self, only_indels=False):
        """Extracts substitution parameters from the alignment folder using modeltest-ng."""
        #Example: python protein_extractor.py ./protein_alignments/ ./results/ /usr/local/bin/modeltest-ng false
        
        m_path = "tools/"
        if self.operate == "osx":
            m_path = "tools/modeltest-ng-osx"
        elif self.operate == "x86":
            m_path = "tools/"
        else:
            m_path = "tools/modeltest-ng-static"
                
        cmd = [
            "python",
            "src/model_gen_aa/extract_params.py",
            self.alignment_folder,
            self.temp_folder,
            m_path, 
            str(only_indels).lower()
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Substitution parameters extracted for {self.label}.")
        except subprocess.CalledProcessError as e:
            print(f"Error extracting substitution parameters: {e}")
            raise
        except subprocess.TimeoutExpired as e:
            print(f"Command timed out: {e}")
            raise
        
        '''
        try:
            for file in os.listdir(self.temp_folder):
                if file.endswith(".csv"):
                    self.params_file = os.path.join(self.temp_folder, file)
        except Exception as e:
            print(f"Error finding parameters file in {self.temp_folder}: {e}")
            raise      
        '''

               
    def generate_ml_trees(self, raxml_ng_path="tools/raxml-ng"):
        try:
            for alignment in os.listdir(self.alignment_folder):
                if os.path.isfile(os.path.join(self.alignment_folder, alignment)):
                    cmd = [
                        "./" + raxml_ng_path,
                        "--search1",
                        "--msa", os.path.join(self.alignment_folder, alignment),
                        "--model", "GTR+G",
                        "--threads", "4",
                        "--tree", "rand",
                        "--prefix", os.path.join(self.tree_folder, alignment.split('.')[0]) + "/"
                    ]
                    
                    subprocess.run(cmd, check=True)
                    print(f"ML tree generated for {alignment}.")
                    
        except subprocess.CalledProcessError as e:
            print(f"Error generating ML trees: {e}")
            raise
        except:
            print(f"Error reading alignments from {self.alignment_folder}.")
            raise
        
        try:
            for tree in os.listdir(self.tree_folder):
                if os.path.isfile(os.path.join(self.tree_folder, tree)):
                    if tree.endswith(".bestTree"):
                        new_tree_name = tree.replace(".bestTree", ".tree")
                        os.rename(os.path.join(self.tree_folder, tree), os.path.join(self.tree_folder, new_tree_name))
                    else:
                        os.remove(os.path.join(self.tree_folder, tree))
            print(f"Tree names cleaned in {self.tree_folder}.")
        
        except Exception as e:
            print(f"Error cleaning tree names: {e}")
            raise
        
         
    def extract_top_params(self):
        """Extracts tree topology parameters from the generated trees with rpanda."""
        if self.params_file == "none":
            print("No parameters file found. Please run extract_substitution_params() first.")
            return
        
        cmd = [
            "Rscript",
            "src/model_gen_aa/extract_treetop.R",
            self.tree_folder,
            self.params_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Tree topology parameters extracted for {self.label}.")
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting topology parameters: {e}")
            raise
    
    def cleanup_params(self, input_f):
        output_f = self.params_wc_file
        
        try:
            cleaned_df = model_gen_aa.clean_table.clean_protein_evolution_data(input_f, output_f)
            print("Cleaned Data Preview")
            print("=" * 50)
            print(cleaned_df.head())
            
            #Display summary statistics
            model_gen_aa.clean_table.display_summary_stats(cleaned_df)
            
            # Show column names and data types
            print("\nColumn Information:")
            print("=" * 50)
            print(cleaned_df.dtypes)
                        
        except FileNotFoundError:
            print(f"Error: File '{input_f}' not found.")
            print("Please make sure the file exists in the current directory.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    
    def estimate_treedist(self, threshold=0.6):
        try:
            cmd = [
                "python",
                "src/model_gen_aa/treedist.py",
                self.tree_folder,
                "-o", self.params_file,
                "--save-consensus", self.alignment_folder.replace("/alignments", "/consensus.tree"),
                "--threshold", str(threshold),
                "--topology"
            ]
            
            subprocess.run(cmd, check=True)
            print(f"Tree distances estimated for {self.label}.")
            
        except subprocess.CalledProcessError as e:
            print(f"Error estimating tree distances: {e}")
            raise
    
    def generate_model(self, parameters_f):
        #python src/model_gen/modelfit.py <output_folder> [parameter_file]
        try:
            cmd = [
                "python",
                "src/model_gen_aa/modelfit.py",
                self.tree_folder.replace("/trees", ""),
                parameters_f
            ]
            
            subprocess.run(cmd, check=True)
            print('Model generation completed successfully.')
        
        except subprocess.CalledProcessError as e:
            print(f'Error generating model: {e}')
            raise
    
    def sample_model(self, n_samples=100):
        #python src/model_gen/modelfit.py <output_folder> [model_path] [n_samples]
        try:
            cmd = [
                "python",
                "src/model_gen_aa/modelfit.py",
                self.tree_folder.replace("/trees", ""),
                'none',
                self.tree_folder.replace("/trees", "/bd_models.pkl"),
                str(n_samples)
            ]
            
            subprocess.run(cmd, check=True)
            print('Model sampled successfully')
        except subprocess.CalledProcessError as e:
            print(f'Error sampling model: {e}')
            raise

def main():
    print('Begun the script')
    if len(sys.argv) < 4:
        print("Usage: python main.py <system> <label> <alignment_folder> [log]")
        sys.exit(1)
    
    operate = sys.argv[1]
    label = sys.argv[2]
    input_folder = sys.argv[3]
    log = sys.argv[4] if len(sys.argv) > 4 else False
    
    mc = modelConstructor(operate, label, input_folder, params_file=input_folder.replace("alignments", "protein_evolution_parameters.csv"), log=log)
    
    start = time.time()
    #*PFAM SOCP TYPES PIPELINE
    '''
    mc.cleanup_pfam_data()
    print(f'Cleaned data- ELAPSED TIME: {time.time() - start}============================')'''
    mc.extract_substitution_params()
    print(f'Extracted substitution params- ELAPSED TIME: {time.time() - start}============================')
    mc.extract_top_params() 
    print(f'Extracted topology parameters- ELAPSED TIME: {time.time() - start}============================')
    
    mc.cleanup_params(mc.params_wr_file)
    print(f'Cleaned up tables- ELAPSED TIME: {time.time() - start}============================')
    
    mc.generate_model(mc.params_wc_file)
    print(f'Generated model- ELAPSED TIME: {time.time() - start}============================')
    mc.sample_model(n_samples=1000)
    print(f'Sampled from model- ELAPSED TIME: {time.time() - start}============================')
    
    print('COMPLEETTEEEETETETETE!!!!')


if __name__ == "__main__":
    main()