#!/usr/bin/env python3

import os
import subprocess

class modelConstructor:
    def __init__(self, label, alignment_folder, tree_folder="none", temp_folder="data/model_gen", output_folder="models"):
        self.label = label
        self.alignment_folder = alignment_folder
        self.tree_folder = tree_folder
        self.temp_folder = temp_folder + "/" + label
        self.output_folder = output_folder
        self.output_file = f"{self.output_folder}/{label}.json"
        
        os.makedirs(self.temp_folder, exist_ok=True)
        if (self.tree_folder == "none"):
            os.makedirs(self.temp_folder + "/trees", exist_ok=True)
            self.tree_folder = self.temp_folder + "/trees"
        
    def extract_substitution_params(self):
        """Extracts substitution parameters from the alignment folder using modeltest-ng."""
        cmd = [
            "python",
            "src/model_gen/extract_params.py",
            self.alignment_folder,
            self.temp_folder,
            "tools/modeltest-ng-osx"
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
    
    def generate_ml_trees(self):
        
    
        
        
        
        
        
    