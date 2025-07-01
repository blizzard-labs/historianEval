#!/usr/bin/env python3

import os
import subprocess

class modelConstructor:
    def __init__(self, label, alignment_folder, tree_folder="none", temp_folder="data/model_gen", output_folder="models", params_file="none"):
        self.label = label
        self.alignment_folder = alignment_folder
        self.tree_folder = tree_folder
        self.temp_folder = temp_folder + "/" + label
        self.output_folder = output_folder
        self.output_file = f"{self.output_folder}/{label}.json"
        self.params_file = params_file
        
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
        
        try:
            for file in os.listdir(self.temp_folder):
                if file.endswith(".csv"):
                    self.params_file = os.path.join(self.temp_folder, file)
        except Exception as e:
            print(f"Error finding parameters file in {self.temp_folder}: {e}")
            raise
        
    def cleanup_modeltest_trees(self):
        modeltest_folder = os.path.join(self.temp_folder, "temp_modeltest")
        try:
            for file in os.listdir(modeltest_folder):
                if file.endswith(".tree"):
                    os.rename(os.path.join(modeltest_folder, file), os.path.join(self.tree_folder, file))
            print(f"Trees cleaned and moved to {self.tree_folder}.")
        except Exception as e:
            print(f"Error cleaning up trees: {e}")
            raise
               
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
            "src/model_gen/extract_treetop.R",
            self.tree_folder,
            self.params_file
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Tree topology parameters extracted for {self.label}.")
            
        except subprocess.CalledProcessError as e:
            print(f"Error extracting topology parameters: {e}")
            raise
        except subprocess.TimeoutExpired as e:
            print(f"Command timed out: {e}")
            raise
        
  
def main():
    mc = modelConstructor('V0_sample', "data/model_gen/V0_sample/alignments", params_file="data/model_gen/V0_sample/gtr_indel_parameters.csv")
    #mc.extract_substitution_params()
    #mc.cleanup_modeltest_trees()
    mc.extract_top_params()  
    
    print('COMPLEETTEEEETETETETE!!!!')


if __name__ == "__main__":
    main()
        
        



        
    
    
'''
# Check if we have any results
if (length(rate_results) == 0) {
  stop("No trees were successfully processed.")
}

cat('ASDKFAJSHDKF')
print(rate_results)

# Ensure all results have consistent structure
rate_results <- lapply(rate_results, function(x) {
  if (is.null(x)) return(NULL)
  
  # Convert to data frame to check structure
  df_test <- tryCatch(data.frame(x, stringsAsFactors = FALSE), 
                     error = function(e) NULL)
  if (is.null(df_test)) return(NULL)
  
  return(x)
})

# Remove any NULL results
rate_results <- rate_results[!sapply(rate_results, is.null)]

if (length(rate_results) == 0) {
  stop("No valid results after structure validation.")
}

'''