import os
import ete3

class Comparison:
    """
    Comparison between model and simulated tree results.
    """
    
    def __init__(self, true_ancestry_folder, simulated_ancestry_folder, output_folder="comparison_results"):
        self.true_ancestry_folder = true_ancestry_folder
        self.simulated_ancestry_folder = simulated_ancestry_folder
        
        os.mkdir(output_folder, exist_ok=True)
    
    def rf_distance(self, true_tree, simulated_tree):
        true_tree = ete3.Tree(true_tree)
        simulated_tree = ete3.Tree(simulated_tree)
        
        return ete3.TreeDistance(true_tree, simulated_tree).robinson_foulds()
    
    def clean_params():
        pass
    
    def compare_params(self, true_params, simulated_params):
        """
        Compare parameters from true and simulated data.
        """
        # Placeholder for parameter comparison logic
        pass
    
    def calculate_spfn(self, true_algn, simulated_algn):
        """
        Calculate SPFN: java -jar FastSP.jar -r reference_alignment_file -e estimated_alignment_file (with fast sp)
        """
        pass
    
    def generate_comparison_report(self, true_tree, simulated_tree, true_params, simulated_params):
        """
        Generate a report comparing the true and simulated trees and parameters.
        """
        rf_dist = self.rf_distance(true_tree, simulated_tree)
        
        # Placeholder for generating a report
        report = {
            "rf_distance": rf_dist,
            "true_params": true_params,
            "simulated_params": simulated_params
        }
        pass
    
        #return report
        
