import argparse
import os
import pandas as pd

def compile_results(model_gen_dir, simulation_dir, results_dir):
    """
    Compiles results from model generation and simulation into CSV files.
    """
    
    historian_data = []
    baliphy_data = []

    # Load and process benchmark results
    benchmark_file = os.path.join(simulation_dir, 'benchmark_results.csv')
    if not os.path.exists(benchmark_file):
        print(f"Benchmark file not found at {benchmark_file}")
        return
    benchmark_df = pd.read_csv(benchmark_file)
    
    
    # Process benchmark data to extract scop_type, experiment, and seq_id
    def extract_info(path):
        parts = path.split('/')
        scop_exp_part = parts[-2]
        seq_part = parts[-1]
        
        scop_type = int(scop_exp_part[scop_exp_part.find('t')+1:scop_exp_part.find('e')])
        experiment = int(scop_exp_part[scop_exp_part.find('e')+1:])
        seq_id = int(seq_part.split('_')[1])
        
        return scop_type, experiment, seq_id

    benchmark_info = benchmark_df['path'].apply(extract_info)
    benchmark_df[['scop_type', 'experiment', 'seq_id']] = pd.DataFrame(benchmark_info.tolist(), index=benchmark_df.index)
    
    benchmark_df = benchmark_df.melt(id_vars=['scop_type', 'experiment', 'seq_id'], 
                                     value_vars=['elapsed_historian', 'elapsed_baliphy'], 
                                     var_name='tool', value_name='wall_clock_s')
    benchmark_df['tool'] = benchmark_df['tool'].str.replace('elapsed_', '')
    benchmark_df.loc[benchmark_df['tool'] == 'baliphy', 'tool'] = 'baliphy-1'

    def extract_mcmc_parameters(mcmc_file):
        """
        Extract specific parameters from mcmc_statistics.csv file.
        Returns a dictionary with the extracted parameters.
        """
        if not os.path.exists(mcmc_file):
            return {}
        
        try:
            mcmc_df = pd.read_csv(mcmc_file)
            
            # Initialize result dictionary
            mcmc_params = {}
            
            # Debug: Print all parameter names to help identify the correct ones
            print(f"Parameters found in {mcmc_file}: {mcmc_df['Parameter'].tolist()}")
            
            # Extract percent_unique_topologies - try multiple variations
            percent_unique_patterns = ['Percent_unique_topologies', 'percent_unique_topologies', 'Tree Topology', 'topology']
            for pattern in percent_unique_patterns:
                percent_unique_row = mcmc_df[mcmc_df['Parameter'].str.contains(pattern, case=False, na=False)]
                if not percent_unique_row.empty:
                    # Try different column names for the value
                    for col in ['Mean', 'Percent_unique_topologies', 'Value']:
                        if col in mcmc_df.columns and col == 'Percent_unique_topologies':
                            print(col)
                            mcmc_params['percent_unique_topologies'] = percent_unique_row.iloc[0][col]
                            break
                    break
            
            # Extract likelihood ESS
            likelihood_row = mcmc_df[mcmc_df['Parameter'].str.contains('likelihood', case=False, na=False)]
            if not likelihood_row.empty:
                mcmc_params['likelihood_ess'] = likelihood_row.iloc[0]['ESS'] if 'ESS' in mcmc_df.columns else None
            
            # Extract n_samples - try multiple variations
            n_samples_patterns = ['N_samples', 'n_samples', 'samples', 'iter']
            for pattern in n_samples_patterns:
                n_samples_row = mcmc_df[mcmc_df['Parameter'].str.contains(pattern, case=False, na=False)]
                if not n_samples_row.empty:
                    # Try different column names for the mode/value
                    for col in ['N_samples', 'Value']:
                        if 'N_samples' in mcmc_df.columns:
                            mcmc_params['n_samples'] = n_samples_row.iloc[0][col]
                            break
                    break
            
            return mcmc_params
            
        except Exception as e:
            print(f"Error reading {mcmc_file}: {e}")
            return {}

    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Iterate through SCOP types
    for scop_type_folder in os.listdir(model_gen_dir):
        scop_type_path = os.path.join(model_gen_dir, scop_type_folder)
        if not os.path.isdir(scop_type_path) or not scop_type_folder.startswith('SCOPtype'):
            continue
        scop_type_num = int(scop_type_folder.replace('SCOPtype', ''))

        # Iterate through experiments
        for i in range(1, 3):
            exp_params_file = os.path.join(scop_type_path, f'experiment{i}_parameters.csv')
            if not os.path.exists(exp_params_file):
                continue

            exp_params_df = pd.read_csv(exp_params_file)

            for _, seq_params in exp_params_df.iterrows():
                sim_label = f'SCOPt{scop_type_num}e{i}'

                seq_name = seq_params['sequence_name']
                seq_id = int(seq_name.split('_')[1])
                # Append SCOP type and experiment to sequence_name
                seq_name_with_scop = f"{seq_name}_SCOPt{scop_type_num}e{i}"

                # Process Historian
                historian_base_path = os.path.join(simulation_dir, sim_label, f'seq_{seq_id}', 'historian')
                historian_mcmc_file = os.path.join(historian_base_path, 'mcmcStats', 'mcmc_statistics.csv')
                historian_output_file = os.path.join(historian_base_path, 'outputStats', 'comparison_results.csv')

                if os.path.exists(historian_mcmc_file) and os.path.exists(historian_output_file):
                    # Extract specific MCMC parameters
                    mcmc_params = extract_mcmc_parameters(historian_mcmc_file)
                    
                    # Get output stats (keeping original functionality)
                    output_stats = pd.read_csv(historian_output_file).to_dict('records')[0] if os.path.exists(historian_output_file) else {}
                    
                    # Get wall clock time
                    wall_clock_time_series = benchmark_df[(benchmark_df['scop_type'] == scop_type_num) & 
                                                   (benchmark_df['experiment'] == i) & 
                                                   (benchmark_df['seq_id'] == seq_id) & 
                                                   (benchmark_df['tool'] == 'historian')]['wall_clock_s']
                    wall_clock_time = wall_clock_time_series.values[0] if not wall_clock_time_series.empty else None
                    
                    # Combine all data
                    historian_data.append({
                        **seq_params.to_dict(), 
                        'sequence_name': seq_name_with_scop, 
                        **mcmc_params, 
                        **output_stats, 
                        'wall_clock_s': wall_clock_time
                    })

                # Process BAli-Phy
                baliphy_base_path = os.path.join(simulation_dir, sim_label, f'seq_{seq_id}', 'baliphy-1')
                baliphy_mcmc_file = os.path.join(baliphy_base_path, 'mcmcStats', 'mcmc_statistics.csv')
                baliphy_output_file = os.path.join(baliphy_base_path, 'outputStats', 'comparison_results.csv')

                if os.path.exists(baliphy_mcmc_file) and os.path.exists(baliphy_output_file):
                    # Extract specific MCMC parameters
                    mcmc_params = extract_mcmc_parameters(baliphy_mcmc_file)
                    
                    # Get output stats (keeping original functionality)
                    output_stats = pd.read_csv(baliphy_output_file).to_dict('records')[0] if os.path.exists(baliphy_output_file) else {}
                    
                    # Get wall clock time
                    wall_clock_time_series = benchmark_df[(benchmark_df['scop_type'] == scop_type_num) & 
                                                   (benchmark_df['experiment'] == i) & 
                                                   (benchmark_df['seq_id'] == seq_id) & 
                                                   (benchmark_df['tool'] == 'baliphy-1')]['wall_clock_s']
                    wall_clock_time = wall_clock_time_series.values[0] if not wall_clock_time_series.empty else None
                    
                    # Combine all data
                    baliphy_data.append({
                        **seq_params.to_dict(), 
                        'sequence_name': seq_name_with_scop, 
                        **mcmc_params, 
                        **output_stats, 
                        'wall_clock_s': wall_clock_time
                    })

    if historian_data:
        historian_df = pd.DataFrame(historian_data)
        historian_df.to_csv(os.path.join(results_dir, 'historian_results.csv'), index=False)
        print(f"Historian results compiled and saved to {os.path.join(results_dir, 'historian_results.csv')}")

    if baliphy_data:
        baliphy_df = pd.DataFrame(baliphy_data)
        baliphy_df.to_csv(os.path.join(results_dir, 'baliphy_results.csv'), index=False)
        print(f"BAli-Phy results compiled and saved to {os.path.join(results_dir, 'baliphy_results.csv')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compile simulation results from Historian and BAli-Phy runs.'
    )
    parser.add_argument(
        'model_gen',
        type=str,
        help='Path to the model_gen directory containing parameter files.'
    )
    parser.add_argument(
        'simulation',
        type=str,
        help='Path to the simulation directory containing output files.'
    )
    parser.add_argument(
        'results',
        type=str,
        help='Path to the directory where the output CSV files will be saved.'
    )

    args = parser.parse_args()
    compile_results(args.model_gen, args.simulation, args.results)