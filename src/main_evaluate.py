import os
import subprocess
import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python src/main_evaluate.py <experiment_folder_path> <process>")
    
    print('Begun the script...')
    exp_folder_path = sys.argv[1]
    process = sys.argv[2]
    
    try:
        for scop_folder in os.listdir(exp_folder_path):
            if os.path.isdir(os.path.join(exp_folder_path, scop_folder)):
                for seq_folder in os.listdir(os.path.join(exp_folder_path, scop_folder)):
                    hlog_path = os.path.join(exp_folder_path, scop_folder, seq_folder, 'historian/trace.log.1')
                    blog_path = os.path.join(exp_folder_path, scop_folder, seq_folder, 'baliphy-1/C1.trees')
                    
                    if process.lower() == 'trace_parser':
                        hcmd = [
                            'python',  'src/simulation/trace_parser.py',
                            'historian', hlog_path, hlog_path.replace('trace.log.1', 'parsed_trace.log'),
                            '--trees', '--sequences'
                        ]
                        
                        bcmd = [
                            "python", "src/simulation/trace_parser.py",
                            "baliphy", blog_path, blog_path.replace('C1.trees', 'cleaned.trees'),
                            '--trees'
                        ]
                    elif process.lower() == 'clean_treestat':
                        hcmd = [
                            'python', 'src/simulation/clean_treestat.py',
                            hlog_path.replace('trace.log.1', 'treetrace.log'), hlog_path.replace('trace.log.1', 'parsed_trace.log'), 
                            hlog_path.replace('trace.log.1', 'combined_trace.log')
                        ]
                        
                        bcmd = [
                            'python', 'src/simulation/clean_treestat.py',
                            blog_path.replace('C1.trees', 'treetrace.log'), blog_path.replace('C1.trees', 'C1.log'), 
                            blog_path.replace('C1.trees', 'combined_trace.log')
                        ]
                    elif process.lower() == 'convergence':
                        hcmd = [
                            'python', 'src/simulation/convergence.py',
                            hlog_path.replace('trace.log.1', 'combined_trace.log'),
                            hlog_path.replace('trace.log.1', 'mcmcStats')
                        ]
                        
                        bcmd = [
                            'python', 'src/simulation/convergence.py',
                            blog_path.replace('C1.trees', 'combined_trace.log'),
                            blog_path.replace('C1.trees', 'mcmcStats')
                        ]
                    elif process.lower() == 'comparison':
                        hcmd = [
                            'python', 'src/simulation/comparison.py',
                            hlog_path.replace('trace.log.1', 'outputStats'), 'comparison_results.csv'
                        ]
                        
                        bcmd = [
                            'python', 'src/simulation/comparison.py',
                            blog_path.replace('C1.trees', 'outputStats'), 'comparison_results.csv'
                        ]
                    
                    if process.lower() not in ['summarize']:
                        subprocess.run(hcmd)
                        subprocess.run(bcmd)
    
        if process.lower() == 'summarize':
            print('in')
            try:
                summary_cmd = [
                        'python', 'src/utils/compile_results.py',
                        'data/model_gen', 'data/simulation/experiment2',
                        'data/results/experiment2'
                    ]
                subprocess.run(summary_cmd)
            except Exception as e:
                print(f"An error occurred during summarization: {e}")
                sys.exit(1)
                    
    except Exception as e:
        print(f"An error occurred during the process: {e}")
        sys.exit(1)