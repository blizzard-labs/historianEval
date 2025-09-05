import os
import subprocess

exp_folder_path = 'data/simulation/experiment2'



#for exp_folder in os.listdir(clean_dir):
    #exp_folder_path = os.path.join(clean_dir, exp_folder)
if os.path.isdir(exp_folder_path):
    for scop_folder in os.listdir(exp_folder_path):
        if os.path.isdir(os.path.join(exp_folder_path, scop_folder)):
            for seq_folder in os.listdir(os.path.join(exp_folder_path, scop_folder)):
                hlog_path = os.path.join(exp_folder_path, scop_folder, seq_folder, 'historian/trace.log.1')
                blog_path = os.path.join(exp_folder_path, scop_folder, seq_folder, 'baliphy-1/C1.trees')
                '''
                
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
                
                '''
                
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
                
                subprocess.run(hcmd)
                subprocess.run(bcmd)
                
            
            