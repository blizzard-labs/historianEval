import re
import os
import logging

def strip_metadata(newick_file):
    '''
    Strips metadata from extended New Hampshire format (.nhx) to standard Newick (.tree)
    # Example input
    extended_newick = "(A:0.1[&support=95],B:0.2[&support=90]):0.3;"
    plain_newick = strip_metadata(extended_newick)

    print(plain_newick)  # Output: (A:0.1,B:0.2):0.3;
    '''

    with open(newick_file, "r") as f:
        newick_str = f.read()
    
    stripped = re.sub(r"\[[^\[\]]*?\]", "", newick_str)
    
    print(newick_file)
    
    with open(newick_file.replace(".newick", ".tree"), "w") as f:
        f.write(stripped.strip())
        
    os.remove(newick_file)
    
    return stripped.strip()

class StreamToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, message):
        # Handle multi-line print statements
        for line in message.rstrip().splitlines():
            self.logger.log(self.level, line)

    def flush(self):
        pass  # Needed for compatibility

