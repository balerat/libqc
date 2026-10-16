from pathlib import Path
import sys

SL_PATH = Path(r"/Users/doph/Code/phd/projects/")

def init_local(file):
    sys.path.append(str(Path(file).parent))
