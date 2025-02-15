import os
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '/home/ah390/Project/OpenSTL/')

import argparse
from dotmap import DotMap
import ruamel.yaml as yaml
from pathlib import Path

from yamls.settings import *

def main():
    ### Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', default='pick-and-place-cloth-flattening-old')
    parser.add_argument('--agent',  default='default') ### TODO: force it let it have input
    parser.add_argument('--hyper_para',  default='')
    args = parser.parse_args()

    cwd = os.getcwd() # get current working directory
    src_index = cwd.find("src") # find the index of "src" in the path
    if src_index != -1: # if "src" is found in the path
        truncated_path = cwd[:src_index+3] # truncate the path at "src"
    else:
        raise Exception("Could not find src directory. Make sure you are running the script from the root of the repository.")
    

    ### Initilise Project
    configs = yaml.safe_load(Path('{}/yamls/agents/{}/{}/{}.yaml'.\
        format(truncated_path, args.agent, args.setting, args.hyper_para)).read_text())
    dict_configs = configs
    configs = DotMap(dict_configs)

    print('Initialising agent {}'.format(args.agent))
    agent = name_to_agent[args.agent](configs)

    print('finish testing')

if __name__ == '__main__':
    main()