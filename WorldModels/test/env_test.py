"""
env_tets.py

Author:
    Joe Miceli
    Mohammed Adib

Description:
    Script to test implementation of multiwalker environment wrapper for world models
    this file must be used with the multiwalker config

    python env_test.py -c ../configs/multiwalker.config
"""

import numpy as np
import random
import os
import gym
import sys 

# TODO: fix this method of importing from relative paths
# First change the cwd to the script path to import modules 
scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
#append the relative location you want to import from
sys.path.append("../")
from env import make_env
from utils import PARSER



# Get the commandline arguments for this file exectuion
args = PARSER.parse_args()
print("Environment Name: {}".format(args.env_name))


env = make_env(args=args, render_mode=args.render_mode, full_episode=args.full_episode, with_obs=True, load_model=False)  ## Model not loaded because it's not trained yet


print("[TEST PASSED!] Arguments parsed and environment succesfully made")