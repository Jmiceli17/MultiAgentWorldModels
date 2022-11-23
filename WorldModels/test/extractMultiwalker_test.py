"""
extractMultiwalker_test.py

Author:
    Joe Miceli
    Mohammed Adib

Description:

    Script to test the generation of multiwalker training data was performed correctly
"""


import numpy as np
import os
import sys 



try:

    # Get the path to this script
    script_dir = os.path.dirname(__file__)
    # append the relative location you want to import from
    dirname = os.path.join(script_dir,"../results/WorldModels/multiwalker_v9/record")
    print("DIR NAME: {}".format(dirname))
    # Get list of filenames
    filenames = os.listdir(dirname)
    # Loop thru all file names
    for j, fname in enumerate(filenames):
            if not fname.endswith('npz'): 
                continue
            file_path = os.path.join(dirname, fname)
            with np.load(file_path) as data:
                raw_agents = data['agent']
                raw_obs = data['obs']
                raw_actions = data['action']
                raw_rewards = data['reward']
                raw_done = data['done']

                # TODO: add a better test here
                print("[DEBUGGING] raw_agents: {}".format(raw_agents))
                print("[DEBUGGING] raw_obs: {}".format(raw_obs))
                print("[DEBUGGING] raw_actions: {}".format(raw_actions))
                print("[DEBUGGING] raw_rewards: {}".format(raw_rewards))
                print("[DEBUGGING] raw_done: {}".format(raw_done))

                
    print("[TEST PASSED!] Data printed successfully")            


except:
     print("[TEST FAILED] Some error encountered")            
        
