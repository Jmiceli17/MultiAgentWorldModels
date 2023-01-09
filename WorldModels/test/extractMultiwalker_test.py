"""
extractMultiwalker_test.py

Author:
    Joe Miceli
    Mohammed Adib Oumer

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
    # dirname = os.path.join(str(script_dir+"/"),"..","/results/WorldModels/multiwalker_v9/record")
    print("[INFO] DIR NAME: {}".format(dirname))
    # Get list of filenames
    filenames = os.listdir(dirname)
    print("[INFO] number of files: {}".format(len(filenames)))
    # Loop thru all file names
    for j, fname in enumerate(filenames):
            if not fname.endswith('npz'): 
                continue
            file_path = os.path.join(dirname,fname)

            with np.load(file_path) as data:
                print("[INFO] File name: {}".format(fname))

                raw_agents = data['agent']
                raw_obs = data['obs']
                raw_actions = data['action']
                raw_rewards = data['reward']
                raw_done = data['done']

                # TODO: add a better test here
                # # print("[DEBUGGING] raw_agents:\n {}".format(raw_agents))
                # print("[DEBUGGING] raw_obs:\n {}".format(raw_obs))
                # print("[DEBUGGING] raw_actions:\n {}".format(raw_actions))
                # print("[DEBUGGING] raw_rewards:\n {}".format(raw_rewards))
                # # print("[DEBUGGING] raw_done:\n {}".format(raw_done))
                
            print("O,A,R,D,ag: ",raw_obs.shape, raw_actions.shape, raw_rewards.shape, raw_done.shape, raw_agents.shape)
            print("Omean,Amean: ",raw_obs.mean(axis=0).shape, raw_actions.mean(axis=0).shape)
            print("Ovar,Avar: ",raw_obs.var(axis=0).shape, raw_actions.var(axis=0).shape)
    print("[TEST PASSED!] Data printed successfully")            


except:
     print("[TEST FAILED] Some error encountered")            
        
