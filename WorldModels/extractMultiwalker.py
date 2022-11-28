"""
extractMultiwalker.py

Author:
    Joe Miceli
    Mohammed Adib

Description:

    Generate episodes from the multiwalker env using a random policy
    Episodes are saved as compressed array .npz files that can be accessed using
    the positional arguments of the savez call 
    see https://numpy.org/doc/stable/reference/generated/numpy.savez.html 

    use the following:
    python extractMultiwalker.py -c ./configs/multiwalker.config
"""

import numpy as np
import random
import os
import gym

from env import make_env
from utils import PARSER

# Get the commandline arguments for this file exectuion
args = PARSER.parse_args()
print("[INFO] Environment Name: {}".format(args.env_name))

# Create log directory
dir_name = 'results/{}/{}/record'.format(args.exp_name, args.env_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Make the env
# TODO: create env using config file
# Note that the rnn model not loaded because it's not trained yet    
env = make_env(args=args, render_mode=args.render_mode, full_episode=args.full_episode, with_obs=True, load_model=False)  

for trial in range(args.max_trials):
    if (trial % 10) == 0:
        print(trial)
    try:
        random_generated_int = random.randint(0, 2**31-1)
        filename = dir_name+"/"+str(random_generated_int)+".npz"
        recording_agent = []
        recording_obs = []
        recording_action = []
        recording_reward = []
        recording_done = []

        # TODO: seed the env

        # Reset the env
        env.reset()

        previousReward = 0.0

        # for i in range(args.max_frames):
        for i in range(50):

            # There's multiple agents in this environment so each of them must apply an action
            for agent in env.agent_iter():
                # Get an observation for this agent
                obs, totalReward, done, truncation, info = env.last()
                # Sample a random action for this agent
                action = None if done or truncation else env.action_space(agent).sample()
                # Apply the action for this agent
                env.step(action)

                # If action was None, set it to an array of zeros for logging purposes
                if action is None:
                    action = np.zeros((4,),dtype=np.float16)

                # The multiwalker env returns the total cummulative reward so we have to subtract it from the prev reward to get the 
                # immediate reward for an action
                # TODO: need to verify this is the correct way to do this
                immediateReward = totalReward-previousReward
                previousReward = totalReward

                # Save the information for this step
                recording_agent.append(agent)
                recording_obs.append(obs.tolist())
                recording_action.append(action.tolist())    # appending as a list to strip the dtype element from array 
                recording_reward.append(immediateReward)
                recording_done.append(done)

            # print("[DEBUGGING] recording_agent: {}".format(recording_agent))
            # print("[DEBUGGING] recording_obs: {}".format(recording_obs))
            # print("[DEBUGGING] recording_action: {}".format(recording_action))
            # print("[DEBUGGING] recording_reward: {}".format(recording_reward))
            # print("[DEBUGGING] recording_done: {}".format(recording_done))

            if done:
                print('[INFO] total reward {}'.format(totalReward))
                break
        recording_agent = np.array(recording_agent, dtype=str) 
        recording_obs = np.array(recording_obs, dtype=np.float16)
        recording_action = np.array(recording_action, dtype=np.float16)
        recording_reward = np.array(recording_reward, dtype=np.float16)
        recording_done = np.array(recording_done, dtype=np.bool_)

        # save the arrays in a compressed file, arrays are accessbile via keyword arguments
        # TODO: change filename to be more descriptive
        np.savez_compressed(filename, agent= recording_agent, obs=recording_obs, action=recording_action, reward=recording_reward, done=recording_done)   

    except gym.error.Error:
        print("stupid gym error, life goes on")
        env.close()
        env = make_env(args=args, render_mode=args.render_mode, full_episode=args.full_episode, with_obs=True, load_model=False)  
        continue

env.close()
