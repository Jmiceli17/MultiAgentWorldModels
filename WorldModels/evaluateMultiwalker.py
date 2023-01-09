"""
evaluate multiwalker.py

Author:
    Joe Miceli
    Mohammed Adib Oumer

Description:
    Script for evaluating a set of controllers that are defined using the weights
    generated from train.py

    Pseudocode:
    create controllers using weights generated from train.py
    initialize list to store total rwds from each episode
    for ep in max_eps:
        reset env
        initialize rwd for this episode
        for step in max_cycles:
            for agent in agents:
                get an observation, rwd, done, truncation, info for this agent
                
                use one-hot-encoding to generate an action for this agent
                ((something like action = controller[agent].get_action(observation)))

                take a step with that action and move on to the next agent
                
                increment the reward for this episode 
        append this reward to the list of rewards
    take mean of list of rwds
    return mean of list of rwds (we want to get around 300)

Usage:
    python evaluateMultiwalker.py -c configs/multiwalker.config
"""

from controller import make_controller, simulate_multiple_controllers
from env import make_env
from utils import PARSER
import argparse
import numpy as np

def main(args):

    # Create controllers and populate the weights that were optimized in trainMultiwalker.py
    # WARNING! Make sure you're loading the correct files here!
    controller_0 = make_controller(args, id="ctrl_0")
    controller_0.load_model('results/{}/{}/log/{}.cma.20.8.{}.best.json'.format(args.exp_name, args.env_name, args.env_name, controller_0.ID))
    
    controller_1 = make_controller(args, id="ctrl_1")
    controller_1.load_model('results/{}/{}/log/{}.cma.20.8.{}.best.json'.format(args.exp_name, args.env_name, args.env_name, controller_1.ID))
    
    controller_2 = make_controller(args, id="ctrl_2")
    controller_2.load_model('results/{}/{}/log/{}.cma.20.8.{}.best.json'.format(args.exp_name, args.env_name, args.env_name, controller_2.ID))
    
    controller_list = [controller_0, controller_1, controller_2]

    # Make the environment to be evaluated
    env_eval = make_env(args=args, dream_env=False, render_mode=False)

    # Define number of simulations to use for evaluations
    num_episodes = 100  # TODO: could make this a separate config parameter from args

    # Run the simulations and collect the total number of rewards and steps from each simulation
    print("[INFO] Evaluating controllers...")
    reward_list, step_list = simulate_multiple_controllers(controller_list, env_eval, num_episode=num_episodes)

    mean_reward = np.mean(reward_list)
    mean_total_steps = np.mean(step_list)

    print("[INFO] Mean cummulative reward per simulation: {}".format(mean_reward))
    print("[INFO] Mean number of steps per simulation: {}".format(mean_total_steps))

    # TODO: Add plots?

    
if __name__ == "__main__":
  args = PARSER.parse_args()
  main(args)    