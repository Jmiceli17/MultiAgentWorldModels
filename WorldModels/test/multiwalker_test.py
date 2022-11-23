"""
multiwalker_test.py

Author:
    Joe Miceli
    Mohammed Adib

Description:
    Script to test installation of multiwalker environment
    use the following:
    python multiwalker_test.py -c ../configs/multiwalker.config
"""


import gym
import time
from pettingzoo.sisl import multiwalker_v9
from pettingzoo.utils import random_demo

print("Type of MultiWalkerEnv: {}".format(type(multiwalker_v9.raw_env())))

# Instantiate environment
env = multiwalker_v9.env(n_walkers=3, position_noise=1e-3, angle_noise=1e-3, forward_reward=1.0, terminate_reward=-100.0, fall_reward=-10.0, shared_reward=True,
terminate_on_fall=True, remove_on_fall=True, terrain_length=75, max_cycles=50, render_mode="human")

# Put environment back in original state
env.reset()

# random_demo(env, render=True, episodes=500)

try:
    # Simulate some episodes using a rollout simulation and a random policy
    ep = 0
    while ep < 1:
        # Put environment back in original state
        env.reset()
        terminated = False
        while not terminated:

            for agent in env.agent_iter():
                print("======================================================")
                print("agent: {}".format(agent))
                env.render()
                # env.render(
                #     mode="rgb_array",
                #     width=256,
                #     height=256,
                # )        
                observation, reward, terminated, truncation, info = env.last()
                # print("======================================================")
                # print("observation: {}".format(observation))
                # print("reward: {}".format(reward))
                # print("terminated: {}".format(terminated))
                # print("truncation: {}".format(truncation))
                # print("======================================================")

                action = None if terminated or truncation else env.action_space(agent).sample()  # random policy
                print("action: {}".format(action))
                env.step(action)
                time.sleep(0.01) # to allow rendering at a normal speed

            print("======================================================")
        ep += 1

    print("[TEST PASSED!] Environment succesfully simulated")

except gym.error.Error:
    print("[TEST FAILED] Error encounterd")

env.close()
