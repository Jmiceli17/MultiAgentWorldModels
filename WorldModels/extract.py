'''
saves ~ 200 episodes generated from a random policy
'''

import numpy as np
import random
import os
import gym

from env import make_env
from controller import make_controller

from utils import PARSER

## Get the commandline arguments for this file exectuion
args = PARSER.parse_args()
dir_name = 'results/{}/{}/record'.format(args.exp_name, args.env_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

controller = make_controller(args=args)

total_frames = 0
env = make_env(args=args, render_mode=args.render_mode, full_episode=args.full_episode, with_obs=True, load_model=False)  ## Model not loaded because it's not trained yet
print(env)

for trial in range(args.max_trials):
  try:
    random_generated_int = random.randint(0, 2**31-1)
    filename = dir_name+"/"+str(random_generated_int)+".npz"
    recording_frame = []
    recording_action = []
    recording_reward = []
    recording_done = []

    np.random.seed(random_generated_int)
    # env.seed(random_generated_int) #seems only for doom

    # random policy
    if args.env_name == 'CarRacing-v0':
      controller.init_random_model_params(stdev=np.random.rand()*0.01)
    else:
      repeat = np.random.randint(1, 11)

    tot_r = 0
    [obs, frame] = env.reset() # pixels
    
    for i in range(args.max_frames):
      # # seems for doom only
      # if args.render_mode:
      #   env.render("human")
      # else:
      #   env.render("rgb_array")

      recording_frame.append(frame)
      
      if args.env_name == 'CarRacing-v0':
        action = controller.get_action(obs)
      else:
        if i % repeat == 0:
          action = np.random.rand(1,1) * 2.0 - 1.0
          repeat = np.random.randint(1, 11)

      recording_action.append(action)

      [obs, frame], reward, done, info = env.step(action)
      tot_r += reward

      recording_reward.append(reward)
      recording_done.append(done)

      if done:
        print('total reward {}'.format(tot_r))
        break

    total_frames += (i+1)
    print('total reward {}'.format(tot_r))
    print("dead at", i+1, "total recorded frames for this worker", total_frames)
    recording_frame = np.array(recording_frame, dtype=np.uint8)
    recording_action = np.array(recording_action, dtype=np.float16)
    recording_reward = np.array(recording_reward, dtype=np.float16)
    recording_done = np.array(recording_done, dtype=np.bool)
    
    if (len(recording_frame) > args.min_frames):
      np.savez_compressed(filename, obs=recording_frame, action=recording_action, reward=recording_reward, done=recording_done)   ## This is where the rollout is stored for ingestion by the VAE
  
  except gym.error.Error:
    print("stupid gym error, life goes on")
    env.close()
    env = make_env(args=args, render_mode=args.render_mode, full_episode=False, with_obs=True)
    continue
env.close()
