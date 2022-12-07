import numpy as np
import random

import json
import sys

from env import make_env
import time

from rnn.rnn import MDNRNN

# controls whether we concatenate (z, c, h), etc for features used for car.
MODE_ZCH = 0
MODE_ZC = 1
MODE_Z = 2
MODE_Z_HIDDEN = 3 # extra hidden later
MODE_ZH = 4       ## This will probably be what's used for multiwalker

def make_controller(args, id=None):
  # can be extended in the future.
  controller = Controller(args, id)
  return controller

def clip(x, lo=0.0, hi=1.0):
  return np.minimum(np.maximum(x, lo), hi)

class Controller:
  ''' simple one layer model for car racing '''
  ## TODO: does anything in here need to change for multiwalker?
  def __init__(self, args, id):
    self.env_name = args.env_name
    self.exp_mode = args.exp_mode
    self.input_size = args.z_size + args.state_space * args.rnn_size
    self.z_size = args.z_size
    self.a_width = args.a_width
    self.args = args
    self.ID = id

    if self.exp_mode == MODE_Z_HIDDEN: # one hidden layer
      print("[INFO] Creating a controller with a hidden layer")
      self.hidden_size = 40 # TODO: config?
      self.weight_hidden = np.random.randn(self.input_size, self.hidden_size)
      self.bias_hidden = np.random.randn(self.hidden_size)
      self.weight_output = np.random.randn(self.hidden_size, self.a_width)
      self.bias_output = np.random.randn(self.a_width)
      self.param_count = ((self.input_size+1)*self.hidden_size) + (self.hidden_size*self.a_width+self.a_width)
    else:
      self.weight = np.random.randn(self.input_size, self.a_width)
      self.bias = np.random.randn(self.a_width)
      self.param_count = (self.input_size)*self.a_width+self.a_width

    self.render_mode = args.render_mode

  ## TODO: what do we need to do to modify this for multiwalker??
  def get_action(self, h):
    '''
    action = np.dot(h, self.weight) + self.bias
    action[0] = np.tanh(action[0])
    action[1] = sigmoid(action[1])
    action[2] = clip(np.tanh(action[2]))
    '''
    if self.exp_mode == MODE_Z_HIDDEN: # one hidden layer
      h = np.tanh(np.dot(h, self.weight_hidden) + self.bias_hidden)
      action = np.tanh(np.dot(h, self.weight_output) + self.bias_output)
    else:
      action = np.tanh(np.dot(h, self.weight) + self.bias)  # This should be what's used for multiwalker, keeps outputs between -1 and 1
    
    # Car Racing steering wheel action has range -1 to 1, the acceleration pedal ranges from 0 to 1, and the brakes range from 0 to 1
    if self.env_name == 'CarRacing-v0': 
      action[1] = (action[1]+1.0) / 2.0
      action[2] = clip(action[2])

    # Check that action is correct size for the multiwalker env
    if self.env_name == 'multiwalker_v9':
      assert len(action)==4, "[error][controller.py], action should be 4 elements for multiwalker environment"

    return action

  def set_model_params(self, model_params):
    if self.exp_mode == MODE_Z_HIDDEN: # one hidden layer
      params = np.array(model_params)
      cut_off = (self.input_size+1)*self.hidden_size
      params_1 = params[:cut_off]
      params_2 = params[cut_off:]
      self.bias_hidden = params_1[:self.hidden_size]
      self.weight_hidden = params_1[self.hidden_size:].reshape(self.input_size, self.hidden_size)
      self.bias_output = params_2[:self.a_width]
      self.weight_output = params_2[self.a_width:].reshape(self.hidden_size, self.a_width)
    else:
      self.bias = np.array(model_params[:self.a_width])
      self.weight = np.array(model_params[self.a_width:]).reshape(self.input_size, self.a_width)

  ## See the Jupyter notebook file for an example of this getting used
  def load_model(self, filename):
    with open(filename) as f:    
      data = json.load(f)
    print('loading file %s' % (filename))
    self.data = data
    model_params = np.array(data[0]) # assuming other stuff is in data
    self.set_model_params(model_params)

  def get_random_model_params(self, stdev=0.1):
    #return np.random.randn(self.param_count)*stdev
    return np.random.standard_cauchy(self.param_count)*stdev # spice things up

  def init_random_model_params(self, stdev=0.1):
    params = self.get_random_model_params(stdev=stdev)
    self.set_model_params(params)

def simulate(controller, env, train_mode=False, render_mode=True, num_episode=5, seed=-1, max_len=-1):
  reward_list = []
  t_list = []

  max_episode_length = controller.args.max_frames

  if train_mode and max_len > 0:
    max_episode_length = max_len

  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)

  for episode in range(num_episode):
    if train_mode: print('episode: {}/{}'.format(episode, num_episode))
    obs = env.reset()

    total_reward = 0.0
    for step in range(max_episode_length):
      ## TODO: do not render if this is the multiwalker env (this isn't how render is called for that env)
      if render_mode:
        env.render("human")
      else:
        env.render('rgb_array')

      action = controller.get_action(obs)
      obs, reward, done, info = env.step(action)

      total_reward += reward
      if done:
        break

    if render_mode:
      print("total reward", total_reward, "timesteps", step)
      env.close()
    reward_list.append(total_reward)
    t_list.append(step)
  return reward_list, t_list

def simulate_multiple_controllers(controller_dict, env, train_mode=False, render_mode=False, num_episode=5, seed=-1, max_len=-1):
  """
  Function for simulating multiple controllers in a multi-agent environment, initially only intended to support the multiwalker env
  """

  # Initialize list to store total rewards from each episode
  reward_list = []
  # Initialize list to store the number of steps taken in each episode
  t_list = []
  # Use the first controller to get the max ep length (note that each controller should have the same arguments)
  # TODO: make sure the key to this dictionary isn't hardcoded
  max_episode_length = controller_dict[0].args.max_frames # should be equal to env.max_cycles

  # Override max_episode length if we're using this simulation for training
  if train_mode and max_len > 0:
    max_episode_length = max_len

  # Seed the environment #TODO: need to verify this is correct for multiwalker
  if (seed >= 0):
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)    

  # Run num_episode # of simulations
  for episode in range(num_episode):

    if train_mode: 
      print('episode: {}/{}'.format(episode, num_episode))

    # Initialize the environment
    env.reset()
    total_reward = 0.0
    
    for step in range(max_episode_length):

      controller_id = 0

      # There's multiple agents in this environment so each of them must apply an action
      for agent in env.agent_iter():
        # Get an observation for this agent
        obs, totalRewardFromStep, done, truncation, info = env.last()
        # Sample a random action for this agent
        action = None if done or truncation else controller_dict[controller_id].get_action(obs) # TODO: need to figure out to access individual controllers
        # Apply the action for this agent
        env.step(action)

        # TODO: verify multiwalker env supples the total reward up to that point!!!!!!!!!!!!!!!
        # Update the total reward, each agent should be getting the same reward so it's ok to update it during each agent's actions
        total_reward = totalRewardFromStep

        controller_id += 1

      # If the env is terminated, start the next simulation, this value is the the same for all agents
      if done:
        break

    # Store the total reward and the number of steps taken during this simulation  
    reward_list.append(total_reward)
    t_list.append(step)

  return reward_list, t_list

