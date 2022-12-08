'''
train mdn-rnn from pre-processed data.
also save 1000 initial mu and logvar, for generative experiments (not related to training).
'''

import numpy as np
import os
import json
import tensorflow as tf
import random
import time

num_eps = 10000
save_eps = 1000
num_agents = 3

from rnn.rnn import MDNRNN, sample_vae
from utils import PARSER
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpu_devices:
    tf.config.experimental.set_memory_growth(gpu, True)
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)
tf.config.experimental_run_functions_eagerly # used for debugging

args = PARSER.parse_args()

DATA_DIR = "results/{}/{}/series".format(args.exp_name, args.env_name)
model_save_path = "results/{}/{}/tf_rnn".format(args.exp_name, args.env_name)
model_save_path2 = "results/{}/{}/tf_rnn_weights".format(args.exp_name, args.env_name)
if not os.path.exists(model_save_path):
  os.makedirs(model_save_path)
if not os.path.exists(model_save_path2):
  os.makedirs(model_save_path2)

raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))

# load preprocessed data
## According to the paper, pre-processed data is Z_t from the VAE
data_mu = raw_data["mu"]
data_logvar = raw_data["logvar"]
data_action =  raw_data["action"]
data_d = raw_data["done"]
N_data = len(data_mu) # should be 10k

# z = np.zeros((args.rnn_batch_size,args.rnn_max_seq_len,args.z_size+num_agents))
# indices = np.random.permutation(N_data)[:args.rnn_batch_size]
# for j in range(num_agents):
#   # suboptimal b/c we are always only taking first set of steps
#   mu = data_mu[indices[j::num_agents],j,:]
#   logvar = data_logvar[indices[j::num_agents],j,:]
#   for k in range(args.rnn_max_seq_len):
#     z[j::num_agents,k,:] = sample_vae(mu, logvar)
#     z[j::num_agents,k,:num_agents] = (z[j::num_agents,k,:num_agents]!=0.).astype(int)
# action = data_action[indices,:args.rnn_max_seq_len,:]
# d = tf.cast(data_d[indices], tf.float32)[:,:args.rnn_max_seq_len]
# print(N_data, z.shape, action.shape, d.shape, tf.concat([z, action], axis=2).shape)

'''
# # save 1000 initial mu and logvars. Used for sampling when training in dreams
# initial_z_save_path = "results/{}/{}/tf_initial_z".format(args.exp_name, args.env_name)
# if not os.path.exists(initial_z_save_path):
#   os.makedirs(initial_z_save_path)
# initial_mu = []
# initial_logvar = []
# for i in range(1000):
#   mu = np.copy(data_mu[i][0, :]*num_eps).astype(np.int).tolist()
#   logvar = np.copy(data_logvar[i][0, :]*num_eps).astype(np.int).tolist()
#   initial_mu.append(mu)
#   initial_logvar.append(logvar)
# with open(os.path.join(initial_z_save_path, "initial_z.json"), 'wt') as outfile:
#   json.dump([initial_mu, initial_logvar], outfile, sort_keys=True, indent=0, separators=(',', ': '))
'''

# EDITED to generate proper obs, action, data. 
# TODO: rnn_max_seq_len might need to change
def random_batch():
  z = np.zeros((args.rnn_batch_size,args.rnn_max_seq_len,args.z_size+num_agents))
  indices = np.random.permutation(N_data)[:args.rnn_batch_size]
  for j in range(num_agents):
    # suboptimal b/c we are always only taking first set of steps
    mu = data_mu[indices[j::num_agents],j,:]
    logvar = data_logvar[indices[j::num_agents],j,:]
    for k in range(args.rnn_max_seq_len):
      z[j::num_agents,k,:] = sample_vae(mu, logvar)
      z[j::num_agents,k,:num_agents] = (z[j::num_agents,k,:num_agents]!=0.).astype(int)
  action = data_action[indices,:args.rnn_max_seq_len,:]
  d = tf.cast(data_d[indices], tf.float32)[:,:args.rnn_max_seq_len]
  return z, action, d

rnn = MDNRNN(args=args)
rnn.compile(optimizer=rnn.optimizer, loss=rnn.get_loss()) ## Configures the model for training

# train loop:
start = time.time()
step = 0              ## Number of steps to take in the env
for step in range(args.rnn_num_steps):
  curr_learning_rate = (args.rnn_learning_rate-args.rnn_min_learning_rate) * (args.rnn_decay_rate) ** step + args.rnn_min_learning_rate
  rnn.optimizer.learning_rate = curr_learning_rate
  
  raw_z, raw_a, raw_d = random_batch()
  print(raw_z.shape, raw_a.shape, raw_d.shape)

  inputs = tf.concat([raw_z, raw_a], axis=2)
  if step == 0: # thank you original paper
    rnn._set_inputs(inputs)

  dummy_zero = tf.zeros([raw_z.shape[0], 1, raw_z.shape[2]], dtype=tf.float32)
  z_targ = tf.concat([raw_z[:, 1:, :], dummy_zero], axis=1) # zero pad the end but we don't actually use it
  z_mask = 1.0 - raw_d
  z_targ = tf.concat([z_targ, z_mask], axis=2) # use a signal to not pass grad
    
  if args.env_name == 'DoomTakeCover-v0':
    d_mask = tf.concat([tf.ones([args.rnn_batch_size, 1, 1], dtype=tf.float32), 1.0 - raw_d[:, :-1, :]], axis=1)
    d_targ = tf.concat([raw_d, d_mask], axis=2)
    outputs = [z_targ, d_targ]
  else:
    outputs = z_targ

  loss = rnn.train_on_batch(x=inputs, y=outputs)

  ## Every 20 steps
  if (step%1==0 and step > 0):
    end = time.time()
    time_taken = end-start
    start = time.time()
    
    if args.env_name == 'DoomTakeCover-v0':
      output_log = "step: %d, lr: %.6f, loss: %.4f, z_loss: %.4f, d_loss: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, loss[0], loss[1], loss[2], time_taken)
    else:
      output_log = "step: %d, lr: %.6f, loss: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, loss, time_taken)
    print(output_log)
    
    # dataset = tf.data.Dataset.from_tensors((inputs,outputs))
    # rnn.fit(dataset,epochs=1)
    # rnn.predict(np.random.rand(args.rnn_batch_size, args.rnn_max_seq_len, args.rnn_input_seq_width)) # stupid tf

    print("Saved? ",rnn.save_spec() is not None)
    tf.keras.models.save_model(rnn, model_save_path, include_optimizer=True, save_format='tf')
    rnn.save_weights(model_save_path2, save_format='tf')
