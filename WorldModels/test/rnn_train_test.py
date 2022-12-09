"""
rnn_train_test.py

Author:
    Joe Miceli
    Mohammed Adib Oumer

Description:
    Script to test the trained rnn model (output of rnn_train.py)

    python rnn_train_test.py -c ../configs/multiwalker.config
"""

import os
import sys
import tensorflow as tf

scriptPath = os.path.realpath(os.path.dirname(sys.argv[0]))
os.chdir(scriptPath)
#append the relative location you want to import from
sys.path.append("../")

from rnn.rnn import MDNRNN, sample_vae
from utils import PARSER

args = PARSER.parse_args()

################################################################### Option 1 - seems to work
model = tf.keras.models.load_model('../results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name),compile=False)#custom_objects={"z_loss_func": z_loss_func}
# model.summary()
rnn=MDNRNN(args=args)
rnn.set_weights(model.get_weights())    # set_weights doesn't work cant not load the model this way because keras.models.load_model will load the configuration that has been defined, not something has been self_customed
rnn.compile(optimizer=rnn.optimizer, loss=rnn.get_loss()) ## Configures the model for training
# print("[INFO] Model Summary")
# rnn.summary()
# # print(rnn.get_weights())

################################################################### Option 2 - seems to work
# rnn=MDNRNN(args=args)
# rnn.load_weights('../results/{}/{}/tf_rnn_weights'.format(args.exp_name, args.env_name)).expect_partial()
# rnn.compile(optimizer=rnn.optimizer, loss=rnn.get_loss()) ## Configures the model for training
# print("[INFO] Model Summary")
# rnn.summary()
# # print(rnn.get_weights())

'''
################################################################### Random things I tried
# print(model)
# infer = model.signatures["serving_default"]
# labeling = infer(tf.constant(np.random.rand(args.rnn_batch_size, args.rnn_max_seq_len, args.rnn_input_seq_width)))
# predict_class = np.argmax(labeling['output_1'].numpy())
# print(infer.structured_outputs, infer)
# for v in infer.trainable_variables:
#     print(v.name)
# model = tf.saved_model.load('../results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name))
# model.summary()

# import tensorflow_hub as hub
# model = tf.keras.Sequential([
#     hub.KerasLayer('../results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name))#,trainable=False
# ])
# # model._set_inputs(np.random.rand(args.rnn_batch_size, args.rnn_max_seq_len, args.rnn_input_seq_width))
# model.build((None, args.rnn_max_seq_len, args.rnn_input_seq_width))
# model.summary()
# print(model.get_weights())
'''

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "../results/{}/{}/series".format(args.exp_name, args.env_name)
raw_data = np.load(os.path.join(DATA_DIR, "series.npz"))
data_mu = raw_data["mu"]
data_logvar = raw_data["logvar"]
data_action =  raw_data["action"]
data_d = raw_data["done"]
N_data = len(data_mu) # should be 10k

def random_batch(num_agents=args.num_agents):
  z = np.zeros((N_data,args.rnn_max_seq_len,args.z_size+num_agents))
  indices = np.random.permutation(N_data)#[:args.rnn_batch_size]
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


raw_z, raw_a, raw_d = random_batch()
print(raw_z.shape, raw_a.shape, raw_d.shape,N_data)
inputs = tf.concat([raw_z, raw_a], axis=2)

dummy_zero = tf.zeros([raw_z.shape[0], 1, raw_z.shape[2]], dtype=tf.float32)
outputs = tf.concat([raw_z[:, 1:, :], dummy_zero], axis=1) # zero pad the end but we don't actually use it
z_mask = 1.0 - raw_d
outputs = tf.concat([outputs, z_mask], axis=2) # use a signal to not pass grad
history = rnn.fit(inputs, outputs, epochs=args.rnn_max_seq_len, verbose=1)
np.savetxt("losses.txt",history.history['loss'])

# losses = []
# for i in range(args.rnn_batch_size):
#     # note I'll use the same input for x & y parameters which is the case for an autoencoder
#     losses.append(rnn.evaluate(x=inputs[i:i+1,:,:],
#                              y=outputs[i:i+1,:,:],
#                              batch_size=None,
#                              verbose=0,
#                              steps=1
#                              ))
#     # print(eval,outputs.shape, inputs.shape)
# # %matplotlib inline
# np.savetxt("losses.txt", losses)

# with tf.device('/device:GPU:0'):
# cmd /k cd c:\crp