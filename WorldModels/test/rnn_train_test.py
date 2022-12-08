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

from rnn.rnn import MDNRNN
from utils import PARSER

args = PARSER.parse_args()


################################################################### Option 1 - seems to work
model = tf.keras.models.load_model('../results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name),compile=False)#custom_objects={"z_loss_func": z_loss_func}
# model.summary()
rnn=MDNRNN(args=args)
rnn.set_weights(model.get_weights())    # set_weights doesn't work cant not load the model this way because keras.models.load_model will load the configuration that has been defined, not something has been self_customed
rnn.compile(optimizer=rnn.optimizer, loss=rnn.get_loss()) ## Configures the model for training
print("[INFO] Model Summary")
rnn.summary()
print(rnn.get_weights())

################################################################### Option 2 - seems to work
# rnn=MDNRNN(args=args)
# rnn.load_weights('../results/{}/{}/tf_rnn_weights'.format(args.exp_name, args.env_name)).expect_partial()
# rnn.compile(optimizer=rnn.optimizer, loss=rnn.get_loss()) ## Configures the model for training
# print("[INFO] Model Summary")
# rnn.summary()
# print(rnn.get_weights())

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
