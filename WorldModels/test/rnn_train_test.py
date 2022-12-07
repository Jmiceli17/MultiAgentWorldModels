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
rnn=MDNRNN(args=args)

rnn.load_weights('../results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name))    # set_weights doesn't work cant not load the model this way because keras.models.load_model will load the configuration that has been defined, not something has been self_customed
rnn.compile(optimizer=rnn.optimizer, loss=rnn.get_loss()) ## Configures the model for training
print("[INFO] Model Summary")
rnn.summary()

