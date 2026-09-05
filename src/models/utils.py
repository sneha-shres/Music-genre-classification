import tensorflow as tf
import numpy as np
import random
from easydict import EasyDict as edict
import json

with open('src/config.json') as f:
    config = edict(json.load(f))
SEED = config.random_seed

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(SEED)