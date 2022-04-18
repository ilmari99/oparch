import oparch
from oparch import optimize, optimize_utils, LossCallback, configurations
from oparch.optimize import *
from oparch.configurations import set_default_misc, set_default_intervals
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot
import sklearn
import random
from pathlib import Path
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
def __reset_random__():
    np.random.seed(seed=42) #Also sets sk.learn random number
    tf.random.set_seed(42)
    random.seed(42)
__reset_random__()