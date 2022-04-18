import oparch
from oparch import optimize, optimize_utils, LossCallback, configurations
from oparch.optimize import *
from oparch.configurations import set_default_misc, set_default_intervals, get_default_misc, get_default_interval
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot
import sklearn
import random
# LOCAL TODO: For some reason, only uses GPU, if CUDA console logging is on
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
__VERSION = "0.0.1"
def __reset_random__():
    np.random.seed(seed=42) #Also sets sk.learn random number
    tf.random.set_seed(42)
    random.seed(42)
def version():
    return __VERSION
__reset_random__()