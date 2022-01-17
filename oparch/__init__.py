from . import optimize
from . import OptimizedModel
from . import model_optimizer_tools
from . import LossCallback
from . import configurations
import os
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.random.seed(seed=42) #for reproducibility
tf.random.set_seed(42)