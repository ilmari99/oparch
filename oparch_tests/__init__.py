'''
from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
print(sys.path)
#"C:\\Users\\ivaht\\Desktop\\PYTHON\\Python_scripts\\optarch\\oparch_tests"
'''
from . import test_utils
import unittest
import oparch
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot
import sklearn
import random
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
oparch.__reset_random__()
