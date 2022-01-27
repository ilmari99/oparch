import oparch
import unittest
import oparch_tests
import sys
from oparch import optimize_utils as utils
import tensorflow.keras as tf
from oparch_tests.test_utils import Test_optimize_utils
from oparch_tests.test_opt_learning_rate import Test_opt_learning_rate
from oparch_tests.test_opt_loss_fun import Test_opt_loss_fun
from oparch_tests.test_opt_activation import Test_opt_activation
with open("test_log.log","w") as sys.stdout:
    unittest.main()
sys.stdout = sys.__stdout__