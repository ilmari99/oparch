if __name__ == "__main__":
    from pathlib import Path
    import sys
    path_root = Path(__file__).parents[1]
    sys.path.append(str(path_root))
import sys
import os
from oparch_tests import testing_utils
import oparch
import unittest
from oparch import optimize_utils as utils
import tensorflow as tf
import numpy as np

class Test_optimize_utils(unittest.TestCase):
    def setUp(self):
        self.X, self.y = testing_utils.get_xy(samples=10,features=3)
        self.layers = [tf.keras.layers.Dense(6,activation="sigmoid"),tf.keras.layers.Dropout(0.1),tf.keras.layers.Dense(1),tf.keras.layers.Dense(1)]
        self.model = tf.keras.models.Sequential(self.layers)
        
    def build_compile(self):
        oparch.__reset_random__()
        self.model.build(np.shape(self.X))
        self.model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError())
        
    def test_opt_learning_rate1(self):
        with self.assertRaises(KeyError):
            oparch.opt_learning_rate(self.model, self.X, self.y)
    
    def test_opt_learning_rate2(self):
        self.build_compile()
        results = oparch.opt_learning_rate(self.model, self.X, self.y,return_model=False)
        self.assertAlmostEqual(results[1][1], results[6][1])
        
    def test_opt_learning_rate3(self):
        self.build_compile()
        results = oparch.opt_learning_rate(self.model, self.X, self.y,return_model=False,learning_rates=[0.01,0.5])
        self.assertAlmostEqual(results[1][1], results[2][1])
    
    def test_opt_learning_rate4(self):
        self.build_compile()
        results = oparch.opt_learning_rate(self.model, self.X, self.y,return_model=False,learning_rates=np.array([0.01,0.5]))
        self.assertAlmostEqual(results[1][1], results[2][1])
        
    def test_opt_learning_rate5(self):
        self.build_compile()
        results = oparch.opt_learning_rate(self.model, self.X, self.y,
                                           return_model=False,
                                           learning_rates=np.array([0.01,0.5]),
                                           learning_metric="RELATIVE_IMPROVEMENT_EPOCH",
                                           verbose=2)
        self.assertAlmostEqual(results[1][1], results[2][1])
    
if __name__ == "__main__":
    unittest.main()