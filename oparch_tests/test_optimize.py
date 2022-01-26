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


class Test_optimize_funs(unittest.TestCase):
    def setUp(self):
        self.X, self.y = testing_utils.get_xy(samples=100,features=3)
        self.layers = [tf.keras.layers.Dense(6,activation="sigmoid"),
                       tf.keras.layers.Dropout(0.1),
                       tf.keras.layers.Dense(1),
                       tf.keras.layers.Dense(1)]
        configs = utils.get_layers_config(self.layers)
        configs = utils.add_seed_configs(configs)
        self.layers = utils.layers_from_configs(self.layers, configs)
        self.model = tf.keras.models.Sequential(self.layers)
        oparch.__reset_random__()
        
    def build_compile(self):
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
                                           verbose=2,
                                           )
        self.assertAlmostEqual(results[1][1], results[2][1])
    
    def test_opt_learning_rate6(self):
        self.build_compile()
        oparch.__reset_random__()
        hist = self.model.fit(self.X,self.y,
                       epochs=5,
                       batch_size=4,
                       )
        results = oparch.opt_learning_rate(self.model, self.X, self.y,
                                           return_model=False,
                                           learning_rates=[0.1,0.2],
                                           epochs=5,
                                           batch_size=4,
                                           )
        self.assertAlmostEqual(results[1][1], round(hist.history["loss"][-1],5))#TODO: Model fit gives slightly different results
        
    def test_opt_learning_rate7(self):
        '''
        Tests learning rates, fits model normally, and tests learning rates again.
        Compares the the learning rate results for equality.
        '''
        self.build_compile()
        results = oparch.opt_learning_rate(self.model, self.X, self.y,
                                           return_model=False,
                                           learning_rates=[0.1,0.2],
                                           epochs=5,
                                           batch_size=4,
                                           )
        hist = self.model.fit(self.X,self.y,
                       epochs=5,
                       batch_size=4,
                       )
        results2 = oparch.opt_learning_rate(self.model, self.X, self.y,
                                           return_model=False,
                                           learning_rates=[0.1,0.2],
                                           epochs=5,
                                           batch_size=4,
                                           )
        self.assertEqual(str(results), str(results2))
        
    
if __name__ == "__main__":
    unittest.main()