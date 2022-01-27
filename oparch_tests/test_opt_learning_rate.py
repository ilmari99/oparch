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


class Test_opt_learning_rate(unittest.TestCase):
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
        results1 = oparch.opt_learning_rate(self.model,self.X, self.y,
                                           return_model=False,
                                           epochs = 12,
                                           batch_size = 70,
                                           samples = 200,
                                           )
        results2 = oparch.opt_learning_rate(self.model,self.X, self.y,
                                           return_model=False,
                                           epochs = 12,
                                           batch_size = 70,
                                           samples = 50,
                                           )
        self.assertNotEqual(str(results1), str(results2))
        
    def test_opt_learning_rate3(self):
        self.build_compile()
        results = oparch.opt_learning_rate(self.model, self.X, self.y,
                                           return_model=False,
                                           learning_rates=[0.03,0.5])
        tested = [lr[0] for lr in results]
        self.assertTrue(0.01 in tested)
    
    def test_opt_learning_rate4(self):
        self.build_compile()
        results1 = oparch.opt_learning_rate(self.model,self.X, self.y,
                                           return_model=False,
                                           epochs = 12,
                                           batch_size = 70,
                                           samples = 200,
                                           optimizer=tf.keras.optimizers.Adam(),
                                           )
        results2 = oparch.opt_learning_rate(self.model,self.X, self.y,
                                           return_model=False,
                                           epochs = 12,
                                           batch_size = 70,
                                           samples = 200,
                                           )
        self.assertNotEqual(str(results1), str(results2))
        
    def test_opt_learning_rate5(self):
        self.build_compile()
        results = oparch.opt_learning_rate(self.model, self.X, self.y,
                                           return_model=False,
                                           learning_rates=[0.01,0.5],
                                           return_metric="RELATIVE_IMPROVEMENT_EPOCH",
                                           verbose=0,
                                           )
        corr = "[[0.01, -0.18671], [0.5, 1.81569]]"
        self.assertAlmostEqual(str(results), corr)
    
    def test_opt_learning_rate6(self):
        self.build_compile()
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
        best = 1000
        for i,result in enumerate(results):
            if result[1] < best:
                best = result[1]
                index = i
        self.assertAlmostEqual(results[index][1], round(hist.history["loss"][-1],5))#TODO: Model fit gives slightly different results
        
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
                       epochs=2,
                       batch_size=4,
                       verbose=0,
                       )
        results2 = oparch.opt_learning_rate(self.model, self.X, self.y,
                                           return_model=False,
                                           learning_rates=[0.1,0.2],
                                           epochs=5,
                                           batch_size=4,
                                           )
        self.assertEqual(str(results), str(results2))
        
    def test_opt_learning_rate8(self):
        self.build_compile()
        model = oparch.opt_learning_rate(self.model, self.X, np.exp(self.y),return_metric="nn")
        results = oparch.opt_learning_rate(self.model, self.X, np.exp(self.y),return_model=False)
        best = 1000
        for i,result in enumerate(results):
            if result[1] < best:
                best = result[1]
                index = i
        self.assertEqual(results[index][0], model.optimizer.get_config().get("learning_rate"))
        
    
if __name__ == "__main__":
    unittest.main()