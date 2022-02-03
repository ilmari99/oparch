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


class Test_opt_activation(unittest.TestCase):
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
        
    def build_compile(self):
        self.model.build(np.shape(self.X))
        self.model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError())
        
    def test_opt_activation1(self):
        with self.assertRaises(KeyError):
            oparch.opt_activation(self.model, 0, self.X, self.y)
            
    
    def test_opt_activation2(self):
        self.build_compile()
        results = oparch.opt_activation(self.model, 0, self.X, self.y,return_model=False,batch_size=16)
        print(f"Test act 2:\n {results}")
        self.assertAlmostEqual(results[1][1], results[2][1])
        
    def test_opt_activation3(self):
        self.build_compile()
        results = oparch.opt_activation(self.model, 3, self.X, self.y,
                                      return_model=False,
                                      return_metric="RELATIVE_IMPROVEMENT_BATCH",
                                      batch_size=16
                                      )
        print(f"Test act 3:\n {results}")
        self.assertAlmostEqual(results[1][1], results[3][1])
    
    def test_opt_activation4(self):
        self.build_compile()
        model = oparch.opt_activation(self.model, 3, self.X, np.exp(self.y),return_metric="nn")
        results = oparch.opt_activation(self.model, 3, self.X, np.exp(self.y),return_model=False)
        results.pop(0)
        best = 1000
        index = 0
        for i,result in enumerate(results):
            if result[1] < best:
                best = result[1]
                index = i
        print(f"Test act 4:\n {results}")
        print(results[index][0])
        print(model.layers[3].get_config().get("activation"))
        self.assertEqual(results[index][0], model.layers[3].activation.__name__)
        
    def test_opt_activation5(self):
        self.build_compile()
        results = oparch.opt_activation(self.model,2, self.X, self.y,
                                           return_model=False,
                                           return_metric="VALIDATION_LOSS",
                                           verbose=0,
                                           )
        print(f"Test act 5:\n {results}")
        self.assertAlmostEqual(results[1][1], results[3][1])
    
    def test_opt_activation6(self):
        self.build_compile()
        oparch.__reset_random__()
        hist = self.model.fit(self.X,self.y,
                       epochs=15,
                       batch_size=16,
                       verbose=0,
                       )
        results = oparch.opt_activation(self.model,2, self.X, self.y,
                                           return_model=False,
                                           epochs=15,
                                           batch_size=16,
                                           )
        from_hist = hist.history["loss"][-1]
        self.assertAlmostEqual(results[1][1], round(from_hist,5))#TODO: Model fit gives slightly different results
        
    def test_opt_activation7(self):
        '''
        Tests learning rates, fits model normally, and tests learning rates again.
        Compares the the learning rate results for equality.
        '''
        self.build_compile()
        results = oparch.opt_activation(self.model,0, self.X, self.y,
                                           return_model=False,
                                           epochs=5,
                                           batch_size=4,
                                           )
        hist = self.model.fit(self.X,self.y,
                       epochs=5,
                       batch_size=4,
                       verbose=0,
                       )
        results2 = oparch.opt_activation(self.model,0, self.X, self.y,
                                           return_model=False,
                                           epochs=5,
                                           batch_size=4,
                                           )
        self.assertEqual(str(results), str(results2))
        
    def test_opt_activation8(self):
        self.build_compile()
        with self.assertRaises(KeyError):
            oparch.opt_activation(self.model, 1, self.X, self.y)
        
    
if __name__ == "__main__":
    unittest.main()