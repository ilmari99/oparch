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


class Test_opt_dense_units(unittest.TestCase):
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
        
    def test_opt_dense_units1(self):
        with self.assertRaises(KeyError):
            oparch.opt_dense_units(self.model, 0, self.X, self.y)
            
    
    def test_opt_dense_units2(self):
        self.build_compile()
        results = oparch.opt_dense_units(self.model, 2, self.X, self.y,return_model=False,batch_size=16)
        print(f"Test act 2:\n {results}")
        corr = [[1, 0.01448], [2, 0.01755], [4, 0.03776], [8, 0.02701],
                [16, 0.02578], [32, 0.02059], [64, 0.03791], [128, 0.01847], [None, 0.05649]]
        self.assertAlmostEqual(results, corr)
        
    def test_opt_dense_units3(self):
        self.build_compile()
        results = oparch.opt_dense_units(self.model, 3, self.X, self.y,
                                      return_model=False,
                                      return_metric="RELATIVE_IMPROVEMENT_BATCH",
                                      batch_size=16
                                      )
        print(f"Test act 3:\n {results}")
        corr = [[1, -0.01734], [2, -0.05649], [4, -0.06604], [8, -0.06982], [16, -0.07228],
                [32, -0.084], [64, -0.09005], [128, -0.07589], [None, -0.06007]]
        self.assertAlmostEqual(results, corr)
    
    def test_opt_dense_units4(self):
        self.build_compile()
        model = oparch.opt_dense_units(self.model, 3, self.X, np.exp(self.y),return_metric="nn")
        results = oparch.opt_dense_units(self.model, 3, self.X, np.exp(self.y),return_model=False)
        best = 1000
        index = 0
        for i,result in enumerate(results):
            if result[1] < best:
                best = result[1]
                index = i
        print(f"Test act 4:\n {results}")
        print(results[index][0])
        print(model.layers[3].get_config().get("units"))
        self.assertEqual(results[index][0], model.layers[3].get_config().get("units"))
        
    def test_opt_dense_units5(self):
        self.build_compile()
        results = oparch.opt_dense_units(self.model,2, self.X, self.y,
                                           return_model=False,
                                           return_metric="VALIDATION_LOSS",
                                           verbose=0,
                                           )
        print(f"Test act 5:\n {results}")
        corr = [[1, 0.00063], [2, 0.00287], [4, 0.00544], [8, 0.00738],
                [16, 0.00189], [32, 0.00128], [64, 0.00209], [128, 0.00175], [None, 0.07601]]
        self.assertAlmostEqual(results, corr)
    
    def test_opt_dense_units6(self):
        self.build_compile()
        oparch.__reset_random__()
        hist = self.model.fit(self.X,self.y,
                       epochs=15,
                       batch_size=16,
                       verbose=0,
                       )
        results = oparch.opt_dense_units(self.model,2, self.X, self.y,
                                           return_model=False,
                                           epochs=15,
                                           batch_size=16,
                                           )
        best = 100000
        for i,result in enumerate(results):
            if result[1] < best:
                best = result[1]
                index = i
        from_hist = hist.history["loss"][-1]
        self.assertAlmostEqual(results[index][1], round(from_hist,5))#TODO: Model fit gives slightly different results
        
    def test_opt_dense_units7(self):
        '''
        Tests learning rates, fits model normally, and tests learning rates again.
        Compares the the learning rate results for equality.
        '''
        self.build_compile()
        results = oparch.opt_dense_units(self.model,0, self.X, self.y,
                                           return_model=False,
                                           epochs=5,
                                           batch_size=4,
                                           )
        hist = self.model.fit(self.X,self.y,
                       epochs=5,
                       batch_size=4,
                       verbose=0,
                       )
        results2 = oparch.opt_dense_units(self.model,0, self.X, self.y,
                                           return_model=False,
                                           epochs=5,
                                           batch_size=4,
                                           )
        self.assertEqual(str(results), str(results2))
        
    def test_opt_dense_units8(self):
        self.build_compile()
        with self.assertRaises(KeyError):
            oparch.opt_dense_units(self.model, 1, self.X, self.y)
            
    def test_opt_dense9(self):
        self.build_compile()
        res1 = oparch.opt_dense_units(self.model, 2, self.X, self.y,return_model=False,test_nodes = [5])
        res2 = oparch.opt_dense_units(self.model, 2, self.X, self.y,return_model=False,test_nodes = [5])
        self.assertEqual(str(res1), str(res2))
        
    
if __name__ == "__main__":
    unittest.main()