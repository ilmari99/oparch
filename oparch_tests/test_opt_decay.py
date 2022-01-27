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


class Test_opt_decay(unittest.TestCase):
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
        
    def test_opt_opt_decay1(self):
        with self.assertRaises(KeyError):
            oparch.opt_decay(self.model, self.X, self.y)
    
    def test_opt_decay2(self):
        self.build_compile()
        results = oparch.opt_decay(self.model, self.X, self.y,return_model=False)
        self.assertAlmostEqual(results[1][1], results[11][1])
        
    def test_opt_decay3(self):
        self.build_compile()
        results = oparch.opt_decay(self.model, self.X, self.y,return_model=False,decays=[0,0.5])
        self.assertAlmostEqual(results[1][1], results[2][1])
    
    def test_opt_decay4(self):
        self.build_compile()
        results = oparch.opt_decay(self.model, self.X, self.y,return_model=False,decays=np.array([0,0.5]))
        self.assertAlmostEqual(results[1][1], results[2][1])
        
    def test_opt_decay5(self):
        self.build_compile()
        results = oparch.opt_decay(self.model, self.X, self.y,
                                           return_model=False,
                                           decays=np.array([0,0.5]),
                                           return_metric="RELATIVE_IMPROVEMENT_EPOCH",
                                           verbose=0,
                                           )
        self.assertAlmostEqual(results[1][1], results[2][1])
    
    def test_opt_decay6(self):
        self.build_compile()
        oparch.__reset_random__()
        hist = self.model.fit(self.X,self.y,
                       epochs=5,
                       batch_size=4,
                       verbose=0,
                       )
        results = oparch.opt_decay(self.model, self.X, self.y,
                                           return_model=False,
                                           decays=[0.1,0.2],
                                           epochs=5,
                                           batch_size=4,
                                           )
        self.assertAlmostEqual(results[1][1], round(hist.history["loss"][-1],5))#TODO: Model fit gives slightly different results
        
    def test_opt_decay7(self):
        self.build_compile()
        results = oparch.opt_decay(self.model, self.X, self.y,
                                           return_model=False,
                                           epochs=5,
                                           batch_size=4,
                                           )
        hist = self.model.fit(self.X,self.y,
                       epochs=5,
                       batch_size=4,
                       verbose=0,
                       )
        results2 = oparch.opt_decay(self.model, self.X, self.y,
                                           return_model=False,
                                           epochs=5,
                                           batch_size=4,
                                           )
        self.assertEqual(str(results), str(results2))
        
    def test_opt_decay8(self):
        self.build_compile()
        model = oparch.opt_decay(self.model, self.X, np.exp(self.y),return_metric="nn")
        results = oparch.opt_decay(self.model, self.X, np.exp(self.y),return_model=False)
        results.pop(0)
        best = 1000
        index = 0
        for i,result in enumerate(results):
            if result[1] < best:
                best = result[1]
                index = i
        print(f"Test act 4:\n {results}")
        print(results[index][0])
        print(model.layers[3].get_config().get("units"))
        self.assertEqual(results[index][0], model.optimizer.get_config().get("decay"))
        
    
if __name__ == "__main__":
    unittest.main()