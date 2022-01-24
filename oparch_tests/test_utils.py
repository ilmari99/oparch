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
import tensorflow.keras as tf
import numpy as np

class Test_optimize_utils(unittest.TestCase):
    def setUp(self):
        self.X, self.y = testing_utils.get_xy(samples=10,features=3)
        self.layers = [tf.layers.Dense(6,activation="sigmoid"),tf.layers.Dropout(0.1),tf.layers.Dense(1),tf.layers.Dense(1)]
        self.model = tf.models.Sequential(self.layers)
        
    def build_compile(self):
        self.model.build(np.shape(self.X))
        self.model.compile(optimizer = tf.optimizers.RMSprop(learning_rate=0.01), loss=tf.losses.MeanSquaredError())
        
    def test_get_dense_indices(self):
        corr = [0,2,3]
        empty = []
        self.assertEqual(utils.get_dense_indices(self.model), corr)
        self.assertEqual(utils.get_dense_indices(tf.models.Sequential([tf.layers.AbstractRNNCell()])), empty)
        
    def test_get_copy_of_layers(self):
        configs = [layer.get_config() for layer in self.layers]
        new_layers = utils.get_copy_of_layers(self.layers)
        new_configs = [layer.get_config() for layer in new_layers]
        self.assertTrue(all([old == new for old,new in zip(configs,new_configs)]))
    
    def test_create_dict(self):
        self.maxDiff = None
        self.layers = [tf.layers.Dense(6,activation="sigmoid"),tf.layers.Dropout(0.1),tf.layers.Dense(1),tf.layers.Dense(1)]
        self.model = tf.models.Sequential(self.layers)
        self.build_compile()
        correct_dict = {'optimizer': {'name': 'RMSprop', 'learning_rate': 0.01, 'decay': 0.0, 'rho': 0.9, 'momentum': 0.0, 'epsilon': 1e-07, 'centered': False},
                        'loss_function': 'MeanSquaredError',
                        'layers': {'dense': {'units': 6, 'activation': 'sigmoid'},
                                   'dropout': {'rate': 0.1},
                                   'dense_1': {'units': 1, 'activation': 'linear'},
                                   'dense_2': {'units': 1, 'activation': 'linear'}
                                   },
                        'learning_metrics': {"LAST_LOSS":0.03567}
                        }
        new_dict = utils.create_dict(self.model,learning_metrics={"LAST_LOSS":0.03567})
        new_keys = ["dense","dropout","dense_1","dense_2"]
        keys = new_dict["layers"].copy()
        for i,name in enumerate(keys): #Change the layer names to the 'correct' ones because they are not interesting to compare
            new_dict["layers"][new_keys[i]] = new_dict["layers"].pop(name)
        self.assertEqual(correct_dict, new_dict)
        self.setUp()
        with self.assertRaises(AttributeError):
            utils.create_dict(self.model)
        
    def test_print_model(self):
        self.build_compile()
        sys.stdout = open(os.devnull,"w")
        utils.print_model(self.model)
        sys.stdout.close()
        sys.stdout = sys.__stdout__
        
    def test_check_compilation(self):
        kwarg_dict = {}
        with self.assertRaises(KeyError):
            utils.check_compilation(self.model, self.X, kwarg_dict) #Not compiled and no optimizer and loss specified
            utils.check_compilation(model, X,kwarg_dict,optimizer=tf.optimizers.RMSprop(0.01))#Only optimizer passed as kwarg
        kwarg_dict["loss"] = tf.losses.MeanSquaredError()
        kwarg_dict["optimizer"] = tf.optimizers.RMSprop(0.01)
        utils.check_compilation(self.model, self.X, kwarg_dict)#loss and optimizer passed as a dict
        utils.check_compilation(self.model, self.X,{},optimizer=kwarg_dict["optimizer"],loss=kwarg_dict["loss"])#using kwargs
        self.build_compile()
        utils.check_compilation(self.model, self.X)#Model compiled before calling
        
    def test_test_learning_speed(self):
        def do_test():
            return utils.test_learning_speed(self.model,
                                  self.X, self.y,
                                  epochs = epochs,
                                  samples = samples,
                                  validation_split=validation_split,
                                  batch_size = batch_size,
                                  return_metric=return_metric
                                  )
        test_values = [(1,10,0.2,"LAST_LOSS",1,0.4371974468231201),
                       (2,10,0.2,"RELATIVE_IMPROVEMENT_EPOCH",1,-0.8865967464562842),
                       (1,10,1,"VALIDATION_LOSS",1,0.0004439638869371265),
                       (1,100,0.2,"LAST_LOSS",1,0.4371974468231201),
                       (1,0,0.2,"LAST_LOSS",1,0.4304800033569336),
                       ]
        self.build_compile() #build and compile self.model
        for values in test_values:
            epochs,samples,validation_split,return_metric,batch_size,expected = values
            m = do_test()
            self.assertAlmostEqual(m, expected)
        
            
        
    

if __name__ == "__main__":
    unittest.main()