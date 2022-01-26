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
        
    def test_get_dense_indices(self):
        corr = [0,2,3]
        empty = []
        self.assertEqual(utils.get_dense_indices(self.model), corr)
        self.assertEqual(utils.get_dense_indices(tf.keras.models.Sequential([tf.keras.layers.AbstractRNNCell()])), empty)
        
    def test_get_copy_of_layers(self):
        configs = [layer.get_config() for layer in self.layers]
        configs = utils.add_seed_configs(configs)
        new_layers = utils.get_copy_of_layers(self.layers)
        new_configs = [layer.get_config() for layer in new_layers]
        self.assertTrue(all([old == new for old,new in zip(configs,new_configs)]))
    
    def test_create_dict(self):
        self.maxDiff = None
        self.layers = [tf.keras.layers.Dense(6,activation="sigmoid"),tf.keras.layers.Dropout(0.1),tf.keras.layers.Dense(1),tf.keras.layers.Dense(1)]
        self.model = tf.keras.models.Sequential(self.layers)
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
        
    def test_create_dict_raises(self):
        with self.assertRaises(AttributeError):
            utils.create_dict(self.model)
        
    def test_print_model(self):
        self.build_compile()
        with open(os.devnull,"w") as sys.stdout:
            utils.print_model(self.model)
        sys.stdout = sys.__stdout__
        
    def test_check_compilation_raises1(self):
        with self.assertRaises(KeyError):
            utils.check_compilation(self.model, self.X) #Not compiled and no optimizer and loss specified
            
    def test_check_compilation_raises2(self):
        with self.assertRaises(KeyError):
            utils.check_compilation(self.model, self.X,optimizer=tf.keras.optimizers.RMSprop(0.01))#Only optimizer passed as kwarg
        
    def test_check_compilation_works1(self):
        kwargs = {}
        kwargs["loss"] = tf.keras.losses.MeanSquaredError()
        kwargs["optimizer"] = tf.keras.optimizers.RMSprop(0.01)
        model = utils.check_compilation(self.model, self.X, **kwargs)#loss and optimizer passed in a dict
        self.assertEqual(id(model), id(self.model))
        
    def test_check_compilation_works2(self):
        kwarg_dict = {}
        kwarg_dict["loss"] = tf.keras.losses.MeanSquaredError()
        kwarg_dict["optimizer"] = tf.keras.optimizers.RMSprop(0.01)
        model = utils.check_compilation(self.model, self.X,optimizer=kwarg_dict["optimizer"],loss=kwarg_dict["loss"])
        self.assertEqual(id(model), id(self.model)) #If optimizer and loss are provided through arguments then memory is same
        
    def test_check_compilation_works3(self):
        self.build_compile()
        model_weights = self.model.get_weights()
        model = utils.check_compilation(self.model, self.X)#Model compiled before calling
        self.assertEqual(id(model), id(self.model)) #TODO: the memory adresses same
        
    def test_check_compilation4(self):
        self.layers = utils.get_copy_of_layers(self.layers)
        self.model = tf.keras.models.Sequential(self.layers)
        self.build_compile()
        self_weights = self.model.get_weights()
        model = utils.check_compilation(self.model, self.X)
        self.assertEqual(str(self_weights), str(model.get_weights())) #The models weights are the same
    
    def test_randomness(self):
        oparch.__reset_random__()
        random1 = tf.random.uniform((6,6))
        oparch.__reset_random__()
        random2 = tf.random.uniform((6,6))
        self.assertEqual(str(random1), str(random2))
        
    def test_test_learning_speed1(self):
        test_values = [(1,10,0.2,"LAST_LOSS",1,0.0433),#Normal run with kwargs
                       (2,10,0.2,"RELATIVE_IMPROVEMENT_EPOCH",1,0.01851),#Run with different loss metric
                       (1,10,1,"VALIDATION_LOSS",1,0.00806),#Run with validation_split = 1
                       (1,100,0.2,"LAST_LOSS",1,0.0433),#run with more samples than data
                       (1,0,0.2,"LAST_LOSS",1,0.0433),#samples = 0, uses lists and appending in LossCallback
                       (2,10,0.2,"RELATIVE_IMPROVEMENT_EPOCH",16,0.01851)
                       ]
        self.build_compile() #build and compile self.model
        for values in test_values:
            epochs,samples,validation_split,return_metric,batch_size,expected = values
            m = utils.test_learning_speed(self.model,
                                  self.X, self.y,
                                  epochs = epochs,
                                  samples = samples,
                                  validation_split=validation_split,
                                  batch_size = batch_size,
                                  return_metric=return_metric
                                  )
            self.assertAlmostEqual(m, expected)
            
    def test_test_learning_speed2(self):
        self.build_compile()
        model_weights = self.model.get_weights()
        utils.test_learning_speed(self.model, self.X, self.y)
        self.assertEqual(str(model_weights), str(self.model.get_weights()))#Model weights remain same when testing the model
        self.assertFalse(self.model.optimizer.get_weights())#optimizer doesn't have weights
        
    def test_test_learning_speed3(self):
        with self.assertRaises(AttributeError):
            utils.test_learning_speed(self.model, self.X, self.y)
    
    def test_learning_speed4(self):
        x,y = testing_utils.get_xy(samples=200,features=5)
        layers = utils.get_copy_of_layers(self.model.layers)
        model = tf.keras.models.Sequential(layers)
        model.build(np.shape(x))
        model.compile(optimizer=tf.keras.optimizers.SGD(),loss=tf.keras.losses.MeanSquaredError())
        oparch.__reset_random__()
        hist = model.fit(
            x,y,
            epochs=5,
            batch_size=45,
            verbose = 0,
        )
        met = utils.test_learning_speed(model, x, y, batch_size=45,verbose=0,epochs=5,samples=200)
        self.assertAlmostEqual(round(hist.history["loss"][-1],5), met) #TODO: some randomness still in the results
        
    def test_test_learning_speed5(self):
        x,y = testing_utils.get_xy(samples=169,features=5)
        layers = utils.get_copy_of_layers(self.model.layers)
        model = tf.keras.models.Sequential(layers)
        model.build(np.shape(x))
        model.compile(optimizer=tf.keras.optimizers.SGD(),loss=tf.keras.losses.MeanSquaredError())
        oparch.__reset_random__()
        hist = model.fit(
            x,y,
            epochs=5,
            batch_size=45,
            verbose = 1
        )
        model.compile(optimizer=tf.keras.optimizers.RMSprop(),loss=tf.keras.losses.MeanSquaredError())
        met = utils.test_learning_speed(model, x, y, batch_size=45,verbose=2,epochs=5,samples=200)
        self.assertEqual(round(hist.history["loss"][-1],5), met) #TODO: if optimizer is changed after training and model is then tested
            
        

if __name__ == "__main__":
    unittest.main()