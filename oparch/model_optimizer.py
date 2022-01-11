import numpy as np
import tensorflow as tf
from . import model_optimizer_tools as mot
from . import LossCallback as lcb
from . import configurations
import copy
from .OptimizedModel import OptimizedModel

def get_dense_layer(args: list):
    """args = list = [number of neurons(int), activation function]"""
    return tf.keras.layers.Dense(args[0],activation=args[1])


def get_optimized_model(x_train: np.ndarray, y_train: np.ndarray, layer_list: list) -> OptimizedModel:
    """
    Tests different configurations (optimizer, learning rate, loss function, nodes/layer, activation functions)
    for a sequential, dense neural network. Returns an OptimizedModel instance.

    Args:
        x_train (np.ndarray): Feature data
        y_train (np.ndarray): Observed y data 
        layer_list (list): list with Layer objects

    Returns:
        OptimizedModel: [description]
    """
    optimized_model = OptimizedModel(layer_list,x_train)
    #TODO: how do these link to the last test run before returning
    #OptimizedModel.optimize_loss_fun(optimized_model.model,x_train,y_train)
    #OptimizedModel.optimize_optimizer(optimized_model.model,x_train,y_train)
    #OptimizedModel.optimize_learning_rate(optimized_model.model,x_train,y_train)
    best_metric = mot.test_learning_speed(optimized_model.model,x_train,y_train)
    print(f"{configurations.LEARNING_METRIC} with default layers was {best_metric}")
    layer_configs = [layer.get_config() for layer in layer_list]
    layer_list = [tf.keras.layers.Dense.from_config(config) for config in layer_configs]
    for index, layer in enumerate(layer_list[:-1]): #No optimization for last layer
        #index = len(layer_list[:-1]) - index - 1
        opt_dense, opt_metric = get_optimized_dense(index, layer_list, x_train, y_train)
        if(opt_metric<best_metric):
            print(f"Model structure changed.\nSubstituted layer at index {index}:")
            print(f"from: {layer_list[index].get_config()}\n {configurations.LEARNING_METRIC}: {best_metric}")
            if(opt_dense != None):
                print(f"To: {opt_dense.get_config()}\n {configurations.LEARNING_METRIC}: {opt_metric}")
                layer_list[index] = opt_dense
            else:
                print(f"To: None\n {configurations.LEARNING_METRIC}:{opt_metric}")
                layer_list.pop(index)
            optimized_model.loss = opt_metric
            best_metric = opt_metric
        optimized_configs = [layer.get_config() for layer in layer_list]
        optimized_model.set_layers_from_config(optimized_configs)
        #TODO Looks like there is a problem in these optimize functions
        #optimized_model.model.save_weights("saved_weights.h5")
        #OptimizedModel.optimize_loss_fun(optimized_model.model,x_train,y_train)
        #OptimizedModel.optimize_optimizer(optimized_model.model,x_train,y_train)
        #OptimizedModel.optimize_learning_rate(optimized_model.model,x_train,y_train)
        #optimized_model.model.load_weights("saved_weights.h5")
        metric = mot.test_learning_speed(optimized_model.model, x_train, y_train)
        print(f"{configurations.LEARNING_METRIC} after optimization: {metric}")
    return optimized_model

def get_optimized_dense(index, layers, x_train, y_train):
    print(f"Testing for a new dense layer at index {index}...")
    nodes = [2**i for i in range(0,6)] #Node amounts to test
    activations = list(configurations.ACTIVATION_FUNCTIONS.values())
    configs = [layer.get_config() for layer in layers]
    print(f"Current layer at index {index}: units{configs[index]['units']} Activation:{configs[index]['activation']}")
    best_dense = None
    #Test with no layer at index
    layer_list = [tf.keras.layers.Dense.from_config(config) for config in configs]
    layer_list.pop(index)
    model = tf.keras.models.Sequential(layer_list)
    best_metric = mot.test_learning_speed(model,x_train,y_train)
    print(f"{configurations.LEARNING_METRIC} with no layer at index {index}: {best_metric}")
    
    for activation in activations:
        for node_amount in nodes:
            #Currently the first metric
            layer_list = [tf.keras.layers.Dense.from_config(config) for config in configs]
            dense_args = [node_amount,activation] #TODO Create a new dense layer by modifying the configs
            
            layer = get_dense_layer(dense_args)
            layer_list[index] = layer
            model = tf.keras.models.Sequential(layer_list)
            #TODO: This is not worth the time. Using these seems to get a worse answer, for some reason.
            #OptimizedModel.optimize_loss_fun(model,x_train,y_train)
            #OptimizedModel.optimize_optimizer(model,x_train,y_train)
            #OptimizedModel.optimize_learning_rate(model,x_train,y_train)
            print(f"Nodes: {dense_args[0]}\nActivation: {dense_args[1]}.......")
            metric = mot.test_learning_speed(model, x_train, y_train, samples=800)
            print(f"{configurations.LEARNING_METRIC}: {metric}")
            if metric<best_metric:
                best_metric = metric
                best_dense = get_dense_layer(dense_args)
                print(f"This is currently the lowest {configurations.LEARNING_METRIC}.")
            print("\n\n")
    if(best_dense != None):
        best_config = best_dense.get_config()
        print(f"Best layer at index {index}: units{best_config['units']} Activation:{best_config['activation']}\n")
    return best_dense, best_metric