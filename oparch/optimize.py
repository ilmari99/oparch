import numpy as np
import tensorflow as tf
from . import model_optimizer_tools as mot
from . import LossCallback as lcb
from . import configurations
import copy
from .OptimizedModel import OptimizedModel

def check_compilation(model: tf.keras.models.Sequential, X, kwarg_dict, **kwargs):
    if not hasattr(model, "optimizer"): #if model is not compiled, compile it with optimizer and loss kwargs
        try:
            model.build(np.shape(X))
            model.compile(optimizer=kwarg_dict["optimizer"], loss=kwarg_dict["loss"])
        except KeyError:
            raise KeyError("If the model is not compiled, you must specify the optimizer and loss")
    #If the optimizer has weights, it has been used before and the measurements would not be accurate
    if model.optimizer.get_weights(): #If optimizer weights list is not empty
        raise Exception("The model has been trained before optimizing and cannot be accurately tested")
    return model



def opt_learning_rate(model: tf.keras.models.Sequential, X, y,**kwargs):
    model = check_compilation(model, X, kwargs)
    default_learning_rates = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
    learning_rates = kwargs.get("learning_rates",default_learning_rates)
    if not (isinstance(learning_rates, list) or isinstance(learning_rates, np.ndarray)):
        print("Invalid learning_rates in opt_learning_rate: {learning_rates}. Expected list or numpy array.\n"+
              "Continuing execution with default learning rates {default_learning_rates}")
        learning_rates = default_learning_rates
    optimizer_type = type(model.optimizer)
    best_lr = model.optimizer.learning_rate.numpy()
    best_metric = mot.test_learning_speed(model,X,y)
    #print(f"Default learning rate: {best_lr}, {configurations.LEARNING_METRIC}:{best_metric}")
    for lr in learning_rates:
        model.build(np.shape(X))
        model.compile(optimizer=optimizer_type(lr),loss=model.loss)
        metric = mot.test_learning_speed(model,X,y)
        print(f"Learning rate: {lr}, {configurations.LEARNING_METRIC}:{metric}")
        if(metric<best_metric):
            best_metric = metric
            best_lr = lr
    return (best_lr, best_metric)


def opt_loss_fun(model: tf.keras.models.Sequential,X,y,**kwargs):
    model = check_compilation(model, X, kwargs)
    metric_type = "RELATIVE_IMPROVEMENT_EPOCH"
    best_loss_fun = model.loss
    best_metric = mot.test_learning_speed(model,X,y,return_metric=metric_type)
    if(not all(isinstance(yi,int) for yi in y)): #TODO Tämän ehdon pitäisi tarkistaa, onko y categorinen vai ei
        loss_function_dict = configurations.REGRESSION_LOSS_FUNCTIONS
    for loss_fun in loss_function_dict.values():
        model.compile(optimizer=model.optimizer, loss=loss_fun)
        metric = mot.test_learning_speed(model, X, y, return_metric=metric_type)
        print(f"Loss function: {type(loss_fun).__name__}, {metric_type}:{metric}")
        if(metric<best_metric):
            best_loss_fun = loss_fun
            best_metric = metric
    #print(f"Optimized loss function: {type(cls.loss_fun).__name__}, {configurations.LEARNING_METRIC}:{base_metric}")
    return (best_loss_fun,best_metric)

def opt_activation(model: tf.keras.models.Sequential, index, X, y, **kwargs) -> dict:
    model = check_compilation(model, X, kwargs)
    if not isinstance(model.layers[index],tf.keras.layers.Dense):
        return None
    layers = model.layers
    index_layer_configuration = layers[index].get_config()
    best_configuration = None
    best_metric = float("inf")
    optimizer_config = model.optimizer.get_config()
    for activation in configurations.ACTIVATION_FUNCTIONS.keys():
        index_layer_configuration["activation"] = activation # activation is the string identifier of the activation function
        layers[index] = tf.keras.layers.Dense.from_config(index_layer_configuration)
        #Create and compile a model with the new activation function
        #The layers weights will still be the same as the models layers, so should be random
        test_model = tf.keras.models.Sequential(layers)
        test_model.build(np.shape(X))
        test_model.compile(optimizer=type(model.optimizer)(optimizer_config["learning_rate"]),
                  loss=model.loss)
        
        metric = mot.test_learning_speed(test_model, X, y)
        if metric < best_metric:
            best_metric = metric
            best_configuration = index_layer_configuration #TODO: check that this variable doesn't change when index layer conf changes
    return (best_configuration, best_metric)
        
        

def opt_dense_units(model: tf.keras.models.Sequential, index, X, y, **kwargs):
    model = check_compilation(model, X, kwargs)    
    #This should be the fast method, but TODO: make the layers have the previous layer as input.
    #layers = model.layers
    
    #Creating the layers from configs each time works, but it should be much slower.
    layer_configs = [layer.get_config() for layer in model.layers] 
    layers = [layer.__class__.from_config(config) for layer,config in zip(model.layers, layer_configs)]
    
    if not isinstance(model.layers[index],tf.keras.layers.Dense):
        return None
    
    nodes = [2**i for i in range(0,6)] #Node amounts to test
    
    #configs = [layer.get_config() for layer in layers]
    #print(f"Current layer at index {index}: units{configs[index]['units']} Activation:{configs[index]['activation']}")
    optimizer_config = model.optimizer.get_config()

    #Test with no layer at index
    best_dense = None
    curr_layer = layers.pop(index) 
    test_model = tf.keras.models.Sequential(layers)
    test_model.build(np.shape(X))
    test_model.compile(
        optimizer=type(model.optimizer)(optimizer_config["learning_rate"]),
        loss=model.loss
    )
    best_metric = mot.test_learning_speed(test_model,X,y)
    print(f"{configurations.LEARNING_METRIC} with no layer at index {index}: {best_metric}")
    best_configuration = None
    index_layer_configuration = layer_configs[index]
    for node_amount in nodes:
        index_layer_configuration["units"] = node_amount
        layer_configs[index] = index_layer_configuration
        layers = [layer.__class__.from_config(config) for layer,config in zip(model.layers, layer_configs)]
        #layers[index] = tf.keras.layers.Dense.from_config(index_layer_configuration)
        test_model = tf.keras.models.Sequential(layers)
        test_model.build(np.shape(X))
        test_model.compile(
            optimizer=type(model.optimizer)(optimizer_config["learning_rate"]),
            loss=model.loss
        )
        metric = mot.test_learning_speed(test_model, X, y)
        print(node_amount, metric)
        if metric<best_metric:
            best_metric = metric
            best_configuration = layers[index].get_config()
    return (best_configuration, best_metric)