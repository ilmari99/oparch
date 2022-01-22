import numpy as np
import tensorflow as tf
from . import optimize_utils as utils
from . import LossCallback as lcb
from . import configurations
import copy
_default_learning_rates = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
_default_nodes = [2**i for i in range(0,8)] #Node amounts to test


def opt_learning_rate(model: tf.keras.models.Sequential, X, y,**kwargs) -> (float, float):
    model = utils.check_compilation(model, X, kwargs)
    learning_rates = kwargs.get("learning_rates",_default_learning_rates)
    if not (isinstance(learning_rates, list) or isinstance(learning_rates, np.ndarray)):
        print("Invalid learning_rates in opt_learning_rate: {learning_rates}. Expected list or numpy array.\n"+
              "Continuing execution with default learning rates {_default_learning_rates}")
        learning_rates = _default_learning_rates
    optimizer_type = model.optimizer.__class__
    optimizer_config = model.optimizer.get_config()
    best_lr = optimizer_config["learning_rate"]
    best_metric = utils.test_learning_speed(model,X,y)
    #print(f"Default learning rate: {best_lr}, {configurations.LEARNING_METRIC}:{best_metric}")
    for lr in learning_rates:
        model.build(np.shape(X))
        optimizer_config["learning_rate"] = lr
        model.compile(optimizer=optimizer_type.from_config(optimizer_config),loss=model.loss)
        metric = utils.test_learning_speed(model,X,y)
        print(f"Learning rate: {lr}, {configurations.LEARNING_METRIC}:{metric}")
        if(metric<best_metric):
            best_metric = metric
            best_lr = lr
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return (best_lr, best_metric)
    optimizer_config["learning_rate"] = best_lr
    model.compile(optimizer=optimizer_type.from_config(optimizer_config),loss=model.loss)
    return (model, best_metric)

def opt_loss_fun(model: tf.keras.models.Sequential,X,y,**kwargs):
    #TODO: when this changes the loss function to logarithmic loss, sometimes the results are very weird
    model = utils.check_compilation(model, X, kwargs)
    metric_type = "RELATIVE_IMPROVEMENT_EPOCH" #Use this, because losses are not necessarily comparable
    best_loss_fun = model.loss
    best_metric = utils.test_learning_speed(model,X,y,return_metric=metric_type)
    optimizer_config = model.optimizer.get_config()
    optimizer_type = model.optimizer.__class__
    if(not all(isinstance(yi,int) for yi in y)): #TODO Tämän ehdon pitäisi tarkistaa, onko y categorinen vai ei
        loss_function_dict = configurations.REGRESSION_LOSS_FUNCTIONS
    for loss_fun in loss_function_dict.values():
        model.compile(optimizer=optimizer_type.from_config(optimizer_config), loss=loss_fun)
        metric = utils.test_learning_speed(model, X, y, return_metric=metric_type)
        print(type(loss_fun).__name__, {metric_type},{metric})
        if(metric<best_metric):
            best_loss_fun = loss_fun
            best_metric = metric
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return (best_loss_fun,best_metric)
    model.compile(optimizer=optimizer_type.from_config(optimizer_config),loss=best_loss_fun)
    return (model, best_metric)

def opt_activation(model: tf.keras.models.Sequential, index, X, y, **kwargs) -> dict:
    print(f"Optimizing activation function at index {index}")
    model = utils.check_compilation(model, X, kwargs)
    if not isinstance(model.layers[index],tf.keras.layers.Dense):
        return None
    layers = model.layers
    index_layer_configuration = layers[index].get_config()
    best_configuration = layers[index].get_config()
    best_metric = float("inf")
    optimizer_config = model.optimizer.get_config()
    for activation in configurations.ACTIVATION_FUNCTIONS.keys():
        index_layer_configuration["activation"] = activation # activation is the string identifier of the activation function
        layers[index] = tf.keras.layers.Dense.from_config(index_layer_configuration)
        #Create and compile a model with the new activation function
        #The layers weights will still be the same as the models layers, so should be random
        test_model = tf.keras.models.Sequential(layers)
        test_model.build(np.shape(X))
        test_model.compile(optimizer=model.optimizer.__class__.from_config(optimizer_config),
                  loss=model.loss)
        
        metric = utils.test_learning_speed(test_model, X, y)
        print(activation,metric)
        if metric < best_metric:
            best_metric = metric
            best_configuration = index_layer_configuration
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return (best_configuration, best_metric)
    layers[index] = tf.keras.layers.Dense.from_config(best_configuration)
    test_model = tf.keras.models.Sequential(layers)
    test_model.build(np.shape(X))
    test_model.compile(optimizer=model.optimizer.__class__.from_config(optimizer_config),
                  loss=model.loss)
    return (test_model,best_metric)

def opt_dense_units(model: tf.keras.models.Sequential, index, X, y, **kwargs):
    if not isinstance(model.layers[index],tf.keras.layers.Dense):
        print(f"layer {model.layers[index]} is not a Dense layer.")
        return None
    print(f"Optimizing dense units at index {index}")
    test_nodes = kwargs.get("test_nodes",_default_nodes)
    if not (isinstance(test_nodes, list) or isinstance(test_nodes, np.ndarray)):
        print("Invalid learning_rates in opt_dense_units: {test_nodes}. Expected list or numpy array.\n"+
              "Continuing execution with default test_nodes {_default_nodes}")
        test_nodes = _default_nodes
    model = utils.check_compilation(model, X, kwargs)
    layer_configs = [layer.get_config() for layer in model.layers]
    layers = [layer.__class__.from_config(config) for layer,config in zip(model.layers, layer_configs)]
    optimizer_type = model.optimizer.__class__
    optimizer_config = model.optimizer.get_config()
    #Test with no layer at index
    best_dense = None
    curr_layer = layers.pop(index) 
    test_model = tf.keras.models.Sequential(layers)
    test_model.build(np.shape(X))
    test_model.compile(
        optimizer=optimizer_type.from_config(optimizer_config),
        loss=model.loss
    )
    best_metric = utils.test_learning_speed(test_model,X,y)
    print(None, best_metric)
    best_configuration = None
    index_layer_configuration = layer_configs[index]
    for node_amount in test_nodes:
        index_layer_configuration["units"] = node_amount
        layer_configs[index] = index_layer_configuration
        layers = [layer.__class__.from_config(config) for layer,config in zip(model.layers, layer_configs)]
        test_model = tf.keras.models.Sequential(layers)
        test_model.build(np.shape(X))
        test_model.compile(
            optimizer=optimizer_type.from_config(optimizer_config),
            loss=model.loss
        )
        metric = utils.test_learning_speed(test_model, X, y)
        print(node_amount, metric)
        if metric<best_metric:
            best_metric = metric
            best_configuration = layers[index].get_config()
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return (best_configuration, best_metric)
    if best_configuration == None:
        layer_configs.pop(index)
        model.layers.pop(index)
    else:
        layer_configs[index] = best_configuration
    layers = [layer.__class__.from_config(config) for layer,config in zip(model.layers, layer_configs)]
    new_layers = utils.get_copy_of_layers(layers)
    test_model = tf.keras.models.Sequential(layers)
    test_model.build(np.shape(X))
    test_model.compile(optimizer=optimizer_type.from_config(optimizer_config),
        loss=model.loss)
    return (test_model, best_metric)

def opt_decay(model: tf.keras.models.Sequential,X,y,**kwargs):
    model = utils.check_compilation(model, X, kwargs)
    default_decays = [0.95, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0]
    decays = kwargs.get("decays",default_decays)
    if not (isinstance(decays, list) or isinstance(decays, np.ndarray) or 1 in decays):
        print("Invalid decay in opt_decay: {decays}. Expected list or numpy array.\n"+
              "Continuing execution with default decays {default_decays}")
        decays = default_decays
    optimizer_type = model.optimizer.__class__
    optimizer_config = model.optimizer.get_config()
    best_decay = optimizer_config["decay"]
    best_metric = utils.test_learning_speed(model,X,y)
    for decay in decays:
        optimizer_config["decay"] = decay
        model.build(np.shape(X))
        model.compile(optimizer=optimizer_type.from_config(optimizer_config),loss=model.loss)
        metric = utils.test_learning_speed(model,X,y)
        print(f"decay: {decay}, {configurations.LEARNING_METRIC}:{metric}")
        if(metric<best_metric):
            best_metric = metric
            best_decay = decay
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return (best_decay, best_metric)
    optimizer_config["decay"] = best_decay
    model.build(np.shape(X))
    model.compile(optimizer=optimizer_type.from_config(optimizer_config),loss=model.loss)
    return (model,best_metric)