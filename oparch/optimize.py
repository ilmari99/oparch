import numpy as np
import tensorflow as tf
from . import optimize_utils as utils
from . import LossCallback as lcb
from . import configurations
import oparch
import copy
_default_learning_rates = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
_default_nodes = [2**i for i in range(0,8)] #Node amounts to test
_default_decays = [0.95, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0]


def opt_learning_rate(model: tf.keras.models.Sequential, X, y,**kwargs) -> (float, float):
    model = utils.check_compilation(model, X, **kwargs)
    learning_rates = kwargs.pop("learning_rates",_default_learning_rates)
    if not (isinstance(learning_rates, list) or isinstance(learning_rates, np.ndarray)):
        print("Invalid learning_rates in opt_learning_rate: {learning_rates}. Expected list or numpy array.\n"+
              "Continuing execution with default learning rates {_default_learning_rates}")
        learning_rates = _default_learning_rates
        
    return_metric = kwargs.get("return_metric",configurations.LEARNING_METRIC)
    
    results = list(range(len(learning_rates)+2))
    results[0] = ["learning_rate",return_metric]
    
    optimizer_type = model.optimizer.__class__
    optimizer_config = model.optimizer.get_config()
    best_lr = optimizer_config.get("learning_rate") #prints correctly for example 0.01
    best_metric = utils.test_learning_speed(model,X,y,**kwargs)
    print(f"Learning rate: {best_lr}, {return_metric}:{best_metric}") #TODO: prints: 0.010000000149011612
    
    results[1] = [best_lr,best_metric]
    
    for i,lr in enumerate(learning_rates):
        model.build(np.shape(X))
        optimizer_config["learning_rate"] = lr
        model.compile(optimizer=optimizer_type.from_config(optimizer_config),loss=model.loss)
        metric = utils.test_learning_speed(model,X,y,**kwargs)
        print(f"Learning rate: {lr}, {return_metric}:{metric}")
        
        results[i+2] = [lr,metric]
        
        if(metric<best_metric):
            best_metric = metric
            best_lr = lr
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return results
    optimizer_config["learning_rate"] = best_lr
    model.compile(optimizer=optimizer_type.from_config(optimizer_config),loss=model.loss)
    return model

def opt_loss_fun(model: tf.keras.models.Sequential,X,y,**kwargs):
    #TODO: when this changes the loss function to logarithmic loss, sometimes the results are very weird
    model = utils.check_compilation(model, X, **kwargs)
    if not "return_metric" in kwargs:
        kwargs["return_metric"] = "RELATIVE_IMPROVEMENT_EPOCH"
    return_metric = kwargs.get("return_metric")
    results = list(range(len(configurations.REGRESSION_LOSS_FUNCTIONS)+2))
    results[0] = ["loss function",return_metric]
    best_loss_fun = model.loss
    best_metric = utils.test_learning_speed(model,X,y,**kwargs)
    optimizer_config = model.optimizer.get_config()
    optimizer_type = model.optimizer.__class__
    results[1] = [type(best_loss_fun).__name__, best_metric]
    print(type(best_loss_fun).__name__, return_metric,best_metric)
    if(not all(isinstance(yi,int) for yi in y)): #TODO Tämän ehdon pitäisi tarkistaa, onko y categorinen vai ei
        loss_function_dict = configurations.REGRESSION_LOSS_FUNCTIONS
    for i, loss_fun in enumerate(loss_function_dict.values()):
        oparch.__reset_random__()
        model.compile(optimizer=optimizer_type.from_config(optimizer_config), loss=loss_fun)
        metric = utils.test_learning_speed(model, X, y, **kwargs)
        results[i+2] = [type(loss_fun).__name__,metric]
        print(type(loss_fun).__name__, return_metric,metric)
        if(metric<best_metric):
            best_loss_fun = loss_fun
            best_metric = metric
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return results
    model.compile(optimizer=optimizer_type.from_config(optimizer_config),loss=best_loss_fun)
    return model

def opt_activation(model: tf.keras.models.Sequential, index, X, y, **kwargs) -> dict:
    print(f"Optimizing activation function at index {index}")
    model = utils.check_compilation(model, X, **kwargs)
    if not isinstance(model.layers[index],tf.keras.layers.Dense):
        return None
    layers = model.layers
    return_metric = kwargs.get("return_metric",configurations.LEARNING_METRIC)
    results = list(range(len(configurations.ACTIVATION_FUNCTIONS.keys())+2))
    results[0] = ["activation",return_metric]
    index_layer_configuration = layers[index].get_config()
    best_configuration = layers[index].get_config()
    best_metric = utils.test_learning_speed(model, X, y)
    optimizer_config = model.optimizer.get_config()
    activation = index_layer_configuration.get("activation")
    print(activation,best_metric)
    results[1] = [activation,best_metric]
    for i, activation in enumerate(configurations.ACTIVATION_FUNCTIONS.keys()):
        index_layer_configuration["activation"] = activation # activation is the string identifier of the activation function
        layers[index] = tf.keras.layers.Dense.from_config(index_layer_configuration)
        #Create and compile a model with the new activation function
        #The layers weights will still be the same as the models layers, so should be random
        test_model = tf.keras.models.Sequential(layers)
        test_model.build(np.shape(X))
        test_model.compile(optimizer=model.optimizer.__class__.from_config(optimizer_config),
                  loss=model.loss)
        metric = utils.test_learning_speed(test_model, X, y,**kwargs)
        print(activation,metric)
        results[i+2] = [activation,metric]
        if metric < best_metric:
            best_metric = metric
            best_configuration = index_layer_configuration
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return results
    layers[index] = tf.keras.layers.Dense.from_config(best_configuration)
    test_model = tf.keras.models.Sequential(layers)
    test_model.build(np.shape(X))
    test_model.compile(optimizer=model.optimizer.__class__.from_config(optimizer_config),
                  loss=model.loss)
    return test_model

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
    return_metric = kwargs.get("return_metric",configurations.LEARNING_METRIC)
    model = utils.check_compilation(model, X, **kwargs)
    layer_configs = [layer.get_config() for layer in model.layers]
    layers = [layer.__class__.from_config(config) for layer,config in zip(model.layers, layer_configs)]
    optimizer_type = model.optimizer.__class__
    optimizer_config = model.optimizer.get_config()
    results = list(range(len(test_nodes)+3))
    results[0] = ["nodes",return_metric]
    index_layer_configuration = layer_configs[index]
    node_amount = index_layer_configuration.get("units")
    best_metric = utils.test_learning_speed(model, X, y)
    best_configuration = layer_configs[index]
    print(node_amount, best_metric)
    results[1] = [node_amount,best_metric]
    #Test with no layer at index
    curr_layer = layers.pop(index) 
    test_model = tf.keras.models.Sequential(layers)
    test_model.build(np.shape(X))
    test_model.compile(
        optimizer=optimizer_type.from_config(optimizer_config),
        loss=model.loss
    )
    metric = utils.test_learning_speed(test_model,X,y)
    print(None, metric)
    results[2] = ["None",metric]
    if metric<best_metric:
        best_metric = metric
        best_configuration = None
    for i,node_amount in enumerate(test_nodes):
        index_layer_configuration["units"] = node_amount
        layer_configs[index] = index_layer_configuration
        layers = [layer.__class__.from_config(config) for layer,config in zip(model.layers, layer_configs)]
        test_model = tf.keras.models.Sequential(layers)
        test_model.build(np.shape(X))
        test_model.compile(
            optimizer=optimizer_type.from_config(optimizer_config),
            loss=model.loss
        )
        metric = utils.test_learning_speed(test_model, X, y,**kwargs)
        print(node_amount, metric)
        results[i +3] = [node_amount,metric]
        if metric<best_metric:
            best_metric = metric
            best_configuration = layers[index].get_config()
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return results
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
    return test_model

def opt_decay(model: tf.keras.models.Sequential,X,y,**kwargs):
    model = utils.check_compilation(model, X, **kwargs)
    decays = kwargs.get("decays",_default_decays)
    if not (isinstance(decays, list) or isinstance(decays, np.ndarray) or 1 in decays):
        print("Invalid decay in opt_decay: {decays}. Expected list or numpy array.\n"+
              "Continuing execution with default decays {_default_decays}")
        decays = default_decays
    return_metric = kwargs.get("return_metric",configurations.LEARNING_METRIC)
    optimizer_type = model.optimizer.__class__
    optimizer_config = model.optimizer.get_config()
    best_decay = optimizer_config["decay"]
    best_metric = utils.test_learning_speed(model,X,y)
    results = list(range(len(decays)+2))
    results[0] = ["decay",return_metric]
    results[1] = [best_decay,best_metric]
    print(f"decay: {best_decay}, {return_metric}:{best_metric}")
    for i,decay in enumerate(decays):
        oparch.__reset_random__()
        optimizer_config["decay"] = decay
        model.build(np.shape(X))
        model.compile(optimizer=optimizer_type.from_config(optimizer_config),loss=model.loss)
        metric = utils.test_learning_speed(model,X,y,**kwargs)
        print(f"decay: {decay}, {return_metric}:{metric}")
        results[i+2] = [decay,metric]
        if(metric<best_metric):
            best_metric = metric
            best_decay = decay
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return results
    optimizer_config["decay"] = best_decay
    model.build(np.shape(X))
    model.compile(optimizer=optimizer_type.from_config(optimizer_config),loss=model.loss)
    return model