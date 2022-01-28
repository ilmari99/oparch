import numpy as np
import tensorflow as tf
from . import optimize_utils as utils
from . import LossCallback as lcb
from . import configurations
import oparch
_default_learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]#[1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
_default_nodes = [2**i for i in range(0,8)] #Node amounts to test
_default_decays = sorted([0.95, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0])

def opt_all_units(model,X,y):
    indices = oparch.utils.get_dense_indices(model)
    for i,layer in enumerate(model.layers.copy()):
        if i in indices:
            model = opt_dense_units(model, i, X, y)
    return model

def opt_learning_rate(model: tf.keras.models.Sequential, X: np.ndarray, y: np.ndarray,**kwargs) -> tf.keras.models.Sequential or list:
    model = utils.check_compilation(model, X, **kwargs)
    learning_rates = kwargs.pop("learning_rates",_default_learning_rates)
    if not (isinstance(learning_rates, list)):
        print(f"Invalid learning_rates in opt_learning_rate: {learning_rates}. Expected list or numpy array.\n"+
              f"Continuing execution with default learning rates {_default_learning_rates}")
        learning_rates = _default_learning_rates.copy()
    optimizer_config = model.optimizer.get_config()
    optimizer_type = model.optimizer.__class__
    get_optimizer = lambda : optimizer_type.from_config(optimizer_config)
    learning_rates.append(optimizer_config["learning_rate"])
    learning_rates = set(learning_rates)
    learning_rates = sorted(learning_rates)
    return_metric = kwargs.get("return_metric",configurations.LEARNING_METRIC)
    results = list(range(len(learning_rates)))
    best_metric = float("inf")
    for i,lr in enumerate(learning_rates):
        optimizer_config["learning_rate"] = lr
        model.compile(optimizer=get_optimizer(),loss=model.loss)
        metric = utils.test_learning_speed(model,X,y,**kwargs)
        print(f"Learning rate: {lr}, {return_metric}:{metric}")
        results[i] = [lr,metric]
        if(metric<best_metric):
            best_metric = metric
            best_lr = lr
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return results
    optimizer_config["learning_rate"] = best_lr
    model.compile(optimizer=get_optimizer(),loss=model.loss)
    return model

def opt_loss_fun(model: tf.keras.models.Sequential,X,y,**kwargs):
    model = utils.check_compilation(model, X, **kwargs)
    if not "return_metric" in kwargs:
        kwargs["return_metric"] = "RELATIVE_IMPROVEMENT_EPOCH"
    if(not all(isinstance(yi,int) for yi in y)): #TODO Tämän ehdon pitäisi tarkistaa, onko y categorinen vai ei
        loss_functions = configurations.REGRESSION_LOSS_FUNCTIONS.values()
    else:
        loss_functions = configurations.LOSS_FUNCTIONS.values()
    return_metric = kwargs.get("return_metric")
    results = list(range(len(loss_functions)+2))
    results[0] = ["loss function",return_metric]
    #test with current loss function
    best_loss_fun = type(model.loss)
    best_metric = utils.test_learning_speed(model,X,y,**kwargs)
    print(best_loss_fun.__name__, return_metric,best_metric)
    results[1] = [best_loss_fun.__name__, best_metric]
    for i, loss_fun in enumerate(loss_functions):
        model.compile(optimizer=model.optimizer, loss=loss_fun)
        metric = utils.test_learning_speed(model, X, y, **kwargs)
        print(type(loss_fun).__name__, return_metric,metric)
        results[i+2] = [type(loss_fun).__name__,metric]
        if(metric<best_metric):
            best_loss_fun = loss_fun
            best_metric = metric
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return results
    model.compile(optimizer=model.optimizer,loss=best_loss_fun)
    return model

def opt_activation(model: tf.keras.models.Sequential, index, X, y, **kwargs) -> dict:
    model = utils.check_compilation(model, X, **kwargs)
    if index > len(model.layers)-1:
        raise IndexError()
    if model.layers[index].get_config().get("activation") is None:
        raise KeyError(f"Layer at index {index} doesn't have an activation function.")
    return_metric = kwargs.get("return_metric",configurations.LEARNING_METRIC)
    optimizer_config = model.optimizer.get_config()
    optimizer_type = model.optimizer.__class__
    get_optimizer = lambda : optimizer_type.from_config(optimizer_config)
    optimizer = get_optimizer()
    loss = model.loss
    index_layer_configuration = model.layers[index].get_config()
    layers = model.layers
    
    results = list(range(len(configurations.ACTIVATION_FUNCTIONS.keys())+2))
    results[0] = ["activation",return_metric]
    #Test current
    best_metric = utils.test_learning_speed(model, X, y,**kwargs)
    best_configuration = model.layers[index].get_config()
    activation = index_layer_configuration.get("activation")
    print(activation,best_metric)
    results[1] = [activation,best_metric]
    for i, activation in enumerate(configurations.ACTIVATION_FUNCTIONS.keys()):
        index_layer_configuration["activation"] = activation # activation is the string identifier of the activation function
        layers[index] = tf.keras.layers.Dense.from_config(index_layer_configuration)
        model = tf.keras.models.Sequential(layers)
        model.build(np.shape(X))
        model.compile(optimizer=get_optimizer(), loss=loss)
        metric = utils.test_learning_speed(model, X, y,**kwargs)
        print(activation,metric)
        results[i+2] = [activation,metric]
        if metric < best_metric:
            best_metric = metric
            best_configuration["activation"] = activation
    return_model = kwargs.get("return_model",True)
    if not return_model:
        print(f"best activation {best_configuration.get('activation')}")
        return results
    layers[index] = tf.keras.layers.Dense.from_config(best_configuration)
    model = tf.keras.models.Sequential(layers)
    model.build(np.shape(X))
    model.compile(optimizer=get_optimizer(),
                  loss=loss)
    print(f"best activation {best_configuration.get('activation')}")
    return model

def opt_dense_units(model: tf.keras.models.Sequential, index, X, y, **kwargs):
    model = utils.check_compilation(model, X, **kwargs)
    if not isinstance(model.layers[index],tf.keras.layers.Dense):
        raise KeyError(f"layer {model.layers[index]} is not a Dense layer.")
    print(f"Optimizing dense units at index {index}")
    test_nodes = kwargs.get("test_nodes",_default_nodes)
    if not (isinstance(test_nodes, list)):
        print("Invalid learning_rates in opt_dense_units: {test_nodes}. Expected list or numpy array.\n"+
              "Continuing execution with default test_nodes {_default_nodes}")
        test_nodes = _default_nodes.copy()
    test_nodes.append(None)
    test_nodes.append(model.layers[index].get_config().get("units"))
    test_nodes = set(test_nodes)
    test_nodes = sorted(test_nodes,key=lambda x : (x is None, x))
    layer_configs = [layer.get_config() for layer in model.layers]
    layers = utils.get_copy_of_layers(model.layers)
    loss = model.loss
    optimizer_type = model.optimizer.__class__
    optimizer_config = model.optimizer.get_config()
    get_optimizer = lambda : optimizer_type.from_config(optimizer_config)
    results = list(range(len(test_nodes)))
    best_metric = float("inf")
    best_configuration = None
    for i,node_amount in enumerate(test_nodes):
        if node_amount is None:
            curr_layer = layers.pop(index)
            config = layer_configs.pop(index)
        else:
            layer_configs[index]["units"] = node_amount
        
        layers = utils.layers_from_configs(layers, layer_configs)
        model = tf.keras.models.Sequential(layers)
        model.build(np.shape(X))
        model.compile(
            optimizer=get_optimizer(),
            loss=loss
        )
        metric = utils.test_learning_speed(model, X, y,**kwargs)
        print(node_amount, metric)
        results[i] = [node_amount,metric]
        if metric<best_metric:
            best_metric = metric
            if best_configuration is None and node_amount is not None:
                best_configuration = layers[index].get_config()
            elif node_amount is None:
                best_configuration = None
            else:
                best_configuration["units"] = node_amount
        if node_amount is None:
            layers.insert(index, curr_layer)
            layer_configs.insert(index,config)
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return results
    if best_configuration == None:
        layer_configs.pop(index)
        layers.pop(index)
    else:
        layer_configs[index] = best_configuration
    layers = utils.layers_from_configs(layers, layer_configs)
    model = tf.keras.models.Sequential(layers)
    model.build(np.shape(X))
    model.compile(optimizer=get_optimizer(),
        loss=loss)
    return model


def opt_decay(model: tf.keras.models.Sequential,X,y,**kwargs):
    model = utils.check_compilation(model, X, **kwargs)
    decays = kwargs.get("decays",_default_decays.copy())
    if not (isinstance(decays, list)) or 1 in decays or not all([isinstance(x, float) or isinstance(x, int) for x in decays]):
        print(f"Invalid decay in opt_decay: {decays}. Expected list or numpy array.\n"+
              f"Continuing execution with default decays {_default_decays}")
        decays = _default_decays.copy()
    optimizer_config = model.optimizer.get_config()
    decays.append(optimizer_config.get("decay"))#0 if optimizer_config.get("decay") == 0.0 else optimizer_config.get("decay"))
    decays = set(decays)
    decays = sorted(decays)
    get_optimizer = lambda : optimizer_type.from_config(optimizer_config)
    return_metric = kwargs.get("return_metric",configurations.LEARNING_METRIC)
    optimizer_type = model.optimizer.__class__
    best_decay = 0
    best_metric = float("inf")
    #optimizer_config = model.optimizer.get_config()
    #best_decay = optimizer_config["decay"]
    #best_metric = utils.test_learning_speed(model,X,y,**kwargs)
    #results = list(range(len(decays)+2))
    results = list(range(len(decays)))
    #results[0] = ["decay",return_metric]
    #results[1] = [best_decay,best_metric]
    #print(f"decay: {best_decay}, {return_metric}:{best_metric}")
    for i,decay in enumerate(decays):
        optimizer_config["decay"] = decay
        model.build(np.shape(X))
        model.compile(optimizer=get_optimizer(),loss=model.loss)
        metric = utils.test_learning_speed(model,X,y,**kwargs)
        print(f"decay: {decay}, {return_metric}:{metric}")
        results[i] = [decay,metric]
        if(metric<best_metric):
            best_metric = metric
            best_decay = decay
    return_model = kwargs.get("return_model",True)
    if not return_model:
        return results
    optimizer_config["decay"] = best_decay
    model.build(np.shape(X))
    model.compile(optimizer=get_optimizer(),loss=model.loss)
    return model