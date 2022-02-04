import numpy as np
import tensorflow as tf
from . import optimize_utils as utils
from . import LossCallback as lcb
from . import configurations
import oparch
_default_learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]#[1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
_default_nodes = [2**i for i in range(0,8)] #Node amounts to test
_default_decays = sorted([0.95, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0])
defaults = {
    "learning_rate":[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
    "decay":[0.95, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0]
}
def opt_all_units(model,X,y,**kwargs):
    indices = oparch.utils.get_dense_indices(model)
    if indices[-1] == len(model.layers)-1:
        indices.pop(-1)
    orig_layers = len(model.layers)
    removed = 0
    for i in indices:
        model = opt_dense_units(model, i+removed, X, y,**kwargs)
        removed = len(model.layers) - orig_layers
    return model

def opt_all_activations(model,X,y,**kwargs):
    indices = utils.get_activation_indices(model)
    for i in indices:
        model = opt_activation(model, i, X, y,**kwargs)
    return model

def opt_learning_rate(model: tf.keras.models.Sequential, X: np.ndarray, y: np.ndarray,**kwargs) -> tf.keras.models.Sequential or list:
    model = opt_optimizer_parameter(model, X, y, "learning_rate",**kwargs)
    return model

def opt_loss_fun(model: tf.keras.models.Sequential,X,y,**kwargs):
    model = utils.check_compilation(model, X, **kwargs)
    return_metric = kwargs.get("return_metric")
    epochs = kwargs.get("epochs",configurations.TEST_EPOCHS)
    if return_metric == "RELATIVE_IMPROVEMENT_EPOCH" or return_metric is None:
        kwargs["return_metric"] = "RELATIVE_IMPROVEMENT_EPOCH"
        return_metric = "RELATIVE_IMPROVEMENT_EPOCH"
        if epochs<2:
            kwargs["epochs"] = 2
    categorical = kwargs.pop("categorical",False)
    if categorical: #TODO Lisää ehto ilman kwargeja
        loss_functions = list(configurations.LOSS_FUNCTIONS.values())
    else:
        loss_functions = list(configurations.REGRESSION_LOSS_FUNCTIONS.values())
    loss_functions.append(model.loss)
    results = list(range(len(loss_functions)))
    best_metric = float("inf")
    best_loss_fun = model.loss.__class__
    for i, loss_fun in enumerate(loss_functions):
        model.compile(optimizer=model.optimizer, loss=loss_fun)
        try:
            metric = utils.test_learning_speed(model, X, y, **kwargs)
        except ValueError:
            metric = None
        print(type(loss_fun).__name__, return_metric,metric)
        results[i] = [type(loss_fun).__name__,metric]
        if(metric is not None and metric<best_metric):
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
    loss = model.loss
    get_optimizer = lambda : optimizer_type.from_config(optimizer_config)
    layers = model.layers
    best_activation = layers[index].activation
    layer_type = model.layers[index].__class__
    loss_dict = loss.get_config()
    if (index == len(layers)-1) and not (loss_dict.get("from_logits",True)):
        kwargs["return_metric"] = "NEG_ACCURACY"
        activation_functions = list(configurations.PROBABILITY_DISTRIBUTION_ACTIVATION_FUNCTIONS.items())
    else:
        activation_functions = list(configurations.ACTIVATION_FUNCTIONS.items())
    current_name = tf.keras.activations.serialize(layers[index].activation)
    if not any([act[0] is current_name for act in activation_functions]):
        activation_functions.append((current_name,layers[index].activation))
    results = list(range(len(activation_functions)))
    best_metric = float("inf")
    for i, activation in enumerate(activation_functions):
        #index_layer_configuration["activation"] = activation
        #layers[index] = layer_type.from_config(index_layer_configuration)
        name = activation[0]
        fun = activation[1]
        layers[index].activation = fun
        model = tf.keras.models.Sequential(layers)
        model.build(np.shape(X))
        model.compile(optimizer=get_optimizer(), loss=loss)
        metric = utils.test_learning_speed(model, X, y,**kwargs)
        results[i] = [name,metric]
        print(name,metric)
        if metric < best_metric:
            best_metric = metric
            best_activation = activation
    return_model = kwargs.get("return_model",True)
    if not return_model:
        print(f"best activation {best_activation[0]}")
        return results
    #layers[index] = layer_type.from_config(best_configuration)
    layers[index].activation = best_activation[1]
    model = tf.keras.models.Sequential(layers)
    model.build(np.shape(X))
    model.compile(optimizer=get_optimizer(),
                  loss=loss)
    print(f"best activation {best_activation[0]}")
    return model

def opt_dense_units(model: tf.keras.models.Sequential, index, X, y, **kwargs):
    model = utils.check_compilation(model, X, **kwargs)
    if not isinstance(model.layers[index],tf.keras.layers.Dense):
        raise KeyError(f"layer {model.layers[index]} is not a Dense layer.")
    print(f"Optimizing dense units at index {index}")
    test_nodes = kwargs.get("test_nodes",_default_nodes.copy())
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

def opt_optimizer_parameter(model,X,y,param,**kwargs):
    model = utils.check_compilation(model, X, **kwargs)
    if not hasattr(model.optimizer,param):
        raise AttributeError(f"{model.optimizer} doesn't have {param} attribute.")#TODO: if is string and return model if false
    return_metric = kwargs.get("return_metric",configurations.LEARNING_METRIC)
    if "RELATIVE" in return_metric:
        epochs = kwargs.get("epochs", configurations.TEST_EPOCHS)
        if epochs < 2:
            kwargs["epochs"] = 2
    vals = kwargs.pop(param,defaults.get(param))
    optimizer_config = model.optimizer.get_config()
    optimizer_type = model.optimizer.__class__
    get_optimizer = lambda : optimizer_type.from_config(optimizer_config)
    curr_param = optimizer_config.get(param) #The current parameter value is best
    if curr_param not in vals:
        vals.append(curr_param)
    vals = set(vals)
    vals = sorted(vals)
    results = []
    for i,val in enumerate(vals):
        optimizer_config[param] = val
        model.compile(optimizer=get_optimizer(),loss=model.loss)
        metric = utils.test_learning_speed(model,X,y,**kwargs)
        print(f"{param:<16}{val:<16}{return_metric:<16}{metric:<16}")
        results.append((val,metric))
    return_model = kwargs.get("return_model",True)
    if return_model:
        best = min(results,key=lambda x : x[1])
        optimizer_config[param] = best[0]
        model.compile(optimizer=get_optimizer(),loss=model.loss)
        return model
    else:
        return results


def opt_decay(model: tf.keras.models.Sequential,X,y,**kwargs):
    model = opt_optimizer_parameter(model, X, y, "decay",**kwargs)
    return model