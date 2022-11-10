from . import LossCallback as lcb
import tensorflow as tf
import numpy as np
import random
from . import configurations
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import oparch
import warnings

_layer_keys = {
    "dense":["units","activation"],
    "dropout":["rate"],
    "conv2d":["filters","kernel_size","activation","strides"],
    "max_pooling2d":["pool_size","strides"],
    "flatten":["dtype"],
}

def test_learning_speed(model: tf.keras.models.Sequential, X: np.ndarray,
                        y: np.ndarray,**kwargs) -> float:
    """Tests a models learning speed. Returns an attribute from LossCallback.learning_metrics specified
    by the return_metric variable. If the model optimizer has weights then recompiles the model with
    the same optimizer configs. Saves the models weights to "test_weights.h5" before calling model.fit. After fitting the data,
    loads the saved weights from the file and recompiles the model. This makes it possible to call this function without
    affecting the state of the model and not requiring to create a copy of the model.
    

    Args:
        model (tf.keras.models.Sequential): A model that has been compiled, but not trained.
        X (np.ndarray): Feature data
        y (np.ndarray): data to be predicted
        kwargs: {"samples", "validation_split","return_metric","epochs","batch_size", verbose","metrics"}

    Raises:
        AttributeError: If model has not been built.

    Returns:
        float: return_metric key in LossCallback.learning_metrics. For example loss after last training epoch.
    """
    tf.keras.backend.clear_session()
    oparch.__reset_random__()
    samples = kwargs.get("samples",configurations.get_default_misc("samples"))#TODO: If samples are configured during run time, the changes are not reflected
    validation_split = kwargs.get("validation_split",configurations.get_default_misc("validation_split"))
    return_metric = kwargs.get("return_metric",configurations.get_default_misc("learning_metric"))
    epochs = kwargs.get("epochs",configurations.get_default_misc("epochs"))
    batch_size = kwargs.get("batch_size",configurations.get_default_misc("batch_size"))
    verbose = kwargs.get("verbose",configurations.get_default_misc("verbose"))
    decimals = kwargs.get("decimals",configurations.get_default_misc("decimals"))
    metrics = kwargs.get("metrics",[])
    if not isinstance(samples,int) or samples > np.size(X,axis=0) or samples <= 0:
        samples = np.size(X,axis=0)
        msg = f"Sample size not specified. Using all ({samples}) samples."
        warnings.warn(msg)
        #print(f"Incorrect sample size. Using {samples} samples.")
    if validation_split<0 or validation_split>1:
        validation_split = 0.2
        msg = f"Incorrect validation_split. Using {validation_split} split."
        warnings.warn(msg)
        #print(f"Incorrect validation_split. Using {validation_split} split.")
    if not isinstance(epochs, int) or epochs<1:
        epochs = 1
        msg = f"Incorrect epochs. Using {epochs} epochs."
        warnings.warn(msg)
        #print(f"Incorrect epochs. Using {epochs} epochs.")
    if not isinstance(batch_size, int) or batch_size<1 or (batch_size>samples and samples > 0):
        batch_size = configurations.get_default_misc("batch_size")
        msg = f"Incorrect batch_size. Using {batch_size} batch_size."
        warnings.warn(msg)
        #print(f"Incorrect batch_size. Using {batch_size} batch_size.")
    #Always add 'accuracy' to metrics
    if "accuracy" not in metrics:
        metrics.append("accuracy")
    try:
        model.optimizer.get_weights()
    except AttributeError:
        raise AttributeError("The model must be built and compiled, but not trained before testing the learning speed.")
    optimizer_type = model.optimizer.__class__
    optimizer_config = model.optimizer.get_config()
    loss = model.loss
    #rebuild and compile the model to get a clean optimizer
    if model.optimizer.get_weights(): #If list is not empty TODO: If model is trained and then compiled with empty optimizer, incorrect results_cpy
        layers = get_copy_of_layers(model.layers)
        model = tf.keras.models.Sequential(layers)
        model.build(np.shape(X))
        print("Rebuilt the model because optimizer was not empty.")
    #Save the models weights to return the model to its original state after testing the learning speed
    model.save_weights("test_weights.h5")
    validation_data = None
    model.compile(optimizer=optimizer_type.from_config(optimizer_config),
                  loss=loss,metrics=metrics)
    if samples>0:
        sample_indexes = random.sample(range(np.size(X,axis=0)),samples,)
        X = X[sample_indexes]
        y = y[sample_indexes]
    if("VALIDATION" in return_metric): #If the learning metric should be calculated from validation set
        X, x_test, y, y_test = train_test_split(X, y,test_size=validation_split, random_state=42)
        validation_data = (x_test,y_test)
    cb_loss = lcb.LossCallback(
                               **kwargs
                               )
    oparch.__reset_random__()
    start = time.time()
    hist = model.fit(
        X, y,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        batch_size=batch_size,
        callbacks=[cb_loss],
        shuffle=False,
    )
    elapsed_time = time.time() - start
    #print("Elapsed Time:",elapsed_time,"\n")
    #Load the weights the model had when it came to testing, so testing doesn't affect the model itself
    #Rebuild and recompile to give the model a clean optimizer
    model.load_weights("test_weights.h5")
    model.build(np.shape(X))
    model.compile(optimizer=model.optimizer.__class__.from_config(model.optimizer.get_config()),
                  loss=model.loss)
    #if only one epoch is done, returns the last loss
    return_value = cb_loss.learning_metric.get(return_metric)
    if return_value == None or np.isnan(return_value):
        if return_metric != "LAST_LOSS":
            print(f"Return metric {return_metric} is None. Using LAST_LOSS instead.")
        return cb_loss.learning_metric.get("LAST_LOSS")
    return return_value

def grid_search(results : list,**kwargs):
    """
    Takes in a list of tuples, sorts them by 0th index, finds the minimum,
    calculates an interval where the minimum value is in the middle,
    rounds the values in the interval and removes duplicates and values which are already counted.
    
    Doesn't require a difference between previously calculated values, so results greater than minimum
    don't have to accurate. It is enough to know that the results corresponding to values are greater than minimum. So this supports early_stopping.
    
    Can only find one local minima.

    Args:
        results (list): list of tuples with two values (value, result)
        decimals (_type_, optional): To which decimal point should be rounded. Defaults to 5.
        samples_in_interval (_type_, optional): How many values should be in the interval. Defaults to 10:int.

    Returns:
        vals (list): an interval where the value corresponding to the minimum result is in the middle
    """
    decimals = kwargs.get("decimals",5)
    samples_in_interval = kwargs.get("samples_in_interval",10)
    tried = [t[0] for t in results]
    if None in tried:
        results = results.copy()
        results.pop(tried.index(None))
        tried.remove(None)
    results = sorted(results,key=lambda x : x[0])
    types = [type(_) for _ in tried]
    n_int = types.count(int)
    n_float = types.count(float) + types.count(np.float64)
    if n_int+n_float != len(types):
        return []
    if n_float == 0:
        decimals = None
    if len(results)<3:
        return []
    s = min(results,key=lambda x:x[1])
    sindex = results.index(s)
    #if minimum cost is the first or last tested value return empty list because grid cant be narrowed around the minima
    if sindex in [0,len(results)-1]:
        return []
    vals = list(np.linspace(results[sindex-1][0], results[sindex+1][0],samples_in_interval))
    vals = [round(v,decimals) for v in vals]
    vals = list(set(vals))
    vals = [v for v in vals if v not in tried]
    return vals


def layers_from_configs(layers: list,configs: list):
    """returns layers from a list of layer configurations

    Args:
        layers (list): _description_
        configs (list): _description_

    Returns:
        _type_: _description_
    """    
    configs = add_seed_configs(configs)
    if isinstance(layers,tf.keras.models.Sequential):
        layers = layers.layers
    new_layers = [layer.__class__.from_config(config) for layer,config in zip(layers, configs)]
    return new_layers

def check_compilation(model: tf.keras.models.Sequential, X, **kwargs) -> tf.keras.models.Sequential:
    """Rebuilds and recompiles the model if possible. Adds seed values to the layers.

    Args:
        model (tf.keras.models.Sequential): _description_
        X (_type_): _description_

    Raises:
        KeyError: if model is not built and compiled and no loss fun and optimizer specified

    Returns:
        tf.keras.models.Sequential: _description_
    """    
    layers = get_copy_of_layers(model.layers)
    if model.optimizer is None: #if model is not compiled, compile it with optimizer and loss kwargs
        model = tf.keras.models.Sequential(layers)
        model.build(np.shape(X))
        optimizer = kwargs.get("optimizer")
        loss = kwargs.get("loss")
        if optimizer is not None and loss is not None:
            model.compile(optimizer=optimizer, loss=loss)
        else:
            raise KeyError("If the model is not compiled, you must specify the optimizer and loss when checking compilation.")
    elif model.optimizer.get_weights() or model.weights: #TODO: Model weights are not empty if the model is compiled, which it always is here
        optimizer_config = model.optimizer.get_config()
        optimizer = model.optimizer.__class__.from_config(optimizer_config)
        optimizer = kwargs.get("optimizer",optimizer)
        loss = kwargs.get("loss",model.loss)
        model = tf.keras.models.Sequential(layers)
        model.build(np.shape(X))
        model.compile(optimizer=optimizer, loss=loss,metrics=["accuracy"])
    return model

def check_types(*args):
    """Checks that every provided tuple isinstance(obj,cls)

    Raises:
        TypeError: If incorrect type
    """    
    incorrect = []
    for a in args:
        if not isinstance(a[0], a[1]):
            incorrect.append(a)
    if incorrect:
        msg = f"Incorrect arguments: expected {incorrect[0][1]} but received type {incorrect[0][0].__class__}"
        raise TypeError(msg)
    
def new_vals_from_results(results:list, **kwargs):
    results_cpy = results.copy()
    tried_vals = [x[0] for x in results]
    none_result = (float("inf"),float("inf"))
    if None in tried_vals:
        none_index = tried_vals.index(None)
        none_result = results_cpy.pop(none_index)
    if not all([isinstance(x[0], float) or isinstance(x[0],int) for x in results_cpy]):
        return (len(results),[x[0] for x in results])
    results_cpy = sorted(results_cpy,key=lambda x: x[0])
    tried_vals = [x[0] for x in results_cpy]
    result_vals = [x[1] for x in results_cpy]
    minimum = min(result_vals)
    mini_i = result_vals.index(minimum)
    if (none_result[1] < minimum or mini_i==len(tried_vals)-1 or mini_i==0):
        return (len(results),[x[0] for x in results])
    try:
        new_vals = list(np.linspace(tried_vals[mini_i-1], tried_vals[mini_i + 1],10))
    except IndexError:
        return (len(results),[x[0] for x in results])
    new_vals = [round(x) for x in new_vals] #Round to int
    new_vals = list(set(new_vals)) #Remove duplicates possibly made from rounding
    new_vals = sorted([x for x in new_vals if x not in tried_vals])
    return (0,new_vals)
    
    

def new_vals(vals: list, results: list, i : int,**kwargs):
    """Calculates new quesses for a float/int parameter if necessary.
    Returns the running index in the while loop as int and the new values as list
    """
    tried_vals = [x[0] for x in results]
    new_i = i
    results_from_i = results[-i:]
    if None in tried_vals:
        results_from_i.pop(tried_vals.index(None))
        tried_vals.remove(None)
        new_i = i - 1
    if new_i < 3 or not all([isinstance(x, float) or isinstance(x,int) for x in tried_vals]):
        return (i,vals)
    #If only descending or ascending
    decimals = kwargs.pop("decimals",configurations.get_default_misc("decimals"))
    if (True and False) not in [k<10**(-decimals) for k in np.diff([r[1] for r in results_from_i])]:
        return (i,vals)
    are_int = False
    if all([isinstance(x,int) for x in tried_vals]):
        are_int = True
    #If local max and not min
    if not (results_from_i[-1][1] > results_from_i[-2][1]):
        return (i,vals)
    #Here we have made the checks and are sure that there is a local minima
    new_vals = list(np.linspace(tried_vals[-3], tried_vals[-1],10))#Get 10 values between the third last and last val
    new_vals[0] = new_vals[0] + 10**(-decimals) # The third last result can be abstractly close to the local minima, so we only add a small number
    if are_int:
        new_vals = [round(x) for x in new_vals] #Round to int
    else:
        new_vals = [round(x,decimals) for x in new_vals] #round_accuracy_parameter
    new_vals = list(set(new_vals)) #Remove duplicates possibly made from rounding
    #new_vals = sorted(new_vals)
    new_vals = sorted([x for x in new_vals if x not in vals])
    return (0,new_vals)

def get_layers_config(layers: list)->list:
    """returns a list of layer configurations from a layer list
    """    
    configs = [layer.get_config() for layer in layers]
    return configs

def plot_results(results):
    """Given a results_cpy list of tuples, plots the results_cpy.
    """    
    plt.figure()
    vals = [x[0] for x in results]
    if None in vals:
        results.pop(vals.index(None))
    if not all([isinstance(x[0], int) or isinstance(x[0], float) for x in results]):
        return
    results = sorted(results, key=lambda x : x[0])
    vals = [x[1] for x in results]
    res = [x[0] for x in results]
    #plt.plot(res,vals)
    plt.scatter(res,vals)
    plt.xlabel("Value")
    plt.ylabel("Learning metric")
    plt.show()

def create_dict(model: tf.keras.models.Sequential,learning_metrics={}) -> dict:
    dic = {
        "optimizer":None,
        "loss_function":None,
        "layers":{},
        "learning_metrics":learning_metrics,
    }
    if model.optimizer != None:
        dic["optimizer"] = model.optimizer.get_config()
    else:
        raise AttributeError("Model must be compiled before creating a dictionary.")
    if isinstance(model.loss,str):
        dic["loss_function"] = model.loss
    else:
        dic["loss_function"] = model.loss.__class__.__name__
    layer_configs = [layer.get_config() for layer in model.layers]
    layers_summary = {}
    for i,config in enumerate(layer_configs):
        name = config.get("name")
        layers_summary[name] = {}
        keys = []
        for layer_name in _layer_keys.keys():
            if layer_name in name:
                keys = _layer_keys[layer_name]
                break
        if not keys:
            print(f"Layer {name} has not been implemented yet.")
        for key in keys:
            layers_summary[name][key] = config.get(key)
    #Now: layers_summary = {
        # "ccconv2d":{"filter":16,"kernel_size":(2,2),"activation":"relu"}
        # "dense_1":{"units":1,"activation":"relu"},
        # }
    for layer in layers_summary:
        dic["layers"][layer] = layers_summary[layer]
    return dic

def _string_format_model_dict(dic: dict):
    string = f"\nOptimizer: {dic.get('optimizer')}\n"
    string = string + f"Loss function: {dic.get('loss_function')}\n"
    string = string + f"Learning metrics: {dic.get('learning_metrics')}\n"
    for layer in dic["layers"]: #Here variable layer is the name of the layer
        string = string + f"{layer:<16}"
        keys = list(dic["layers"][layer].keys())
        for key in keys:
            string = string + f"{key:<16}:{str(dic['layers'][layer][key]):<16}"
        string = string + "\n"
    return string

def print_model(dic, learning_metrics={}):
    """Takes in a a dict or a model. If argument is model, creates a dictionary from that model that is then used to print the model.

    Args:
        dic (dict or model): dictionary or model to be printed to stdout
        learning_metrics (dict, optional): Dictionary with learning metrics for example {"LAST_LOSS":0.03567}. Defaults to {}.

    Raises:
        TypeError: dic is not a Sequential model or a dictionary
    """    
    if isinstance(dic,tf.keras.models.Sequential):
        dic = create_dict(dic,learning_metrics=learning_metrics)
    if not isinstance(dic, dict):
        raise TypeError(f"Excpected argument of type Sequential or dict, but received {type(dic)}")
    string = _string_format_model_dict(dic)
    print(string)
    
    
def get_copy_of_layers(layers:list) -> list:
    configs = get_layers_config(layers)
    names = list(range(len(configs)))
    for config in configs:
        name = config["name"]
        if name in names:
            config["name"] = "c"+config["name"]
        names.append(name)
    configs = add_seed_configs(configs)
    new_layers = [layer.__class__.from_config(config) for layer,config in zip(layers, configs)]
    return new_layers

def add_seed_configs(configs):
    for config in configs:
        keys = config.keys()
        if "seed" in keys:
            config["seed"] = 42
        elif "kernel_initializer" in keys and "config" in config["kernel_initializer"].keys():
            config["kernel_initializer"]["config"]["seed"] = 42
        else:
            pass#print(f"No seed key found for layer {config.get('name')}")
    return configs

def get_dense_indices(model: tf.keras.models.Sequential) -> list:
    dense_indices = []
    for i,layer in enumerate(model.layers):
        if (isinstance(layer, tf.keras.layers.Dense)):
            dense_indices.append(i)
    return dense_indices

def get_activation_indices(model: tf.keras.models.Sequential) -> list:
    indices = []
    for i,layer in enumerate(model.layers):
        config = layer.get_config()
        if config.get("activation") is not None:
            indices.append(i)
    return indices