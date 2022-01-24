from . import LossCallback as lcb
import tensorflow as tf
import numpy as np
import random
from . import configurations
import time
from sklearn.model_selection import train_test_split
import oparch

_layer_keys = {
    "dense":["units","activation"],
    "dropout":["rate"],
    "conv2d":["filters,kernel_size","activation"]
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
        kwargs: {"samples", "validation_split","return_metric","epochs","batch_size"}

    Raises:
        AttributeError: If model has not been built.

    Returns:
        float: return_metric key in LossCallback.learning_metrics. For example loss after last training epoch.
    """
    oparch.__reset_random__()
    allowed_kwargs = {"samples", "validation_split","return_metric","epochs","batch_size"}
    samples = kwargs.get("samples",configurations.TEST_SAMPLES)#TODO: If samples are configured during run time, the changes are not reflected
    validation_split = kwargs.get("validation_split",0.2)
    return_metric = kwargs.get("return_metric",configurations.LEARNING_METRIC)
    epochs = kwargs.get("epochs",configurations.TEST_EPOCHS)
    batch_size = kwargs.get("batch_size",configurations.BATCH_SIZE)
    if not isinstance(samples,int) or samples > np.size(X,axis=0):
        samples = np.size(X,axis=0)
        #print(f"Incorrect sample size. Using {samples} samples.")
    if validation_split<0 or validation_split>1:
        validation_split = 0.2
        #print(f"Incorrect validation_split. Using {validation_split} split.")
    if not isinstance(epochs, int) or epochs<1:
        epochs = 1
        #print(f"Incorrect epochs. Using {epochs} epochs.")
    if not isinstance(batch_size, int) or batch_size<1 or (batch_size>samples and samples > 0):
        batch_size = configurations.BATCH_SIZE
        #print(f"Incorrect batch_size. Using {batch_size} batch_size.")
    try:
        model.optimizer.get_weights()
    except AttributeError:
        raise AttributeError("The model must be built before testing the learning speed")
    #rebuild and compile the model to get a clean optimizer
    if model.optimizer.get_weights(): #If list is not empty
        model.build(np.shape(X))
        model.compile(optimizer=model.optimizer.__class__.from_config(model.optimizer.get_config()),
                  loss=model.loss)
        print("rebuild and compile the model to get a clean optimizer")
    #Save the models weights to return the model to its original state after testing the learning speed
    model.save_weights("test_weights.h5")
    verbose = 0
    validation_data = None
    if samples>0:
        sample_indexes = random.sample(range(np.size(X,axis=0)),samples)
        X = X[sample_indexes]
        y = y[sample_indexes]
    if("VALIDATION" in return_metric): #If the learning metric should be calculated from validation set
        X, x_test, y, y_test = train_test_split(X, y,test_size=validation_split, random_state=42)
        validation_data = (x_test,y_test)
    cb_loss = lcb.LossCallback(samples=samples,
                               epochs=epochs,
                               batch_size=batch_size,
                               )
    start = time.time()
    hist = model.fit(
        X, y,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        batch_size=batch_size,
        callbacks=[cb_loss],
        shuffle=True,
        use_multiprocessing=True,
    )
    elapsed_time = time.time() - start
    #Load the weights the model had when it came to testing, so testing doesn't affect the model itself
    #Rebuild and recompile to give the model a clean optimizer
    model.load_weights("test_weights.h5")
    model.build(np.shape(X))
    model.compile(optimizer=model.optimizer.__class__.from_config(model.optimizer.get_config()),
                  loss=model.loss)
    #if only one epoch is done, returns the last loss
    return_value = cb_loss.learning_metric[return_metric]
    if return_value == None or return_value == np.nan:
        print(f"Return metric {return_metric} is None. Using LAST_LOSS instead.")
        return cb_loss.learning_metric["LAST_LOSS"]
    return return_value

def check_compilation(model: tf.keras.models.Sequential, X, kwarg_dict={}, **kwargs) -> tf.keras.models.Sequential:
    oparch.__reset_random__()
    if model.optimizer is None: #if model is not compiled, compile it with optimizer and loss kwargs
        try:
            model.build(np.shape(X))
            optimizer = kwargs.get("optimizer",kwarg_dict["optimizer"])
            loss = kwargs.get("loss",kwarg_dict["loss"])
            model.compile(optimizer=optimizer, loss=loss)
        except KeyError:
            raise KeyError("If the model is not compiled, you must specify the optimizer and loss")
    elif model.optimizer.get_weights() or model.weights: #TODO: Model weights are not empty if the model is compiled, which it always is here
        layers = get_copy_of_layers(model.layers)
        optimizer_config = model.optimizer.get_config()
        optimizer = model.optimizer.__class__.from_config(optimizer_config)
        optimizer = kwarg_dict.get("optimizer",optimizer)
        loss = kwarg_dict.get("loss",model.loss)
        model = tf.keras.models.Sequential(layers)
        model.build(np.shape(X))
        model.compile(optimizer=optimizer, loss=loss)
        #print("It is recommended to use a fresh model, that has not been trained, to ensure correct results and faster execution")
    return model

def get_layers_config(layers: list)->list:
    configs = [layer.get_config() for layer in layers]
    return configs

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
    string = string + f"Loss function: {type(dic.get('loss_function')).__name__}\n"
    string = string + f"Learning metrics: {dic.get('learning_metrics')}\n"
    #string = string + f"{'LAYER':<12}{'ACTIVATION':<12}{'UNITS'}\n"
    for layer in dic["layers"]: #Here variable layer is the name of the layer
        string = string + f"{layer:<12}"
        keys = dic["layers"][layer].keys()
        for key in keys:
            string = string + f"{key:<12}:{dic['layers'][layer][key]:<12}"
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
    new_layers = [layer.__class__.from_config(config) for layer,config in zip(layers, configs)]
    return new_layers

def get_dense_indices(model: tf.keras.models.Sequential) -> list:
    dense_indices = []
    for i,layer in enumerate(model.layers):
        if (isinstance(layer, tf.keras.layers.Dense)):
            dense_indices.append(i)
    return dense_indices