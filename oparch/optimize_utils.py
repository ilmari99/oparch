from . import LossCallback as lcb
import tensorflow as tf
import numpy as np
import random
from . import configurations
import time
from sklearn.model_selection import train_test_split

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
    allowed_kwargs = {"samples", "validation_split","return_metric","epochs","batch_size"}
    samples = kwargs.get("samples",configurations.TEST_SAMPLES)#TODO: If samples are configured during run time, the changes are not reflected
    validation_split = kwargs.get("validation_split",0.2)
    return_metric = kwargs.get("return_metric",configurations.LEARNING_METRIC)
    epochs = kwargs.get("epochs",configurations.TEST_EPOCHS)
    batch_size = kwargs.get("batch_size",configurations.BATCH_SIZE)
    if samples > np.size(X,axis=0):
        samples = np.size(X,axis=0)
    
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
    #samples = np.shape(y)[0] #TODO: this uses all available data instead of samples
    verbose = 0
    validation_data = None
    sample_indexes = random.sample(range(np.size(X,axis=0)),samples)
    X = X[sample_indexes]
    y = y[sample_indexes]
    if("VALIDATION" in configurations.LEARNING_METRIC): #If the learning metric should be calculated from validation set
        X, x_test, y, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
        validation_data = (x_test,y_test)
    cb_loss = lcb.LossCallback(samples=samples)
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
    if(configurations.TEST_EPOCHS==1):
        return cb_loss.learning_metric["LAST_LOSS"]
    return cb_loss.learning_metric[return_metric]

def check_compilation(model: tf.keras.models.Sequential, X, kwarg_dict, **kwargs) -> tf.keras.models.Sequential:
    if model.optimizer is None: #if model is not compiled, compile it with optimizer and loss kwargs
        try:
            model.build(np.shape(X))
            model.compile(optimizer=kwarg_dict["optimizer"], loss=kwarg_dict["loss"])
            print("Model compiled with given optimizer and loss")
        except KeyError:
            raise KeyError("If the model is not compiled, you must specify the optimizer and loss")
    if model.optimizer.get_weights() or model.weights: #TODO: Model weights are not empty if the model is compiled, which it always is here
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
    try:
        dic["optimizer"] = model.optimizer.get_config()
        dic["loss_function"] = model.loss.__class__.__name__
        layer_configs = [layer.get_config() for layer in model.layers]
    except AttributeError:
        raise AttributeError("Model must be compiled before creating a dictionary.")
    layers_summary = [(config.get("name"),config.get("units"), config.get("activation")) for config in layer_configs]
    for summary in layers_summary:
        dic["layers"][summary[0]] = {}
        dic["layers"][summary[0]]["units"] = summary[1]
        dic["layers"][summary[0]]["activation"] = summary[2]
    return dic

def _string_format_model_dict(dic: dict):
    string = f"\nOptimizer: {dic.get('optimizer')}\n"
    string = string + f"Loss function: {type(dic.get('loss_function')).__name__}\n"
    string = string + f"{dic.get('learning_metrics')}\n"
    string = string + f"{'LAYER':<12}{'ACTIVATION':<12}{'UNITS'}\n"
    for layer in dic["layers"]:
        string = string + f"{layer:<12}{dic['layers'][layer]['activation']:<12}{dic['layers'][layer]['units']}\n"
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