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
    OptimizedModel.optimize_loss_fun(optimized_model.model,x_train,y_train)
    
    OptimizedModel.optimize_optimizer(optimized_model.model,x_train,y_train)
    
    OptimizedModel.optimize_learning_rate(optimized_model.model,x_train,y_train)
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
        OptimizedModel.optimize_loss_fun(optimized_model.model,x_train,y_train)
        OptimizedModel.optimize_optimizer(optimized_model.model,x_train,y_train)
        OptimizedModel.optimize_learning_rate(optimized_model.model,x_train,y_train)
        #optimized_model.model.load_weights("saved_weights.h5")
        
        metric = mot.test_learning_speed(optimized_model.model, x_train, y_train)
        print(f"{configurations.LEARNING_METRIC} after optimization: {metric}")
    return optimized_model

def check_compilation(model: tf.keras.models.Sequential, X, kwarg_dict):
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
        
        

def opt_dense(model: tf.keras.models.Sequential, index, X, y, **kwargs):
    model = check_compilation(model, X, kwargs)
    
    #Now we have a model that is not trained
    
    verbose = 0
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
    #layers.insert(index,curr_layer)
    #print(f"{configurations.LEARNING_METRIC} with no layer at index {index}: {best_metric}")

    best_metric = float("inf")###########Only until empty layer test works
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
        (configuration, metric) = opt_activation(test_model, index, X, y)
        if metric<best_metric:
            best_metric = metric
            best_configuration = configuration
    return (best_configuration, best_metric)
    

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
            #TODO: This is not worth the time. Using these seems to get a worse answer anyway, for some reason.
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