import tensorflow as tf
import warnings
import numpy as np

default_misc = {
    "image_size":(180,180), #Default image_size
    "batch_size": 32, #default batch_size
    "learning_metric":"LAST_LOSS", #The metric to minimize when optimizing parameters
    "epochs": 5, # Default number of epochs to run when testing learning speed
    "samples": float("inf"), #How many samples to train on when testing learning speed
    "verbose" : 0, #verbose
    "validation_split" : 0.2, #
    "decimals" : 5, #Decimal points to round to
    "maxiter":50, #maximum iterations per optimizing call
    "optimizing_algo":"TNC", #Default optimizing algorithm
    
    
}
default_intervals = {
    "amsgrad":[1,0],
    "learning_rate":[0.00001, 0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
    "decay":[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    "momentum":[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
    "units":[None,1,2,4,8,16,32,64,128,256,512],
    "filters":[None,1,2,4,8,16,32,64,128,256,512],
    "strides":[None,(1,1),(2,1),(2,2),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3),(4,4),(5,1),(5,2),(5,3),(5,4),(5,5)],
    "kernel_size":[(2,1),(2,2),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3),(4,4),(5,1),(5,2),(5,3),(5,4),(5,5)],
    "pool_size":[(2,1),(2,2),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3),(4,4),(5,1),(5,2),(5,3),(5,4),(5,5)],
    "activation":["relu","sigmoid","linear","tanh","exponential","elu","softmax",],
    "rho":list(np.linspace(0.8,1,10,endpoint=True)),
}    

ACTIVATION_FUNCTIONS = {
    "sigmoid":tf.keras.activations.sigmoid,
    "linear":None,
    "tanh":tf.keras.activations.tanh,
    "exponential":tf.keras.activations.exponential,
    "relu":tf.keras.activations.relu,
    "elu":tf.keras.activations.elu,
    "softmax":tf.keras.activations.softmax,
}

REGRESSION_LOSS_FUNCTIONS = {
    "mse":tf.keras.losses.MeanSquaredError(),
    "mean_squared_logarithmic_error":tf.keras.losses.MeanSquaredLogarithmicError(),
    "mean_absolute_error":tf.keras.losses.MeanAbsoluteError(),
}

LOSS_FUNCTIONS = {
    "categorical_hinge":tf.keras.losses.CategoricalHinge(),
    "hinge":tf.keras.losses.Hinge(),
    "huber":tf.keras.losses.Huber(),
    "KLDivergence":tf.keras.losses.KLDivergence(),
    "mse":tf.keras.losses.MeanSquaredError(),
    "mean_squared_logarithmic_error":tf.keras.losses.MeanSquaredLogarithmicError(),
    "poisson":tf.keras.losses.Poisson(),
    "categorical_crossentropy":tf.keras.losses.CategoricalCrossentropy(),
    "sparse_categorical_crossentropy":tf.keras.losses.SparseCategoricalCrossentropy()
}

PROBABILITY_DISTRIBUTION_ACTIVATION_FUNCTIONS = {
    "sigmoid":tf.keras.activations.sigmoid,
    "softmax":tf.keras.activations.softmax,
}

OPTIMIZERS = {
    "sgd":tf.keras.optimizers.SGD(),
    "adadelta":tf.keras.optimizers.Adadelta(),
    "adagrad":tf.keras.optimizers.Adagrad(),
    "adam":tf.keras.optimizers.Adam(),
    "Adamax":tf.keras.optimizers.Adamax(),
    "rmsprop":tf.keras.optimizers.RMSprop(),
}

BINARY_LOSSES = {
    "binary_crossentropy":tf.keras.losses.BinaryCrossentropy,
}

    
def set_default_misc(**kwargs):
    global default_misc
    for item in kwargs.items():
        if item[0] not in default_misc:
            print(f"Item {item[0]} not found.")
            continue
        print(f"Changing {item[0]} default value from {default_misc[item[0]]} to {item[1]}")
        default_misc[item[0]] = item[1]
        
def set_default_intervals(**kwargs):
    global default_intervals
    for item in kwargs.items():
        if item[0] not in default_intervals.keys():
            print(f"Adding {item[0]} to default values")
        else:
            print(f"Changing {item[0]} default value from {default_intervals[item[0]]} to {item[1]}")
        default_intervals[item[0]] = item[1]

def get_default_interval(param=None):
    global default_intervals
    if param == None:
        return default_intervals.copy()
    return default_intervals[param].copy()

def get_default_misc(param=None):
    global default_misc
    if param == None:
        return default_intervals.copy()
    return default_misc[param]