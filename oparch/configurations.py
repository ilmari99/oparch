import tensorflow as tf
IMAGE_SIZE = (180,180)
BATCH_SIZE = 32
LEARNING_METRIC = "LAST_LOSS" #LAST_LOSS or RELATIVE_IMPROVEMENT_EPOCH seems to work best
TEST_EPOCHS = 5
TEST_SAMPLES = 5000
VERBOSE = 0
VALIDATION_SPLIT = 0.2

def configure(**kwargs):
    allowed_kwargs = {"image_size","batch_size","learning_metric","epochs","samples","verbose","validation_split"}
    global IMAGE_SIZE,BATCH_SIZE,LEARNING_METRIC,TEST_EPOCHS,TEST_SAMPLES,VERBOSE,VALIDATION_SPLIT
    IMAGE_SIZE = kwargs.get("image_size",IMAGE_SIZE)
    BATCH_SIZE = kwargs.get("batch_size",BATCH_SIZE)
    LEARNING_METRIC = kwargs.get("learning_metric",LEARNING_METRIC)
    TEST_EPOCHS = kwargs.get("epochs",TEST_EPOCHS)
    TEST_SAMPLES = kwargs.get("samples", TEST_SAMPLES)
    VERBOSE = kwargs.get("verbose",VERBOSE)
    VALIDATION_SPLIT = kwargs.get("validation_split",VALIDATION_SPLIT)
    
    

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