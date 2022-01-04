import tensorflow as tf
IMAGE_SIZE = (180,180)
BATCH_SIZE = 16
LEARNING_METRIC = "LAST_LOSS" #LAST_LOSS seems to work best
TEST_EPOCHS = 15

ACTIVATION_FUNCTIONS = {
    "sigmoid":tf.keras.activations.sigmoid,
    "linear":None,
    "tanh":tf.keras.activations.tanh,
    "exponential":tf.keras.activations.exponential,
    "relu":tf.keras.activations.relu,
    "elu":tf.keras.activations.elu,
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
}

OPTIMIZERS = {
    "adadelta":tf.keras.optimizers.Adadelta(),
    "adagrad":tf.keras.optimizers.Adagrad(),
    "adam":tf.keras.optimizers.Adam(),
    "Adamax":tf.keras.optimizers.Adamax(),
    "rmsprop":tf.keras.optimizers.RMSprop(),
    "sgd":tf.keras.optimizers.SGD(),
}

BINARY_LOSSES = {
    "binary_crossentropy":tf.keras.losses.BinaryCrossentropy,
}