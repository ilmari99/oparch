import tensorflow as tf
IMAGE_SIZE = (180,180)
BATCH_SIZE = 16
ACTIVATION_FUNCTIONS = {
    "relu":tf.keras.activations.relu,
    "tanh":tf.keras.activations.tanh,
    "sigmoid":tf.keras.activations.sigmoid,
    "elu":tf.keras.activations.elu,
    "exponential":tf.keras.activations.exponential,    
}

BINARY_LOSSES = {
    "binary_crossentropy":tf.keras.losses.BinaryCrossentropy,
}