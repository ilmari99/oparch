import numpy as np
import tensorflow as tf
import model_optimizer_tools as mot
import custom_callback as ccb
import constants

def get_dense_layer(args):
    """args = list = [number of neurons(int), activation function"""
    return tf.keras.layers.Dense(args[0],activation=args[1])

def get_conv_layer(args):
    '''args = [filters, kernel_size(tuple), activation]'''
    return tf.keras.layers.Conv2D(args[0],args[1],activation=args[3])


def get_last_layers():
    layer_list = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1,activation="sigmoid")
              ]
    return layer_list

def get_model(x_train, y_train):
    layer_list = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1,activation="sigmoid")
              ]
    model = tf.keras.models.Sequential(layer_list)
    model.build(np.shape(x_train))
    best_loss = mot.test_learning_speed(model,x_train,y_train)
    best_layers = layer_list
    print(f"Loss on epoch end is {best_loss}")

    nodes = [2**i for i in range(0,6)]
    activations = list(constants.ACTIVATION_FUNCTIONS.values())
    for activation in activations:
        for node_amount in nodes:
            layer = get_dense_layer([node_amount,activation])
            layer_list = get_last_layers()
            layer_list.insert(1,layer)
            model = tf.keras.models.Sequential(layer_list)
            model.build(np.shape(x_train))
            loss = mot.test_learning_speed(model,x_train,y_train)
            print(f"Nodes: {node_amount}\nActivation: {activation}")
            print(f"Loss on epoch end is {loss}")
            if loss>best_loss:
                best_loss = loss
                best_layers = layer_list
                print(f"This loss is currently the lowest loss.")
            print("\n\n")


    # while:
    # Add a new layer(args).
    # Create a list with different args as objects.
    # Test the learning speed of the network.
    # Save the learning speeds to a list.
    # if max(learning_speed_list) > current_learning_speed:
    #     Add the corresponding layer to the model.
    return tf.keras.models.Sequential(best_layers)