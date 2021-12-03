import numpy as np
import tensorflow as tf
import model_optimizer_tools as mot
import custom_callback as ccb

def get_dense_layer(args):
    return tf.keras.layers.Dense(args[0],activation=args[1])



def get_last_layers():
    layer_list = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1,activation="sigmoid")
              ]
    return layer_list

def get_model(x_train, y_train):
    layer_list = get_last_layers()
    model = tf.keras.models.Sequential(layer_list)
    model.build(np.shape(x_train))
    best_loss = mot.test_learning_speed(model,x_train,y_train)
    best_layers = layer_list
    print(f"Loss on epoch end is {best_loss}")

    nodes = [2**i for i in range(0,6)]
    activations = [tf.keras.activations.relu,
                   tf.keras.activations.tanh,
                   tf.keras.activations.sigmoid,
                   tf.keras.activations.elu,
                   tf.keras.activations.exponential,
                   ]
    for activation in activations:
        for node_amount in nodes:
            layer = get_dense_layer([node_amount,activation])
            layer_list = [tf.keras.layers.Flatten(),
                          layer,
                          tf.keras.layers.Dense(1,activation="sigmoid")]
            model = tf.keras.models.Sequential(layer_list)
            model.build(np.shape(x_train))
            loss = mot.test_learning_speed(model,x_train,y_train)
            model.summary()
            print(f"Nodes: {node_amount}\nActivation: {activation}")
            print(f"Loss on epoch end is {loss}")
            if loss<best_loss:
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