import numpy as np
import tensorflow as tf
import model_optimizer_tools as mot
import custom_callback as ccb

def get_last_layers():
    layer_list = [tf.keras.layers.Flatten(),
                  tf.keras.layers.Dense(1,activation="sigmoid")
              ]
    return layer_list

def get_model(x_train, y_train):
    layer_list = get_last_layers()
    model = tf.keras.models.Sequential(layer_list)

    model.build(np.shape(x_train))
    best_loss = mot.test_learning_speed(model,x_train,y_train)
    print(f"Loss on epoch end is {best_loss}")

    # while:
    # Add a new layer(args).
    # Create a list with different args as objects.
    # Test the learning speed of the network.
    # Save the learning speeds to a list.
    # if max(learning_speed_list) > current_learning_speed:
    #     Add the corresponding layer to the model.

    new_layer = tf.keras.layers.Conv2D(16, (3 ,3), activation="relu")
    layer_list = get_last_layers()
    layer_list.insert(0, new_layer)
    model = tf.keras.models.Sequential(layer_list)
    model.build(np.shape(x_train))
    best_loss = mot.test_learning_speed(model, x_train, y_train, samples=500)
    print(f"Loss on epoch end is {best_loss}.")

    return model