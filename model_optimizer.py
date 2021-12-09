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
        #tf.keras.layers.Dense(1,activation="sigmoid")
              ]
    return layer_list

##TODO Should create a class OptimizedModel to more easily set optimized variables of the model
def get_model(x_train, y_train):
    layer_list = [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1,activation="sigmoid")
              ]
    model = tf.keras.models.Sequential(layer_list)
    best_loss = mot.test_learning_speed(model,x_train,y_train)
    best_layers = layer_list
    
    print(f"Loss on epoch end is {best_loss}")
    dense_args, loss = get_best_dense_args(x_train, y_train)
    print(f"Best dense args:\nnodes: {dense_args[0]}\nActivation: {dense_args[1]}")
    if(loss<best_loss):
        new_layer = get_dense_layer(dense_args)
        layer_list = get_last_layers()
        layer_list.insert(1, new_layer)
        best_layers = layer_list
        best_loss = loss
    print(f"Best layers: {best_layers}")
    return tf.keras.models.Sequential(best_layers)

def get_best_dense_args(x_train, y_train):
    nodes = [2**i for i in range(0,6)]
    activations = list(constants.ACTIVATION_FUNCTIONS.values())
    best_loss = 100
    best_dense_args=[]
    for activation in activations:
        for node_amount in nodes:
            dense_args=[node_amount,activation]
            layer = get_dense_layer(dense_args)
            
            layer_list = get_last_layers() #Requires a Flatten layer at index 0
            
            layer_list.insert(1,layer)
            model = tf.keras.models.Sequential(layer_list)
            print(f"Nodes: {dense_args[0]}\nActivation: {dense_args[1]}.......")
            loss = mot.test_learning_speed(model,x_train,y_train,samples=800)
            print(f"Loss on validation set is {loss}")
            if loss<best_loss:
                best_loss = loss
                best_dense_args=dense_args
                print(f"This loss is currently the lowest loss.")
            print("\n\n")
    return best_dense_args, best_loss