import numpy as np
import tensorflow as tf
import model_optimizer_tools as mot
import custom_callback as ccb
import constants
from OptimizedModel import OptimizedModel

def get_dense_layer(args: list):
    """args = list = [number of neurons(int), activation function]"""
    return tf.keras.layers.Dense(args[0],activation=args[1])

def get_conv2d_layer(args: list):
    '''args = [filters, kernel_size(tuple), activation]'''
    return tf.keras.layers.Conv2D(args[0],args[1],activation=args[3])


def get_model(x_train: np.ndarray, y_train: np.ndarray) -> OptimizedModel:
    optimized_model = OptimizedModel([tf.keras.layers.Dense(1,constants.ACTIVATION_FUNCTIONS["tanh"])],x_train)
    
    optimized_model.loss = mot.test_learning_speed(optimized_model.get_model(),x_train,y_train)
    best_layers = optimized_model.layers
    
    print(f"Loss on epoch end is {optimized_model.loss}")
    dense_args, loss = get_best_dense_args(optimized_model, x_train, y_train)
    print(f"Best dense args:\nnodes: {dense_args[0]}\nActivation: {dense_args[1]}")
    if(loss<optimized_model.loss):
        new_layer = get_dense_layer(dense_args)
        optimized_model.layers.insert(1, new_layer)
        optimized_model.loss = loss
        print(f"Best layers: {optimized_model.layers}")
    return optimized_model

def get_best_dense_args(optimized_model, x_train, y_train):
    nodes = [2**i for i in range(0,6)] #Node amounts to test
    activations = list(constants.ACTIVATION_FUNCTIONS.values())
    best_loss = 10000
    best_dense_args=[]
    for activation in activations:
        for node_amount in nodes:
            dense_args = [node_amount,activation]
            layer = get_dense_layer(dense_args)
            layer_list = optimized_model.get_layers() #Copy of layers
            layer_list.insert(1,layer)
            model = tf.keras.models.Sequential(layer_list)
            OptimizedModel.build_and_compile(model,np.shape(x_train))
            
            print(f"Nodes: {dense_args[0]}\nActivation: {dense_args[1]}.......")
            loss = mot.test_learning_speed(model,x_train,y_train,samples=800)
            print(f"{constants.LEARNING_METRIC} {loss}")
            if loss<best_loss:
                best_loss = loss
                best_dense_args=dense_args
                print(f"This loss is currently the lowest loss.")
            print("\n\n")
    return best_dense_args, best_loss