import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import zipfile
import numpy as np
import tensorflow as tf
from OptimizedModel import OptimizedModel
import custom_callback as ccb
import process_data_tools as pdt
import model_optimizer as mod_op
import model_optimizer_tools as mot
import constants
import random
import matplotlib.pyplot as plt
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def get_happiness_xy():
    """Reads a local csv file and returns np arrays. Removes all rows with NaN values.

    Returns:
        numeric_norm [np.ndarray]: normalized feature data
        happiness_norm [np.ndarray]: normalized happiness data
        happiness [np.ndarray]: happiness data
    """    
    csv_data = pd.read_csv("C:\\Users\\ivaht\\Downloads\\PersonalData.csv")
    happiness = csv_data.pop("How good was the day (1-10)")
    csv_data.__delitem__("Date")
    numeric_features = ['How busy (1-10)','Weight (kg)','How drunk','Nicotine (mg)','Studying (hours)','Sleep (hours)','Time spent with people (hours)']
    numeric_data = csv_data[numeric_features]
    #tf.convert_to_tensor(numeric_data)
    numeric_data = numeric_data.to_numpy()
    happiness = happiness.to_numpy()

    indices_to_drop = []
    for i,row in enumerate(numeric_data):
        if(np.isnan(np.sum(row))):
            indices_to_drop.append(i)

    numeric_data = np.delete(numeric_data, indices_to_drop,0)
    happiness = np.delete(happiness, indices_to_drop,0)
    
    #normalize the values by column
    happiness_norm = happiness / happiness.max(axis=0)
    numeric_norm = numeric_data / numeric_data.max(axis=0)
    return numeric_norm, happiness_norm, happiness

def y_function(x) -> np.ndarray:
    y = np.zeros(max(np.shape(x))) #Assumes more training samples than features which should be true
    print(f"{np.shape(y)}")
    features = len(x[0])
    coeffs = np.array([(c+1)*0.5 for c in range(features)])
    for i,row in enumerate(x):
            y[i] = np.sum(coeffs*row, axis=0) + random.uniform(-0.2,0.2) #Add some randomness to better see overfitting
    return y

if __name__ == '__main__':

    #train_generator, validation_generator = own_funs.get_image_generators_from_path("C://Users//ivaht//Downloads//aalto_lut_train_validation//aalto_lut_training",
    #                                                       "C://Users//ivaht//Downloads//aalto_lut_train_validation//aalto_lut_validation")
    #x,y, happiness = get_happiness_xy()
    np.random.seed(seed=42)
    
    x = np.random.rand(300,4) #300 samples with 4 features
    y = y_function(x)
    
    x_max = x.max(axis=0) #Max value by column
    y_max = y.max(axis=0)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)
    
    x_train_norm = x_train / x_max
    y_train_norm = y_train / y_max
    x_test_norm = x_test / x_max
    y_test_norm = y_test / y_max
    
    cb1 = ccb.loss_callback()
    layers = [ 
              tf.keras.layers.Dense(1,activation = "tanh"),
              tf.keras.layers.Dense(1),
              ]
    
    print(f"Config:{layers[0].get_config()}")
    
    optimized_model = mod_op.get_optimized_model(x_train_norm, y_train_norm, layers)
    
    model = optimized_model.model
    learning_speed = mot.test_learning_speed(tf.keras.models.clone_model(model),x_train_norm,y_train_norm)
    OptimizedModel.build_and_compile(model,np.shape(x_train))
    model.summary()
    model.fit(
        x_train_norm, y_train_norm,
        epochs=30,
        verbose=2,
        batch_size=constants.BATCH_SIZE,
        validation_data=(x_test_norm, y_test_norm),
        callbacks=[cb1],
        shuffle=True,
    )
    print(f"MODEL INFO (learning_speed={learning_speed}):\nOptimizer: {optimized_model.optimizer.get_config()}\n")
    print(f"Layers: {optimized_model.layer_configs}\n")
    print(f"Loss function: {optimized_model.loss_fun}\n")
    results = model.evaluate(x_test_norm, y_test_norm, constants.BATCH_SIZE)
    print(f"Test_loss: {results[0]} Test acc: {results[1]}")
    predictions = model.predict(x_test_norm)
    predictions = np.mean(predictions, axis=1)*y_max #Denormalize
    print("Predictions: ", predictions)
    print("Y actual: ", y_test)
    plt.plot(list(range(len(predictions))), predictions,label = "predictions")
    plt.plot(list(range(len(predictions))), y_test, label = "actual")
    plt.figure(2)
    cb1.plot_loss()
    plt.show()
    
    