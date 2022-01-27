import numpy as np
import oparch

def y_function(x) -> np.ndarray:
    y = np.zeros(max(np.shape(x))) #Assumes more training samples than features which should be true
    features = len(x[0])
    coeffs = np.random.uniform(low=-1, high=1, size=(1,features))[0]
    for i,row in enumerate(x):
        coeffs_randomness = np.random.uniform(low=0.8, high=1.2, size=(1,features))
        y[i] = np.sum(coeffs*row*coeffs_randomness[0], axis=0) #Add some randomness
    y = 1/(1+np.exp(-y)) #Sigmoid
    return y

def get_xy(samples=10, features=3,categorical=False,):
    X = np.random.rand(samples,features)
    y = y_function(X)
    return X,y