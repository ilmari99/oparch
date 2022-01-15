import tensorflow as tf
import numpy as np
from oparch import optimize as opt
def y_function(x) -> np.ndarray:
    y = np.zeros(max(np.shape(x))) #Assumes more training samples than features which should be true
    features = len(x[0])
    coeffs = np.array([(c+1)*0.5 for c in range(features)])
    for i,row in enumerate(x):
        coeffs_randomness = np.random.uniform(low=-0.5, high=0.5, size=(1,features))
        y[i] = np.sum(coeffs*row*coeffs_randomness[0], axis=0) #Add some randomness
        y = 1/(1+np.exp(-y))
    return y

np.random.seed(seed=42) #for reproducibility
X = np.random.rand(300,4)
y = y_function(X)

layers = [tf.keras.layers.Dense(1),tf.keras.layers.Dropout(0.1),tf.keras.layers.Dense(1)]
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.SGD(0.1))
(configuration, metric) = opt.opt_learning_rate(model, X, y)
print(configuration, metric)
#(configuration, metric) = mod_op.opt_dense(model, 0, X, y)
#print(configuration)
#print(metric)
