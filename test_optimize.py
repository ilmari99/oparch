import oparch
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
def y_function(x) -> np.ndarray:
    y = np.zeros(max(np.shape(x))) #Assumes more training samples than features which should be true
    features = len(x[0])
    coeffs = np.random.uniform(low=-1, high=1, size=(1,features))[0]
    for i,row in enumerate(x):
        coeffs_randomness = np.random.uniform(low=0.8, high=1.2, size=(1,features))
        y[i] = np.sum(coeffs*row*coeffs_randomness[0], axis=0) #Add some randomness
    y = 1/(1+np.exp(-y)) #Sigmoid
    return y
X = np.random.rand(500,3)
y = y_function(X)
layers = [tf.keras.layers.Dense(1),tf.keras.layers.Dropout(0.1),tf.keras.layers.Dense(1,activation="sigmoid")]
model = tf.keras.models.Sequential(layers)
optimizer = tf.keras.optimizers.RMSprop()
loss=tf.keras.losses.MeanSquaredError()
(model,_) = oparch.opt_learning_rate(model, X, y,learning_rates=[0.01,0.1],optimizer=optimizer,loss=loss)
(model,_) = oparch.opt_activation(model, 0, X, y,return_model=True)
(model,_) = oparch.opt_decay(model, X, y,decays=[0.01,0.02])
(model,_) = oparch.opt_dense_units(model, 0, X, y,test_nodes=[6,12,1])
(model,_) = oparch.opt_learning_rate(model, X, y,learning_rates=[0.01,0.1])
cb_loss = oparch.LossCallback.LossCallback(samples=500)
hist = model.fit(
        X, y,
        epochs=5,
        verbose=1,
        batch_size=32,
        callbacks=[cb_loss],
        shuffle=True,
        use_multiprocessing=True,
    )
cb_loss.plot_loss()
plt.show() #requires a gui backend
