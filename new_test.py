import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from oparch import optimize as opt
from sklearn.model_selection import train_test_split
from oparch import configurations
from oparch import LossCallback
def y_function(x) -> np.ndarray:
    y = np.zeros(max(np.shape(x))) #Assumes more training samples than features which should be true
    features = len(x[0])
    coeffs = np.array([(c+1)*0.5 for c in range(features)])
    for i,row in enumerate(x):
        coeffs_randomness = np.random.uniform(low=0.1, high=2, size=(1,features))
        y[i] = np.sum(coeffs*row*coeffs_randomness[0], axis=0) #Add some randomness
        #y = 1/(1+np.exp(-y))
    return y

np.random.seed(seed=42) #for reproducibility
X = np.random.rand(600,10)
y = y_function(X)

layers = [tf.keras.layers.Dense(1),tf.keras.layers.Dropout(0.1),tf.keras.layers.Dense(1)]
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.RMSprop())
X,X_test,y,y_test = train_test_split(X,y,test_size=0.1)
hist = model.fit(
        X, y,
        epochs=60,
        verbose=1,
        validation_data=(X_test,y_test),
        batch_size=configurations.BATCH_SIZE,
        #callbacks=[cb_loss],
        shuffle=True,
        use_multiprocessing=True,
    )
y_pred = model.predict(X_test)
print(y_pred[:9])
print(y_test[:9])
plt.plot(range(len(y_pred)),y_pred)
plt.plot(range(len(y_test)),y_test)
plt.figure(2)
model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.RMSprop())
(decay,loss) = opt.opt_decay(model, X, y)
print(decay,loss)
model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.RMSprop(decay=decay))
(lr, loss_lr) = opt.opt_learning_rate(model, X, y)
model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.RMSprop(lr,decay=decay))
print(lr,loss_lr)
index = 0
(config, act_loss) = opt.opt_activation(model, index, X, y)
new_layer = tf.keras.layers.Dense.from_config(config)
layers[0] = new_layer
layers = opt.get_copy_of_layers(layers)
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.RMSprop(lr,decay=decay))
(config, loss_units) = opt.opt_dense_units(model, 0, X, y)
if config != None:
    new_layer = tf.keras.layers.Dense.from_config(config)
    layers[0] = new_layer
else:
    layers.pop(0)
layers = opt.get_copy_of_layers(layers)
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.RMSprop(lr,decay=decay))

index = 2
if config==None:
    index = 1
(config, loss_units) = opt.opt_dense_units(model, index, X, y)
if config != None:
    new_layer = tf.keras.layers.Dense.from_config(config)
    layers[index] = new_layer
else:
    layers.pop(index)
layers = opt.get_copy_of_layers(layers)
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(loss=tf.keras.losses.MeanSquaredError(),optimizer=tf.keras.optimizers.RMSprop(lr,decay=decay))

model.summary()
hist = model.fit(
        X, y,
        epochs=15,
        verbose=1,
        validation_data=(X_test,y_test),
        batch_size=configurations.BATCH_SIZE,
        #callbacks=[cb_loss],
        shuffle=True,
        use_multiprocessing=True,
    )
y_pred = model.predict(X_test)
print(y_pred)
plt.plot(range(len(y_pred)),y_pred)
plt.plot(range(len(y_test)),y_test)
plt.show()