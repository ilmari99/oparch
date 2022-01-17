import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from oparch import optimize as opt
from sklearn.model_selection import train_test_split
from oparch import configurations
from oparch import LossCallback
import pandas as pd
np.set_printoptions(precision=3, suppress=True)
abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop('Age')
X = np.array(abalone_features)
y = np.array(abalone_labels)
print(f"Abalone samples: {len(y)}")
X,X_test,y,y_test = train_test_split(X,y,test_size=0.2)
layers = [tf.keras.layers.Dense(64),tf.keras.layers.Dense(1)] #A typical structure
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError())
hist = model.fit(
        X, y,
        epochs=10,
        verbose=1,
        validation_data=(X_test,y_test),
        batch_size=configurations.BATCH_SIZE,
        #callbacks=[cb_loss],
        shuffle=True,
        use_multiprocessing=True,
)
y_pred = model.predict(X_test)
plt.plot(range(len(y_pred)),y_pred)
plt.plot(range(len(y_test)),y_test)
plt.figure(2)
layers = [tf.keras.layers.Dense(1),tf.keras.layers.Dense(1),tf.keras.layers.Dense(1),tf.keras.layers.Dense(1)]
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError())
index = 0
(model, loss_units) = opt.opt_dense_units(model, index, X, y,return_model=True)
if model == None:
    index = index - 1
else:
    index = index + 1
(model, loss_units) = opt.opt_dense_units(model, index, X, y,return_model=True)
if model == None:
    index = index - 1
else:
    index = index + 1
(model, loss_units) = opt.opt_dense_units(model, index, X, y,return_model=True)
(lr, loss_lr) = opt.opt_learning_rate(model, X, y)
model.compile(optimizer=tf.keras.optimizers.Adam(lr),loss=tf.keras.losses.MeanSquaredError())
(decay,loss_decay) = opt.opt_decay(model, X, y)
model.compile(optimizer=tf.keras.optimizers.Adam(lr,decay=decay),loss=tf.keras.losses.MeanSquaredError())

hist = model.fit(
        X, y,
        epochs=10,
        verbose=1,
        validation_data=(X_test,y_test),
        batch_size=configurations.BATCH_SIZE,
        #callbacks=[cb_loss],
        shuffle=True,
        use_multiprocessing=True,
)
opt.print_optimized_model(model)
y_pred = model.predict(X_test)
plt.plot(range(len(y_pred)),y_pred)
plt.plot(range(len(y_test)),y_test)
plt.show()

