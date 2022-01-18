import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from oparch import optimize as opt
from sklearn.model_selection import train_test_split
from oparch import configurations
from oparch import LossCallback
import pandas as pd
configurations.configure(TEST_EPOCHS = 10,samples=5000)
abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])
abalone_features = abalone_train.copy()
abalone_labels = abalone_features.pop("Age")
X = np.array(abalone_features)
y = np.array(abalone_labels)
print(f"Abalone samples: {np.shape(X)}")
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
#(model, loss_lossfun) = opt.opt_loss_fun(model, X, y)
(model, loss_lr) = opt.opt_learning_rate(model, X, y)
(model, loss_decay) = opt.opt_decay(model, X, y)
(model, loss_act) = opt.opt_activation(model, len(model.layers)-1, X, y)
model.load_weights("ONLY_TEST.h5")
index = 0
(model, loss_act) = opt.opt_activation(model, index, X, y)
model.load_weights("ONLY_TEST.h5")
(model, loss_units) = opt.opt_dense_units(model, index, X, y)
if model == None:
    index = index - 1
else:
    index = index + 1
(model, loss_act) = opt.opt_activation(model, index, X, y)
(model, loss_units) = opt.opt_dense_units(model, index, X, y)
if model == None:
    index = index - 1
else:
    index = index + 1
(model, loss_act) = opt.opt_activation(model, index, X, y)
(model, loss_units) = opt.opt_dense_units(model, index, X, y)
print(model.optimizer.weights)
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

