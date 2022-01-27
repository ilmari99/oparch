if __name__ == "__main__":
    from pathlib import Path
    import sys
    path_root = Path(__file__).parents[1]
    sys.path.append(str(path_root))
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import oparch as opt
from sklearn.model_selection import train_test_split
from oparch import configurations
from oparch import LossCallback
import pandas as pd
import time
#configurations.configure(TEST_EPOCHS = 10,samples=5000)TODO
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
layers = [tf.keras.layers.Dense(16,activation="relu"),
          tf.keras.layers.Dense(8,activation="relu"),
          tf.keras.layers.Dense(2,activation="relu"),
          tf.keras.layers.Dense(1,activation="relu")] #A typical structure
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError())
opt.opt_dense_units(model, 2, X, y)
opt.opt_dense_units2(model, 2, X, y)
exit()
cb_loss = opt.LossCallback.LossCallback()
hist = model.fit(
        X, y,
        epochs=10,
        verbose=1,
        validation_data=(X_test,y_test),
        batch_size=configurations.BATCH_SIZE,
        callbacks=[cb_loss],
        shuffle=True,
        use_multiprocessing=True,
)
cb_loss.plot_loss()
opt.utils.print_model(model,learning_metrics=cb_loss.learning_metric)
y_pred = model.predict(X_test)
plt.figure()
plt.plot(range(len(y_pred)),y_pred)
plt.plot(range(len(y_test)),y_test)
#layers = [tf.keras.layers.Dense(16,activation="relu"),
#          tf.keras.layers.Dense(8,activation="relu"),
#          tf.keras.layers.Dense(2,activation="relu"),
#          tf.keras.layers.Dense(1,activation="relu")]
layers = [tf.keras.layers.Dense(1),
          tf.keras.layers.Dense(1),
          tf.keras.layers.Dense(1),
          tf.keras.layers.Dense(1)]
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError())
#No loss optimization for better comparibility
model = opt.opt_learning_rate(model, X, y)
model = opt.opt_decay(model, X, y)
model = opt.opt_activation(model, len(model.layers)-1, X, y)
index = 0
model = opt.opt_activation(model, index, X, y)
model = opt.opt_dense_units(model, index, X, y)
if model == None:
    index = index - 1
else:
    index = index + 1
model = opt.opt_activation(model, index, X, y)
model = opt.opt_dense_units(model, index, X, y)
if model == None:
    index = index - 1
else:
    index = index + 1
model = opt.opt_activation(model, index, X, y)
model = opt.opt_dense_units(model, index, X, y)
cb_loss = opt.LossCallback.LossCallback()
hist = model.fit(
        X, y,
        epochs=10,
        verbose=1,
        validation_data=(X_test,y_test),
        batch_size=configurations.BATCH_SIZE,
        callbacks=[cb_loss],
        shuffle=True,
        use_multiprocessing=True,
)
cb_loss.plot_loss()
opt.utils.print_model(model,learning_metrics=cb_loss.learning_metric)
y_pred = model.predict(X_test)
plt.figure()
plt.plot(range(len(y_pred)),y_pred)
plt.plot(range(len(y_test)),y_test)
plt.show()

