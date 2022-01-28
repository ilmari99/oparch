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
other_test=False
if other_test:
    layers = [tf.keras.layers.Dense(16,activation="relu"),
          tf.keras.layers.Dense(8,activation="relu"),
          tf.keras.layers.Dense(2,activation="relu"),
          tf.keras.layers.Dense(1,activation="relu")]
else:
    layers = [tf.keras.layers.Dense(1),
          tf.keras.layers.Dense(1),
          tf.keras.layers.Dense(1),
          tf.keras.layers.Dense(1)]
configs = opt.utils.add_seed_configs(opt.utils.get_layers_config(layers))#To add random seed 42 to each layer
opt.utils.layers_from_configs(layers, configs)
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError())

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
layers = opt.utils.get_copy_of_layers(layers)
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError())
model = opt.opt_learning_rate(model, X, y)
model = opt.opt_decay(model, X, y)
model = opt.opt_all_units(model, X, y)
indices = opt.utils.get_dense_indices(model)
for i in indices:
    model = opt.opt_activation(model, i, X, y)
model = opt.opt_learning_rate(model, X, y)
model = opt.opt_decay(model, X, y)
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

