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
    layers = [tf.keras.layers.Dense(16),
          tf.keras.layers.Dense(8),
          tf.keras.layers.Dense(2),
          tf.keras.layers.Dense(1)]
else:
    layers = [tf.keras.layers.Dense(1),
          tf.keras.layers.Dense(1),
          tf.keras.layers.Dense(1),
          tf.keras.layers.Dense(1)]
configs = opt.utils.add_seed_configs(opt.utils.get_layers_config(layers))#To add random seed 42 to each layer
layers = opt.utils.layers_from_configs(layers, configs)
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=tf.keras.losses.MeanSquaredError())

cb_loss = opt.LossCallback.LossCallback(early_stopping = False)
hist = model.fit(
        X, y,
        epochs=10,
        verbose=1,
        validation_data=(X_test,y_test),
        batch_size=16,
        callbacks=[cb_loss],
        shuffle=True,
        use_multiprocessing=True,
)
opt.utils.print_model(model,learning_metrics=cb_loss.learning_metric)
y_pred = model.predict(X_test)
plt.figure()
plt.plot(range(len(y_pred)),y_pred)
plt.plot(range(len(y_test)),y_test)
cb_loss.plot_loss()
layers = opt.utils.get_copy_of_layers(layers)
model = tf.keras.models.Sequential(layers)
model.build(np.shape(X))
model.compile(optimizer=tf.keras.optimizers.Adam(),loss="mse")
opt.set_default_misc(epochs=1,batch_size=16,learning_metric="LAST_LOSS",verbose=0, decimals=5)
opt.set_default_intervals(rho=list(np.linspace(0.8,1,10,endpoint=True)))
#model = opt.opt_loss_fun(model, X, y) #The fastest descending loss is logarithmic, so don't do this for better comparison plot
model = opt.opt_all_layer_params(model, X, y, "units")
model = opt.opt_all_layer_params(model, X, y, "activation")
model = opt.opt_optimizer_parameter(model, X, y, "learning_rate")
model = opt.opt_optimizer_parameter(model, X, y, "rho")
model = opt.opt_optimizer_parameter(model, X, y, "decay")
model = opt.opt_optimizer_parameter(model, X, y, "momentum")
model = opt.opt_optimizer_parameter(model, X, y, "amsgrad")
cb_loss = opt.LossCallback.LossCallback(early_stopping = False)
hist = model.fit(
        X, y,
        epochs=10,
        verbose=1,
        validation_data=(X_test,y_test),
        batch_size=16,
        callbacks=[cb_loss],
        shuffle=True,
)
cb_loss.plot_loss(new_figure=False)
opt.utils.print_model(model,learning_metrics=cb_loss.learning_metric)
y_pred = model.predict(X_test)
plt.figure()
plt.plot(range(len(y_pred)),y_pred)
plt.plot(range(len(y_test)),y_test)
plt.show()

