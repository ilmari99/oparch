if __name__ == "__main__":
    from pathlib import Path
    import sys
    path_root = Path(__file__).parents[1]
    sys.path.append(str(path_root))
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import oparch as opt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from oparch import configurations
from oparch import LossCallback
import pandas as pd
import time
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
(X,y),(X_test,y_test) = mnist.load_data()
X = X.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)
#X = X.astype('float32') / 255
#X_test = X_test.astype('float32') / 255
X = X/255
X_test = X_test/255
'''
layers = [
    Conv2D(64,(5,5),strides=(1,1)),
    MaxPooling2D((2,1),strides=(2,1)),
    Conv2D(64, (3,3),strides=(1,1)),
    MaxPooling2D((2,1),strides=(2,2)),
    Conv2D(64,(3,3),strides=(1,1)),
    Flatten(),
    Dense(1),
    Dense(1),
    Dense(64,activation=tf.keras.activations.relu),
    Dense(64),
    Dense(10,activation=tf.keras.activations.softmax),
]
'''
layers = [
    Conv2D(32,(3,3)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3)),
    MaxPooling2D((2,2)),
    Conv2D(32,(3,3)),
    Flatten(),
    Dense(1),
    Dense(1),
    Dense(1,activation=tf.keras.activations.relu),
    Dense(1),
    Dense(10,activation=tf.keras.activations.softmax),
]
layers = opt.utils.get_copy_of_layers(layers)

model = tf.keras.models.Sequential(layers)
model.build(X.shape)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["accuracy"])
cb = opt.LossCallback.LossCallback(early_stopping=True)

model.fit(
    X,y,
    epochs=2,
    batch_size=128,
    callbacks=[cb],
    validation_data=(X_test,y_test)
)
opt.utils.print_model(model,learning_metrics=cb.learning_metric)
cb.plot_loss(new_figure=False, show=False)

opt.set_default_misc(samples=6000,epochs=1,batch_size=128,learning_metric="NEG_ACCURACY",verbose=0)
#model = opt.opt_loss_fun(model, X, y,categorical=True)
model = opt.opt_optimizer_parameter(model, X, y, ["learning_rate","decay","momentum","rho"])
model = opt.opt_all_layer_params(model, X, y, "filters")
model = opt.opt_all_layer_params(model, X, y, "units")
#model = opt.opt_all_layer_params(model, X, y, "pool_size")
#model = opt.opt_all_layer_params(model, X, y, "kernel_size")
model = opt.opt_all_layer_params(model, X, y, "activation")
model = opt.opt_optimizer_parameter(model, X, y, ["learning_rate","decay","momentum","rho"])


model.compile(optimizer=model.optimizer,loss=model.loss,metrics=["accuracy"]) #Must be compiled again if you want to add metrics
cb = opt.LossCallback.LossCallback()
hist = model.fit(
    X,y,
    epochs=2,
    batch_size=128,
    callbacks=[cb],
    validation_data=(X_test,y_test)
)
cb.plot_loss(show=True,new_figure=False)
plt.figure()
plt.show()
opt.utils.print_model(model,learning_metrics=cb.learning_metric)


