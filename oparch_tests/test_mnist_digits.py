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
X = X.astype('float32') / 255
X_test = X_test.astype('float32') / 255
X = X/255
X_test = X_test/255
layers = [
    Conv2D(32,(3,3),activation=tf.keras.activations.relu),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3)),
    MaxPooling2D((2,2)),
    Conv2D(16,(3,3)),
    Flatten(),
    Dense(64,activation=tf.keras.activations.relu),
    Dense(16),
    Dense(10,activation=tf.keras.activations.softmax),
]
layers = opt.utils.get_copy_of_layers(layers)

model = tf.keras.models.Sequential(layers)
model.build(X.shape)
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=["accuracy"])
cb = opt.LossCallback.LossCallback()
model.fit(
    X,y,
    epochs=1,
    batch_size=128,
    callbacks=[cb],
    validation_data=(X_test,y_test)
)
opt.utils.print_model(model,learning_metrics=cb.learning_metric)
cb.plot_loss(new_figure=False)

configurations.configure(samples=6000,epochs=1,batch_size=128,learning_metric="NEG_ACCURACY",verbose=0)
model = opt.opt_loss_fun(model, X, y,categorical=True,return_metric="RELATIVE_IMPROVEMENT_EPOCH")
model = opt.opt_all_activations(model, X, y)
model = opt.opt_all_units(model, X, y)
model = opt.opt_learning_rate(model, X, y)
model = opt.opt_decay(model, X, y)
model.compile(optimizer=model.optimizer,loss=model.loss,metrics=["accuracy"]) #Must be compiled again if you want to add metrics
cb = opt.LossCallback.LossCallback()
hist = model.fit(
    X,y,
    epochs=1,
    batch_size=128,
    callbacks=[cb],
    validation_data=(X_test,y_test)
)
cb.plot_loss(show=True,new_figure=False)
opt.utils.print_model(model,learning_metrics=cb.learning_metric)


