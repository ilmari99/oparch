import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class loss_callback(tf.keras.callbacks.Callback):
    loss_array = []
    loss_on_epoch_end = 100000000
    def on_train_batch_end(self, batch, logs=None):
        self.loss_array.append(logs["loss"])

    def on_epoch_end(self, epoch, logs=None):
        self.loss_on_epoch_end = logs["loss"]

    def plot_loss(self):
        plt.plot(range(len(self.loss_array)), self.loss_array)
        plt.show()