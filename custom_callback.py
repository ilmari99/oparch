import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class loss_callback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.loss_array = []
        self.loss_on_epoch_end = 100000000

    def on_train_batch_end(self, batch, logs=None):
        self.loss_array.append(logs["loss"])

    def on_epoch_end(self, epoch, logs=None):
        self.loss_on_epoch_end = logs["loss"]
        print(f"Loss array: {self.loss_array}")

    def plot_loss(self):
        plt.plot(range(len(self.loss_array)), self.loss_array)
        plt.show()