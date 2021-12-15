import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class loss_callback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.loss_array = []
        self.relative_improvement = []
        self.loss_on_epoch_end = 1000

    def on_train_batch_end(self, batch, logs=None):
        self.loss_array.append(logs["loss"])

    def on_epoch_end(self, epoch, logs=None):
        if(logs["loss"]>10000):
            self.loss_on_epoch_end = np.nan
        if(epoch==0):
            self.loss_on_epoch_end= logs["loss"]
        else:
            if(not isinstance(self.loss_on_epoch_end,list)):
                self.loss_on_epoch_end = [self.loss_on_epoch_end]
            self.loss_on_epoch_end.append(logs["loss"])
            self.relative_improvement.append(logs["loss"]/self.loss_on_epoch_end[0])

    def plot_loss(self):
        plt.plot(range(len(self.loss_array)), self.loss_array)
        plt.xlabel("Batches")
        plt.ylabel("Loss on batch")
        plt.show()