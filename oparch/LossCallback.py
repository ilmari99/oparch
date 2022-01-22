import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from . import configurations
import time

class LossCallback(tf.keras.callbacks.Callback):
    
    learning_metric = {
    "LAST_LOSS":None,
    "AVERAGE_LOSS_BATCH":None,
    "RELATIVE_IMPROVEMENT_EPOCH":None,
    "RELATIVE_IMPROVEMENT_BATCH":None,
    "VALIDATION_LOSS":None,
    "LAST_VALIDATION_LOSS":None,
    "VALIDATION_ACCURACY":None,
    }
    
    def __init__(self, **kwargs):
        super().__init__()
        epochs = kwargs.get("epochs",configurations.TEST_EPOCHS)
        verbose = kwargs.get("verbose",0)
        samples = kwargs.get("samples",0)
        self.verbose = verbose
        self.samples = samples
        self.current_epoch = 0
        self.epoch_start = 0
        if samples>0:
            self.batch_count = int(np.ceil((samples/configurations.BATCH_SIZE)))
            self.loss_array_epoch = np.zeros(epochs)
            self.loss_array_validation = np.zeros(epochs)
            self.accuracy_array_validation = np.zeros(epochs)
            self.loss_array_batch = np.zeros(self.batch_count * configurations.TEST_EPOCHS)
        else:
            print("It is recommended to specify sample size when creating the LossCallback")
            self.loss_array_epoch = []
            self.loss_array_validation = []
            self.accuracy_array_validation = []
            self.loss_array_batch = []
        
    def on_train_end(self, logs=None):
        self.learning_metric["LAST_LOSS"] = self.loss_array_epoch[-1]
        self.learning_metric["AVERAGE_LOSS_BATCH"] = np.mean(self.loss_array_batch)
        #Compare the difference in loss to the previous loss
        self.learning_metric["RELATIVE_IMPROVEMENT_EPOCH"] = np.mean(np.diff(self.loss_array_epoch)/self.loss_array_epoch[0:-1])
        self.learning_metric["RELATIVE_IMPROVEMENT_BATCH"] = np.mean(np.diff(self.loss_array_batch)/self.loss_array_batch[0:-1])
        if "VALIDATION" in configurations.LEARNING_METRIC:
            self.learning_metric["VALIDATION_LOSS"] = np.mean(self.loss_array_validation)
            self.learning_metric["LAST_VALIDATION_LOSS"] = self.loss_array_validation[-1]
            self.learning_metric["VALIDATION_ACCURACY"] = np.mean(self.accuracy_array_validation)
        if(self.verbose > 0):
            print(f"ITEMS:{self.learning_metric.items()}")
            if(self.verbose == 2):
                self.plot_loss()

    def on_train_batch_end(self, batch, logs=None):
        if self.samples == 0:
            self.loss_array_batch.append(logs["loss"])
        else:
            self.loss_array_batch[batch + self.current_epoch*self.batch_count] = logs["loss"]
            
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self.samples == 0:
            self.loss_array_epoch.append(logs["loss"])
            if "VALIDATION" in configurations.LEARNING_METRIC:
                self.loss_array_validation.append(logs["val_loss"])
                self.accuracy_array_validation.append(logs["val_accuracy"])
        else:
            self.loss_array_epoch[epoch] = logs["loss"]
            if "VALIDATION" in configurations.LEARNING_METRIC:
                self.loss_array_validation[epoch] = logs["val_loss"]
                self.accuracy_array_validation[epoch] = logs["val_accuracy"]
        if self.verbose == 2:
            elaps_time = time.time() - self.epoch_start
            print(f"Elapsed time in epoch: {elaps_time}")
            
    def plot_loss(self):
        plt.figure()
        plt.plot(range(len(self.loss_array_batch)), self.loss_array_batch)
        plt.xlabel("Batches")
        plt.ylabel("Loss on batch")
        #plt.show()