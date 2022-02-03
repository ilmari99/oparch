import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from . import configurations
import time

_learning_metrics = {
    "LAST_LOSS":None,
    "AVERAGE_LOSS_BATCH":None,
    "RELATIVE_IMPROVEMENT_EPOCH":None,
    "RELATIVE_IMPROVEMENT_BATCH":None,
    "VALIDATION_LOSS":None,
    "NEG_VALIDATION_ACCURACY":None,
    "NEG_ACCURACY":None,
    }

class LossCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, **kwargs):
        self.early_stopping = kwargs.get("early_stopping",True)
        self.learning_metric = _learning_metrics.copy()
        self.verbose = kwargs.get("verbose",configurations.VERBOSE)
        self.epoch_start = 0
        self.loss_array_epoch = []
        self.loss_array_validation = []
        self.accuracy_array = []
        self.accuracy_array_validation = []
        self.loss_array_batch = []
        
    def on_train_end(self, logs=None):
        self.learning_metric["LAST_LOSS"] = self.loss_array_epoch[-1]
        self.learning_metric["AVERAGE_LOSS_BATCH"] = np.mean(self.loss_array_batch)
        self.learning_metric["NEG_ACCURACY"] = -self.accuracy_array[-1] if self.accuracy_array[-1] is not None else None
        self.learning_metric["RELATIVE_IMPROVEMENT_BATCH"] = self.relative_diff_list(self.loss_array_batch)
        self.learning_metric["RELATIVE_IMPROVEMENT_EPOCH"] = self.relative_diff_list(self.loss_array_epoch)        
        self.learning_metric["VALIDATION_LOSS"] = self.loss_array_validation[-1] if self.loss_array_validation[-1] is not None else None
        if self.accuracy_array_validation and not None in self.accuracy_array_validation:
            self.learning_metric["NEG_VALIDATION_ACCURACY"] = -np.mean(self.accuracy_array_validation)
        if self.verbose > 0:
            print(f"ITEMS:{self.learning_metric.items()}")
            if(self.verbose >= 2):
                self.plot_loss(show=True)

    def on_train_batch_end(self, batch, logs=None):
        self.loss_array_batch.append(logs.get("loss"))
        if self.early_stopping and batch % 5 == 0:
            self.try_early_stop()
        
        
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.loss_array_epoch.append(logs.get("loss"))
        self.loss_array_validation.append(logs.get("val_loss"))
        self.accuracy_array_validation.append(logs.get("val_accuracy"))
        self.accuracy_array.append(logs.get("accuracy"))
        if self.verbose == 2:
            elaps_time = time.time() - self.epoch_start
            print(f"Elapsed time in epoch: {elaps_time}")
            print(logs)

    def plot_loss(self,show=False,new_figure=True):
        if new_figure:
            plt.figure()
        plt.plot(range(len(self.loss_array_batch)), self.loss_array_batch)
        plt.xlabel("Batches")
        plt.ylabel("Loss on batch")
        if show:
            plt.show()

    def relative_diff_list(cls,arr):
        #Compare the differences in array to the previous value
        if len(arr)>1 and not None in arr:
            diff_arr = np.mean(np.diff(arr)/arr[0:-1])
        else:
            return None
        return diff_arr
    
    def try_early_stop(self):
        k = self.relative_diff_list(self.loss_array_batch)
        if k is not None and k > 0.01:
            print(f"\nModel is not descending k>0.01  (k= {k}). Stopping training.")
            self.model.stop_training = True
        