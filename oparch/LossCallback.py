import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from . import configurations
import time
__lowest__ = float("inf")
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
    """
    Callback that is used to keep track of different parameters through the training of the model.
    inherits from tf.keras.callbacks.Callback.
    
    kwargs {early_stopping:Bool, verbose:int, }
    
    """    
    
    def __init__(self, **kwargs):
        """Initialises results as empty lists.
        kwargs {early_stopping:Bool, verbose:int, } 
        """        
        self.early_stopping = kwargs.get("early_stopping",True)
        self.learning_metric = _learning_metrics.copy()
        self.verbose = kwargs.get("verbose",configurations.get_default_misc("verbose"))
        self.epoch_start = 0
        self.loss_array_epoch = []
        self.loss_array_validation = []
        self.accuracy_array = []
        self.accuracy_array_validation = []
        self.loss_array_batch = []
        
    def on_train_end(self, logs=None):
        """Automaticall called after the model has been trained with model.fit.
        Creates values to the learning_metric dictionary.
        """        
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
        """Appends results to loss_array_batch after a training batch.
        If early_stopping = True, tests if the loss is decreasing after 20 batches.
        """        
        self.loss_array_batch.append(logs.get("loss"))
        if (batch+1) % 20 == 0 and self.early_stopping:
            self.try_early_stop()
        
        
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.
        Starts a timer.
        """        
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """Called after en epoch. Adds values to different arrays to keep track of epoch specific values.
        If verbose == 2, prints elapsed time and logs after each epoch.
        """        
        self.loss_array_epoch.append(logs.get("loss"))
        self.loss_array_validation.append(logs.get("val_loss"))
        self.accuracy_array_validation.append(logs.get("val_accuracy"))
        self.accuracy_array.append(logs.get("accuracy"))
        if self.verbose == 2:
            elaps_time = time.time() - self.epoch_start
            print(f"Elapsed time in epoch: {elaps_time}")
            print(logs)

    def plot_loss(self,show=False,new_figure=True):
        """plots loss/batch, with batch number on x -axis and loss on batch on y-axis

        Args:
            show (bool, optional): Shows the plot immediately. Defaults to False.
            new_figure (bool, optional): Creates a new figure. Defaults to True.
        """        
        if new_figure:
            plt.figure()
        plt.plot(range(len(self.loss_array_batch)), self.loss_array_batch)
        plt.xlabel("Batches")
        plt.ylabel("Loss on batch")
        if show:
            plt.show()
            
            
    @classmethod
    def relative_diff_list(cls,arr):
        """Returns the mean difference of elements in x, for example x2 - x1.
        If len(arr)<1 or None in arr, returns None

        Args:
            arr list,np.ndarray: array to count differences with

        Returns:
            None or float
        """        
        #Compare the differences in array to the previous value
        if len(arr)>1 and not None in arr:
            diff_arr = np.mean(np.diff(arr)/arr[0:-1])
        else:
            return None
        return diff_arr
    
    
    def try_early_stop(self, threshold=0.1):
        """
        Checks if the instances loss is decreasing by calculating the differences of its loss_on_batch vector.
        
        threshold(optional): if the mean slope of the calculated loss_on_batch is > threshold, stops training
        """
        k = self.relative_diff_list(self.loss_array_batch)
        isnan = [np.isnan(loss) for loss in self.loss_array_batch]
        if k is not None and (k > threshold or any(isnan)):
            print(f"\nModel is not descending k>{threshold}  (k= {k}). Stopping training.")
            self.model.stop_training = True
        self.early_stopping = False