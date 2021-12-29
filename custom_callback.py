import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class loss_callback(tf.keras.callbacks.Callback):
    
    learning_metric = {
    "LAST_LOSS":None,
    "AVERAGE_LOSS_BATCH":None,
    "RELATIVE_IMPROVEMENT_EPOCH":None,
    "RELATIVE_IMPROVEMENT_BATCH":None,
    "VALIDATION_LOSS":None,
    "VALIDATION_ACCURACY":None,
    }
    
    def __init__(self, verbose = 0):
        super().__init__()
        self.verbose = verbose
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
        self.learning_metric["VALIDATION_LOSS"] = np.mean(self.loss_array_validation)
        self.learning_metric["VALIDATION_ACCURACY"] = np.mean(self.accuracy_array_validation)
        if(self.verbose > 0):
            print(f"ITEMS:{self.learning_metric.items()}")
            if(self.verbose == 2):
                self.plot_loss()

    def on_train_batch_end(self, batch, logs=None):
        self.loss_array_batch.append(logs["loss"])

    def on_epoch_end(self, epoch, logs=None):
        self.loss_array_epoch.append(logs["loss"])
        self.loss_array_validation.append(logs["val_loss"])
        self.accuracy_array_validation.append(logs["val_acc"])
            
    def plot_loss(self):
        plt.plot(range(len(self.loss_array_batch)), self.loss_array_batch)
        plt.xlabel("Batches")
        plt.ylabel("Loss on batch")
        plt.show()