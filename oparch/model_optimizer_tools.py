from .OptimizedModel import OptimizedModel
from . import LossCallback as lcb
import tensorflow as tf
import numpy as np
from . import configurations
import time
from sklearn.model_selection import train_test_split

def test_learning_speed(model, x_train, y_train,samples=500, validation_split=0.2):
    """Tests the learning speed of the model.
       Returns configurations.LOSS_METRIC from LossCallback

    Args:
        model (Sequential): Doesn't have to be built
        x_train (np.array): feature data
        y_train (np.array): feature labels
        samples (int, optional): How many samples should be used for the training. Defaults to 500.

    Returns:
        float: if epochs == 1 returns the loss on validation set.
        else returns configurations.LOSS_METRIC.
    """
    try:
        model.save_weights("test_weights.h5")
        #print("*******Model is already build*******")
    except ValueError:
        pass
    OptimizedModel.build_and_compile(model,np.shape(x_train))
    model.save_weights("test_weights.h5")
    samples = len(y_train)#TODO change to numpy function
    verbose = 0
    validation_data = None
    if("VALIDATION" in configurations.LEARNING_METRIC):
        x_train, x_test, y_train, y_test = train_test_split(x_train[0:samples], y_train[0:samples],test_size=0.2, random_state=42)
        validation_data = (x_test,y_test)
    cb_loss = lcb.LossCallback(samples=y_train.shape[0])
    start = time.time()
    hist = model.fit(
        x_train, y_train,
        epochs=configurations.TEST_EPOCHS,
        verbose=verbose,
        validation_data=validation_data,
        batch_size=configurations.BATCH_SIZE,
        callbacks=[cb_loss],
        shuffle=True,
        use_multiprocessing=True,
    )
    model.load_weights("test_weights.h5")
    elapsed_time = time.time() - start
    #print(f"Elapsed time: {elapsed_time}")
    #cb_loss.plot_loss()
    
    #if only one epoch is done, returns the last loss
    if(configurations.TEST_EPOCHS==1):
        return cb_loss.learning_metric["LAST_LOSS"]
    return cb_loss.learning_metric[configurations.LEARNING_METRIC]

