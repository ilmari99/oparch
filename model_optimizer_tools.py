import OptimizedModel
import custom_callback as ccb
import tensorflow as tf
import numpy as np
import constants
import time
from sklearn.model_selection import train_test_split


def test_learning_speed(model, x_train, y_train,samples=500, validation_split=0.2):
    """Tests the learning speed of the model.
       The learning speed is measured by the loss on the validation set

    Args:
        model (Sequential): Doesn't have to be built
        x_train (np.array): feature data
        y_train (np.array): feature labels
        samples (int, optional): How many samples should be used for the training. Defaults to 500.

    Returns:
        float: if epochs == 1 returns the loss on validation set.
        else returns constants.LOSS_METRIC.
    """
    if not model.built:
        OptimizedModel.OptimizedModel.build_and_compile(model,np.shape(x_train))
    samples = len(y_train)#TODO change to numpy function
    verbose = 0
    x_train, x_test, y_train, y_test = train_test_split(x_train[0:samples], y_train[0:samples],test_size=0.1, random_state=42)
    cb_loss = ccb.loss_callback(samples=y_train.shape[0])
    start = time.time()
    hist = model.fit(
        x_train, y_train,
        epochs=constants.TEST_EPOCHS,
        verbose=verbose,
        validation_data=(x_test,y_test),
        batch_size=constants.BATCH_SIZE,
        callbacks=[cb_loss],
        shuffle=True,
        use_multiprocessing=True,
    )
    elapsed_time = time.time() - start
    #print(f"Elapsed time: {elapsed_time}")
    #cb_loss.plot_loss()
    
    #if only one epoch is done, returns the last loss
    if(constants.TEST_EPOCHS==1):
        return cb_loss.learning_metric["LAST_LOSS"]
    return cb_loss.learning_metric[constants.LEARNING_METRIC]

