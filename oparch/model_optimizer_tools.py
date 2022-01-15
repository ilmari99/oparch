from .OptimizedModel import OptimizedModel
from . import LossCallback as lcb
import tensorflow as tf
import numpy as np
from . import configurations
import time
from sklearn.model_selection import train_test_split

def test_learning_speed(model: tf.keras.Model, x_train: np.ndarray,
                        y_train: np.ndarray, samples=500, validation_split=0.2) -> float:
    """Tests the learning speed of the model without changing any parameters of the model.
       Returns configurations.LOSS_METRIC from LossCallback.

    Args:
        model (Sequential): Doesn't have to be built
        x_train (np.array): feature data
        y_train (np.array): variable to predict
        samples (int, optional): How many samples should be used for the training. Defaults to 500.

    Returns:
        float: if epochs == 1 returns the loss on validation set.
        else returns configurations.LOSS_METRIC.
    """
    try:
        model.optimizer.get_weights()
    except AttributeError:
        raise AttributeError("The model must be built before testing the learning speed")
    #rebuild and compile the model to get a clean optimizer
    if model.optimizer.get_weights(): #If list is not empty or model has not been
        model.build(np.shape(x_train))
        model.compile(optimizer=type(model.optimizer)(model.optimizer.get_config()["learning_rate"]),
                  loss=model.loss)
    #Save the models weights to return the model to its original state after testing the learning speed
    model.save_weights("test_weights.h5")
    samples = np.shape(y_train)[0] #TODO: this uses all available data instead of samples
    verbose = 0
    validation_data = None
    if("VALIDATION" in configurations.LEARNING_METRIC): #If the learning metric should be calculated from validation set
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
    elapsed_time = time.time() - start
    #Load the weights the model had when it came to testing, so testing doesn't affect the model itself
    #Rebuild and recompile to give the model a clean optimizer
    model.load_weights("test_weights.h5")
    model.build(np.shape(x_train))
    model.compile(optimizer=type(model.optimizer)(model.optimizer.get_config()["learning_rate"]),
                  loss=model.loss)
    #if only one epoch is done, returns the last loss
    if(configurations.TEST_EPOCHS==1):
        return cb_loss.learning_metric["LAST_LOSS"]
    return cb_loss.learning_metric[configurations.LEARNING_METRIC]

