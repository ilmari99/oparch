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
        model.save_weights("test_weights.h5")
        #print("*******Model is already build*******")
    except ValueError:
        pass
    OptimizedModel.build_and_compile(model,np.shape(x_train))
    model.save_weights("test_weights.h5")
    optimizer_state = model.optimizer.get_weights()
    samples = np.shape(y_train)[0] #TODO change to numpy function
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
    after_training_optimizer_state = model.optimizer.get_weights()
    model.load_weights("test_weights.h5")
    elapsed_time = time.time() - start
    
    #The model has random weigths here, but its optimizer still has a state from the training
    OptimizedModel.build_and_compile(model, np.shape(x_train))
    #if only one epoch is done, returns the last loss
    if(configurations.TEST_EPOCHS==1):
        return cb_loss.learning_metric["LAST_LOSS"]
    return cb_loss.learning_metric[configurations.LEARNING_METRIC]

