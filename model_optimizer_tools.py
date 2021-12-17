from OptimizedModel import OptimizedModel
import custom_callback as ccb
import tensorflow as tf
import numpy as np
import constants
from sklearn.model_selection import train_test_split

def build_and_compile(model, input_shape):
    """Builds and compiles a Sequential model

    Args:
        model (tf.keras.Model.Sequential): Tensorflow model
        input_shape (tuple): A tuple that is used as the input_shape of the model
                             Use for example np.shape(input)
                             
    Returns: model (tf.keras.Model.Sequential): returns the built model
    """
    model.build(input_shape)
    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=["accuracy"])
    return model


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
        else returns the average decrease of loss per validation set.
    """
    OptimizedModel.build_and_compile(model,np.shape(x_train))
    samples=len(y_train)#TODO change to numpy function
    epochs = 5
    verbose = 2
    x_train, x_test, y_train, y_test = train_test_split(x_train[0:samples], y_train[0:samples],test_size=0.2)
    cb_loss = ccb.loss_callback()
    hist = model.fit(
        x_train, y_train,
        epochs=epochs,
        verbose=verbose,
        validation_data=(x_test,y_test),
        batch_size=constants.BATCH_SIZE,
        callbacks=[cb_loss],
        shuffle=True
    )
    #cb_loss.plot_loss()
    
    #if only one epoch is done, returns the
    if(epochs==1):
        return cb_loss.learning_metric["LAST_LOSS"]
    return cb_loss.learning_metric[constants.LEARNING_METRIC]

