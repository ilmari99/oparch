import custom_callback as ccb
import tensorflow as tf
import numpy as np
import constants

def build_and_compile(model, input_shape):
    """Builds and compiles a Sequential model

    Args:
        model (tf.keras.Model.Sequential): Tensorflow model
        input_shape (tuple): A tuple that is used as the input_shape of the model
                             Use for example np.shape(input)
                             
    doesnt return anything, but builds and compiles the input model
    """
    model.build(input_shape)
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
                  metrics=['accuracy'])


def test_learning_speed(model, x_train, y_train,samples=500):
    """Tests the learning speed of the model.
       The learning speed is measured by the loss on the validation set

    Args:
        model (Sequential): Doesn't have to be built
        x_train (np.array): feature data
        y_train (np.array): feature labels
        samples (int, optional): How many samples should be used for the training. Defaults to 500.

    Returns:
        float: loss on validation set
    """    
    cb_loss = ccb.loss_callback()
    build_and_compile(model, np.shape(x_train))
    hist = model.fit(
        x_train[:samples], y_train[:samples],
        epochs=1,
        verbose=2,
        validation_data=(x_train[samples:-1],y_train[samples:-1]),
        batch_size=constants.BATCH_SIZE,
        callbacks=[cb_loss],
        shuffle=True
    )
    #cb_loss.plot_loss()
    return hist.history["val_loss"][0]

