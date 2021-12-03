import custom_callback as ccb
import tensorflow as tf
import numpy as np


def test_learning_speed(model, x_train, y_train,samples=500):
    cb_loss = ccb.loss_callback()
    model.build(np.shape(x_train))
    model.compile(loss=tf.keras.losses.binary_crossentropy,
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
                  metrics=['accuracy'])
    hist = model.fit(
        x_train[:samples], y_train[:samples],
        epochs=1,
        verbose=2,
        validation_data=(x_train[samples:-1],y_train[samples:-1]),
        batch_size=16,
        callbacks=[cb_loss],
        shuffle=True
    )
    #cb_loss.plot_loss()
    return hist.history["val_loss"][0]

