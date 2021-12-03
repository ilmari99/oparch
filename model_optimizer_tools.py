import custom_callback as ccb
import tensorflow as tf
import numpy as np


def test_learning_speed(model, x_train, y_train,samples=500):
    cb_loss = ccb.loss_callback()
    model.build(np.shape(x_train))
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])
    model.fit(
        x_train[:samples], y_train[:samples],
        epochs=1,
        verbose=2,
        batch_size=10,
        callbacks=[cb_loss],
    )
    return cb_loss.loss_on_epoch_end

