import os
import zipfile

from keras_preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import numpy as np
import tensorflow as tf
import custom_callback as ccb
import process_data_tools as pdt
import sandbox_funs as own_funs
import model_optimizer as mod_op
import model_optimizer_tools as mot
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    #train_generator, validation_generator = own_funs.get_image_generators_from_path("C://Users//ivaht//Downloads//aalto_lut_train_validation//aalto_lut_training",
    #                                                       "C://Users//ivaht//Downloads//aalto_lut_train_validation//aalto_lut_validation")
    local_zip = 'C:\\Users\\ivaht\\Downloads\\horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp/horse-or-human')
    zip_ref.close()

    # Directory with our training horse pictures
    #train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human',
        target_size=(50,50),
        batch_size=8,
        class_mode="binary",
    )
    X, y = pdt.get_x_y_from_img_generator(train_generator)
    x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
    #x_test, y_test = pdt.get_x_y_from_img_generator(validation_generator)

    cb1 = ccb.loss_callback()
    model = mod_op.get_model(x_train, y_train)
    print("The returned model:")
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["accuracy"])
    model.fit(
        x_train, y_train,
        epochs=10,
        verbose=2,
        batch_size=16,
        validation_data=(x_test, y_test),
        callbacks=[cb1],
        shuffle=True
    )
    model.summary()
    cb1.plot_loss()