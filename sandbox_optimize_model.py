import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import numpy as np
import tensorflow as tf
import custom_callback as ccb
import process_data_tools as pdt
import sandbox_funs as own_funs
import model_optimizer as mod_op
import model_optimizer_tools as mot


if __name__ == '__main__':

    train_generator, validation_generator = own_funs.get_image_generators_from_path("C://Users//ivaht//Downloads//aalto_lut_train_validation//aalto_lut_training",
                                                           "C://Users//ivaht//Downloads//aalto_lut_train_validation//aalto_lut_validation")
    x_train, y_train = pdt.get_x_y_from_img_generator(train_generator)
    x_test, y_test = pdt.get_x_y_from_img_generator(validation_generator)

    cb1 = ccb.loss_callback()
    model = mod_op.get_model(x_train, y_train)

    cb1.plot_loss()