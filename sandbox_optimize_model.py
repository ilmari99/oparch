import os
import zipfile
import numpy as np
import tensorflow as tf
from OptimizedModel import OptimizedModel
import custom_callback as ccb
import process_data_tools as pdt
import model_optimizer as mod_op
import model_optimizer_tools as mot
import constants
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

def get_happiness_xy():
    csv_data = pd.read_csv("C:\\Users\\ivaht\\Downloads\\PersonalData.csv")
    happiness = csv_data.pop("How good was the day (1-10)")
    csv_data.__delitem__("Date")
    numeric_features = ['How busy (1-10)','Weight (kg)','How drunk','Nicotine (mg)','Studying (hours)','Sleep (hours)','Time spent with people (hours)']
    numeric_data = csv_data[numeric_features]
    #tf.convert_to_tensor(numeric_data)
    numeric_data = numeric_data.to_numpy()
    happiness = happiness.to_numpy()

    indices_to_drop = []
    for i,row in enumerate(numeric_data):
        if(np.isnan(np.sum(row))):
            indices_to_drop.append(i)

    numeric_data = np.delete(numeric_data, indices_to_drop,0)
    happiness = np.delete(happiness, indices_to_drop,0)
    
    #normalize the values by column
    happiness_norm = happiness / happiness.max(axis=0)
    numeric_norm = numeric_data / numeric_data.max(axis=0)
    return numeric_norm, happiness_norm

def get_horse_human_xy():
    local_zip = 'C:\\Users\\ivaht\\Downloads\\horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/tmp/horse-or-human')
    zip_ref.close()

    # Directory with our training horse pictures
    #train_horse_dir = os.path.join('/tmp/horse-or-human/horses')
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human',
        target_size=constants.IMAGE_SIZE,
        batch_size=constants.BATCH_SIZE,
        class_mode="binary",
    )
    X, y = pdt.get_x_y_from_img_generator(train_generator)
    return X,y

if __name__ == '__main__':

    #train_generator, validation_generator = own_funs.get_image_generators_from_path("C://Users//ivaht//Downloads//aalto_lut_train_validation//aalto_lut_training",
    #                                                       "C://Users//ivaht//Downloads//aalto_lut_train_validation//aalto_lut_validation")
    
    x,y = get_happiness_xy()
    x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2)
    cb1 = ccb.loss_callback()
    model = mod_op.get_model(x_train, y_train)
    model = OptimizedModel.build_and_compile(model,np.shape(x_train))
    model.fit(
        x_train, y_train,
        epochs=15,
        verbose=2,
        batch_size=constants.BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[cb1],
        shuffle=True,
    )
    model.summary()
    #pdt.predict_image(model,["C:\\Users\\ivaht\\Downloads\\horse-or-human\\horses\\horse01-0.png","C:\\Users\\ivaht\\Downloads\\horse-or-human\\horses\\horse01-3.png",
    #                         "C:\\Users\\ivaht\\Downloads\\horse-or-human\\horses\\horse01-6.png","C:\\Users\\ivaht\\Downloads\\horse-or-human\\horses\\horse01-13.png"],
    #                  target_size=constants.IMAGE_SIZE)
    
    
    cb1.plot_loss()
    
    