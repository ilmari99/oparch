from PIL import Image
from keras_preprocessing.image.image_data_generator import ImageDataGenerator
import numpy as np


def get_image_generators_list(paths, batch_size=8, class_mode="binary",target_size=(180,180)):
    generators = []
    for path in paths:
        datagen = ImageDataGenerator(rescale=1/255)
        generator = datagen.flow_from_directory(
            path,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode
            )
        generators.append(generator)
    return generators


def get_x_y_from_img_generator(generator):
    """Iterates over a DirectoryIterator. Returns the images and labels as np.arrays

    Args:
        generator (DirectoryIterator): generator.flow_from_directory returns this object

    Returns:
        x(array), y(array): x contains the images as numpy arrays, y contains the corresponding labels
    """
    x = np.concatenate([generator.next()[0] for _ in range(generator.__len__())])
    y = np.concatenate([generator.next()[1] for _ in range(generator.__len__())])
    return x, y


def get_image_as_array(path, target_size=(300,300)):
    """Returns an image as an np.array. Also returns the opened PIL image as second return value

    Args:
        path (String): Path to the image
        target_size (tuple, optional): Desired image size. Defaults to (300,300).

    Returns:
        img_arr(array), img(PIL.Image): Returns the pixel values as an np.array and the opened PIL.Image
    """    
    try:
        img = Image.open(path)
    except Exception:
        return -1, -1
    img = img.resize(target_size)
    return np.asarray(img), img

def predict_image(model,image_paths,target_size=(300,300)):
    """Uses a model to predict the labels propabilities.

    Args:
        model (Model): tensorflow model
        image_paths (list): list of path names even if just one
        target_size (tuple, optional): The image size the model is designed for. Defaults to (300,300).

    Raises:
        TypeError: if image_paths is not a list

    Returns:
        list: list of predicted propabilities
    """    
    if(not isinstance(image_paths,list)):
        raise TypeError("Argument image_paths excpects a list")
    for path in image_paths:
        x, img = get_image_as_array(path,target_size=target_size)
        if(x==-1 and img==-1):
            print(f"Error with path {path}. Skipping image.")
        x = np.expand_dims(x, axis=0)
        classes = model.predict(x)
        print(f"The probability distribution for picture with path {path} is {classes}")
        img.show()
    return classes