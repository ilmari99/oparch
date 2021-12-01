from PIL import Image
import numpy as np


def get_x_y_from_img_generator(generator):
    x = np.concatenate([generator.next()[0] for _ in range(generator.__len__())])
    y = np.concatenate([generator.next()[1] for _ in range(generator.__len__())])
    return x, y


def get_image_as_array(path, target_size=(300,300)):
    try:
        img = Image.open(path)
    except Exception:
        return -1, -1
    img = img.resize(target_size)
    return np.asarray(img), img

def predict_image(model,image_paths,target_size=(300,300)):
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