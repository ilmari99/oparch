from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_image_generators_from_path(training_path, validation_path):
    target_size = (180,180)
    batch_size = 16
    class_mode = "binary"
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    validation_datagen = ImageDataGenerator(rescale=1 / 255)
    train_generator = train_datagen.flow_from_directory(
        training_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
    )
    return train_generator, validation_generator