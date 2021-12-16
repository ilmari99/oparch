import tensorflow as tf
import numpy as np

class OptimizedModel:
    layers = []
    loss_fun = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.RMSprop()
    loss = 10000
    model = tf.keras.models.Sequential()
    def __init__(self,last_layers,x_data):
        self.layers = last_layers
        self.model = tf.keras.models.Sequential(last_layers)
        self.input_shape = np.shape(x_data)
        
    def get_model(self):
        self.build_and_compile(self.model,self.input_shape)
        return tf.keras.models.Sequential(self.layers)
    
    def set_layers(self, layers):
        self.layers = layers
        
    def get_layers(self):
        """returns a copy of the models layers
        """        
        return self.layers.copy()
    
    @classmethod
    def build_and_compile(cls, model, input_shape):
        """Builds and compiles a Sequential model

        Args:
            model (tf.keras.Model.Sequential): Tensorflow model
            input_shape (tuple): A tuple that is used as the input_shape of the model
                                 Use for example np.shape(input)
                             
        Returns: model (tf.keras.Model.Sequential): returns the built model
        """
        model.build(input_shape)
        model.compile(loss=cls.loss_fun,
                    optimizer=cls.optimizer,
                    metrics=["accuracy"])
        return model
    