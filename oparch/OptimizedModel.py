import tensorflow as tf
import numpy as np
import configurations
from model_optimizer_tools import test_learning_speed


class OptimizedModel:
    layers = []
    loss_fun = tf.keras.losses.MeanSquaredError()
    learning_rate = 0.01
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    loss = 10000
    model = tf.keras.models.Sequential()
    layer_configs = []
    def __init__(self,layers,x_data):
        tf.random.set_seed(42)
        self.layers = layers
        self.model = tf.keras.models.Sequential(self.layers)
        self.input_shape = np.shape(x_data)
        self.layer_configs = [layer.get_config() for layer in layers]
        
    def get_model(self):
        self.model = self.build_and_compile(self.model, self.input_shape)
        return self.model
    
    def set_layers_from_config(self, layer_configs, index=None):
        if(index == None):
            self.layer_configs = layer_configs
        else:
            self.layer_configs[index] = layer_configs
        self.layers = [tf.keras.layers.Dense.from_config(config) for config in layer_configs]
        self.model = tf.keras.models.Sequential(self.layers)
        self.build_and_compile(self.model,self.input_shape)
        
        
    def get_layers(self):
        """returns a shallow copy of the models layers
        """     
        return self.layers.copy()
    
    def get_model_clone(self):
        model_clone = tf.keras.models.clone_model(self.model)
        return model_clone
    
    @classmethod
    def optimize_learning_rate(cls, model, x, y):
        #print(f"Optimizing learning rate...")
        def_lr = cls.learning_rate
        base_metric = test_learning_speed(model,x,y)
        model.reset_states()
        lrs = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        for lr in lrs:
            cls.learning_rate = lr
            metric = test_learning_speed(model,x,y)
            model.reset_states()
            #print(f"Learning rate: {lr}, {configurations.LEARNING_METRIC}:{metric}")
            if(metric<base_metric):
                def_lr = lr
                base_metric = metric
        cls.learning_rate = def_lr
        #print(f"Optimized learning_rate: {cls.learning_rate} with {configurations.LEARNING_METRIC}:{base_metric}")
        cls.set_learning_rate(def_lr)
        return def_lr
    
    @classmethod
    def optimize_loss_fun(cls,model,x,y):
        #print(f"Optimizing loss function...")
        def_loss = cls.loss_fun
        base_metric = test_learning_speed(model,x,y)
        model.reset_states()
        if(not all(isinstance(yi,int) for yi in y)): #TODO Tämän ehdon pitäisi tarkistaa, onko y categorinen vai ei
            loss_function_dict = configurations.REGRESSION_LOSS_FUNCTIONS
        for loss_fun in loss_function_dict.values():
            cls.loss_fun = loss_fun
            metric = test_learning_speed(model,x,y)
            model.reset_states()
            #print(f"Loss function: {type(cls.loss_fun).__name__}, {configurations.LEARNING_METRIC}:{metric}")
            if(metric<base_metric):
                def_loss = loss_fun
                base_metric = metric
        cls.loss_fun = def_loss
        #print(f"Optimized loss function: {type(cls.loss_fun).__name__}, {configurations.LEARNING_METRIC}:{base_metric}")
        return def_loss
    
    @classmethod
    def optimize_optimizer(cls,model,x,y):
        #print(f"Optimizing optimizer...")
        def_optimizer = cls.optimizer
        base_metric = test_learning_speed(model,x,y)
        model.reset_states()
        for opt in configurations.OPTIMIZERS.values():
            cls.optimizer = opt
            cls.set_learning_rate(cls.learning_rate)
            metric = test_learning_speed(model,x,y)
            model.reset_states()
            #print(f"Optimizer: {cls.optimizer.get_config()}, {configurations.LEARNING_METRIC}:{metric}")
            if(metric<base_metric):
                def_optimizer = opt
                base_metric = metric
        cls.optimizer = def_optimizer
        #print(f"Optimized optimizer: {cls.optimizer.get_config()}, {configurations.LEARNING_METRIC}:{base_metric}")
        return def_optimizer
    
    @classmethod
    def set_learning_rate(cls,learning_rate):
        cls.learning_rate = learning_rate
        optimizer_config = cls.optimizer.get_config()
        optimizer_config["learning_rate"] = learning_rate
        cls.optimizer = cls.optimizer.__class__.from_config(optimizer_config)
    
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
    
    
    