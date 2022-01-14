import oparch
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

def y_function(x) -> np.ndarray:
    y = np.zeros(max(np.shape(x))) #Assumes more training samples than features which should be true
    print(f"{np.shape(y)}")
    features = len(x[0])
    coeffs = np.array([(c+1)*0.5 for c in range(features)])
    for i,row in enumerate(x):
        coeffs_randomness = np.random.uniform(low=-0.5, high=0.5, size=(1,features))
        y[i] = np.sum(coeffs*row*coeffs_randomness[0], axis=0) #Add some randomness
        y = 1/(1+np.exp(-y))
    return y

np.random.seed(seed=42) #for reproducibility
x = np.random.rand(300,4)
y = y_function(x)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
layers = [tf.keras.layers.Dense(1),tf.keras.layers.Dense(1)]
layers_configs = [layer.get_config() for layer in layers]
model = tf.keras.models.Sequential(layers)
model.build(np.shape(x_train))
model.compile(loss="mse",optimizer="sgd",)
hist = model.fit(
        x_train, y_train,
        epochs=oparch.configurations.TEST_EPOCHS,
        verbose=1,
        validation_data=(x_test,y_test),
        batch_size=oparch.configurations.BATCH_SIZE,
        shuffle=True,
        use_multiprocessing=True,
    )

layers = [tf.keras.layers.Dense.from_config(config) for config in layers_configs]
optimized_model = oparch.model_optimizer.get_optimized_model(x_train, y_train, layers)

#optimized_model.build_and_compile(optimized_model.model, np.shape(x_train))
optimized_model.print_optimized_model()
optimized_model.model.fit(
        x_train, y_train,
        epochs=oparch.configurations.TEST_EPOCHS,
        verbose=1,
        validation_data=(x_test,y_test),
        batch_size=oparch.configurations.BATCH_SIZE,
        shuffle=True,
        use_multiprocessing=True,
)


