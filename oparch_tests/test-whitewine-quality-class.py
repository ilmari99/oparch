from collections import Counter
import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.insert(0, str(path_root))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import scipy
import tensorflow as tf
from sklearn.model_selection import train_test_split
import oparch as opt

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data,dtype=int)
    arr = np.zeros((len(targets),nb_classes),dtype=int)
    print(targets)
    for i,_ in enumerate(arr):
        arr[i,int(targets[i])] = 1
    return arr
def max_norm(df, inplace = True):
    # Normalize data
    if not inplace:
        df = df.copy()
    df = df.apply(lambda x : x / max(x))
    return df

def multip_rows(df,ntimes=3,mask_cond=None):
    if mask_cond is None:
        mask_cond = lambda df : (np.abs(scipy.stats.zscore(df)) >= 3).any(axis=1)
    weirds = df[mask_cond(df)]
    print(f"Found {len(weirds)} weird rows.")
    weirds = pd.concat([weirds,weirds.copy()])
    df = pd.concat([df,weirds])
    return df

if __name__ == "__main__":
    # Read and handle data
    df = pd.read_csv("/home/ilmari/python/Late-tyokurssi/viinidata/winequality-white.csv",sep=";")
    #df = df[(np.abs(scipy.stats.zscore(df)) < 3).all(axis=1)]
    #df = multip_rows(df)
    df = multip_rows(df,mask_cond=lambda df : df["quality"].isin([1,2,3,4,8,9]))
    endog = df.pop("quality")
    endog = pd.DataFrame(indices_to_one_hot(endog,10))
    exog = max_norm(df)
    x_train,x_test, y_train, y_test = train_test_split(exog,endog,test_size=0.25,random_state=42)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    layers=[
        tf.keras.layers.Dense(35,activation="relu"),
        tf.keras.layers.Dense(128,activation="linear"),
        tf.keras.layers.Dropout(0.00166),
        tf.keras.layers.Dense(128,activation="linear"),
        tf.keras.layers.Dropout(0.019),
        #tf.keras.layers.Dense(12,activation="linear"),
        tf.keras.layers.Dense(10,"softmax"),
        ]
    layers = opt.utils.get_copy_of_layers(layers)
    model = tf.keras.models.Sequential(layers)
    model.build(np.shape(x_train))
    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.03, decay=0.16,momentum=0,rho=0,epsilon=10**-7),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
        )
    cb_loss = opt.LossCallback.LossCallback(early_stopping = False)
    print("Num of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    hist = model.fit(
        x_train, y_train,
        epochs=100,
        verbose=1,
        validation_data=(x_test,y_test),
        batch_size=256,
        callbacks=[cb_loss],
    )
    
    #model.save("red-wine-model.h5",overwrite=False)
    opt.utils.print_model(model,learning_metrics=cb_loss.learning_metric)
    
    cb_loss.plot_loss(new_figure=True, show=False)
    opt.set_default_misc(epochs=10,batch_size=256,learning_metric="NEG_VALIDATION_ACCURACY",verbose=0,validation_split=0.25)

    model = opt.opt_all_layer_params(model, x_train, y_train, "units")
    model = opt.opt_all_layer_params(model, x_train, y_train, "rate")
    model = opt.opt_all_layer_params(model, x_train, y_train, "activation")
    model = opt.opt_optimizer_parameter(model, x_train, y_train, ["learning_rate","decay","momentum","rho","epsilon"])
    
    model.compile(optimizer=model.optimizer,loss=model.loss,metrics=["accuracy"])

    cb = opt.LossCallback.LossCallback()
    hist = model.fit(
        x_train,y_train,
        epochs=50,
        batch_size=256,
        callbacks=[cb],
        validation_data=(x_test,y_test)
    )
    cb.plot_loss(show=False,new_figure=True)
    preds = pd.DataFrame(model.predict(x_test))
    preds = preds.idxmax(axis=1)
    y_test = pd.DataFrame(y_test)
    y_test = y_test.idxmax(axis=1)
    print(preds)
    print(y_test)
    print("Neural network pedictions",Counter(preds).items())
    print("Actual values",Counter(y_test).items())
    count = 0
    for obs, pred in zip(y_test,preds):
        if obs == pred:
            count += 1
    print("Accuracy of model: ", round(count/len(preds),3))
    errs = preds - y_test
    fig,ax = plt.subplots()
    #print(errs)
    #ax.scatter(y_test,errs)
    ax.hist(errs)
    opt.utils.print_model(model,learning_metrics=cb_loss.learning_metric)
    plt.show()
    
    
    
