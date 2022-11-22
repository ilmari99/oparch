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
    weirds = pd.concat([weirds.copy() for _ in range(ntimes)])
    df = pd.concat([df,weirds])
    print(f"Added {len(weirds)} rows.")
    return df

def add_rows(x_train, y_train):
    # multiple values in train set
    train = pd.concat([x_train,y_train],axis=1)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([3/9]),ntimes=6) #13
    #train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([7/9]),ntimes=4)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([1]),ntimes=10)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([8/9]),ntimes=2)
    #train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([2/8]),ntimes=10)
    y_train = train.pop("quality")
    x_train = train
    return x_train, y_train

if __name__ == "__main__":
    # Read and handle data
    #df = pd.read_csv("/home/ilmari/python/Late-tyokurssi/viinidata/winequality-red.csv",sep=";")
    df = pd.read_csv("C:\\Users\\ivaht\\Desktop\\PYTHON\\Python_scripts\\Late-tyokurssi\\viinidata\\winequality-white.csv",sep=";")
    print(df.describe())
    #weirds = df[(np.abs(scipy.stats.zscore(df)) >= 3).any(axis=1)]
    #print(f"Found {len(weirds)} weird rows.")
    #weirds = pd.concat([weirds,weirds.copy(),weirds.copy()])
    df = max_norm(df)
    endog = df.pop("quality")
    exog = max_norm(df)
    # Create partition, here y_* is still in integers
    x_train,x_test, y_train, y_test = train_test_split(exog,endog,test_size=0.25,random_state=42)
    # multiple values in train set
    x_train, y_train = add_rows(x_train, y_train)
    
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    layers=[
        tf.keras.layers.Dense(49,activation="elu"),
        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dense(8,activation="softmax"),
        tf.keras.layers.Dense(256,activation="linear"),
        tf.keras.layers.Dense(1,"sigmoid"),
        ]
    """
    layers = [
        tf.keras.layers.Dense(32,activation="linear"),
        tf.keras.layers.Dense(512,activation="relu"),
        tf.keras.layers.Dense(85,activation="tanh"),
        tf.keras.layers.Dense(10,activation="linear"),
        tf.keras.layers.Dense(1,activation="sigmoid"),
    ]
    """
    layers = opt.utils.get_copy_of_layers(layers)
    model = tf.keras.models.Sequential(layers)
    model.build(np.shape(x_train))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.009, amsgrad=True),
        #loss=tf.keras.losses.Hinge()
        #loss=tf.keras.losses.LogCosh() #Gives perhaps a little better results on the rare values
        loss=tf.keras.losses.MeanAbsoluteError()
        )
    cb_loss = opt.LossCallback.LossCallback(early_stopping = False)
    print("Num of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    hist = model.fit(
        x_train, y_train,
        epochs=30,
        verbose=1,
        validation_data=(x_test,y_test),
        batch_size=256,
        callbacks=[cb_loss],
    )
    model.save("white-wine-model.h5",overwrite=False)
    """
    #model.save("red-wine-model.h5",overwrite=False)
    opt.utils.print_model(model,learning_metrics=cb_loss.learning_metric)
    cb_loss.plot_loss(new_figure=True, show=False)
    
    opt.set_default_misc(epochs=20,batch_size=256,learning_metric="VALIDATION_LOSS",verbose=0,validation_split=0.25)

    #model = opt.opt_loss_fun(model,x_train,y_train,categorical=False,return_metric=None)
    
    model = opt.opt_all_layer_params(model, x_train, y_train, "activation")
    model = opt.opt_all_layer_params(model, x_train, y_train, "units")
    model = opt.opt_optimizer_parameter(model, x_train, y_train, ["learning_rate","decay"],algo="Nelder-Mead")
    
    
    model.compile(optimizer=model.optimizer,loss=model.loss,metrics=["accuracy"])

    cb = opt.LossCallback.LossCallback()
    hist = model.fit(
        x_train,y_train,
        epochs=20,
        batch_size=256,
        callbacks=[cb],
        validation_data=(x_test,y_test)
    )
    cb.plot_loss(show=False,new_figure=True)
    #"""
    
    preds = pd.DataFrame(model.predict(x_test))
    preds = round(8*preds)
    y_test = pd.DataFrame(y_test)
    y_test = round(8*y_test)
    #print("Predictions:",preds)
    #print("Observations",y_test)
    pred_count = Counter(preds.to_numpy().flatten())
    obs_count = Counter(y_test.to_numpy().flatten())
    print("Neural network predictions",pred_count.items())
    print("Actual values",obs_count.items())
    count = 0
    for obs, pred in zip(y_test.values,preds.values):
        if int(obs) == int(pred):
            count += 1
    print("Accuracy of model: ", round(count/len(preds),3))
    errs = preds - y_test
    fig,ax = plt.subplots()
    #print("Errors:",errs)
    #ax.scatter(y_test,errs)
    bins = list(range(9))
    ax.bar(list(obs_count.keys()),list(obs_count.values()),label="Observations")
    ax.bar(pred_count.keys(),pred_count.values(),label="Predictions")
    #ax.hist(y_test,bins=bins,label="Observations")
    #ax.hist(preds,bins=bins,label="Predictions")
    ax.legend()
    fig, ax = plt.subplots()
    ax.hist(errs,label="Errors")
    ax.legend()
    opt.utils.print_model(model,learning_metrics=cb_loss.learning_metric)
    plt.show()
    
    
    
