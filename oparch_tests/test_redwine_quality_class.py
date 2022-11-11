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

if __name__ == "__main__":
    # Read and handle data
    # Read and separate data
    df = pd.read_csv("/home/ilmari/python/Late-tyokurssi/viinidata/winequality-red.csv",sep=";")
    fig, ax = plt.subplots()
    ax.hist(df["quality"],bins=6,label="Full data hist")
    ax.legend()
    
    
    endog = df.pop("quality")
    exog = df
    # Separate to different sets
    x_train,x_test, y_train, y_test = train_test_split(exog,endog,test_size=0.25,random_state=42)
    
    # Multiply rare (picked manually) quality rows
    train = pd.concat([x_train,y_train],axis=1)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([3]),ntimes=6)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([4]),ntimes=2)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([8]),ntimes=4)
    #train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([7/8]),ntimes=1)
    y_train = train.pop("quality")
    
    # Convert the goal values to one-hot-encoded
    y_train = indices_to_one_hot(y_train,10)
    y_test = indices_to_one_hot(y_test,10)
    
    #Normalize the X values
    x_train = max_norm(train)
    x_test = max_norm(x_test)
    
    #Convert to numpy arrays
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    layers=[
        tf.keras.layers.Dense(512,activation="linear"),
        tf.keras.layers.Dense(206,activation="elu"),
        tf.keras.layers.Dropout(0.05),
        tf.keras.layers.Dense(10,activation="linear"),
        tf.keras.layers.Dropout(0.02),
        tf.keras.layers.Dense(12,activation="linear"),
        tf.keras.layers.Dense(10,"softmax"),
        ]
    layers = opt.utils.get_copy_of_layers(layers)
    model = tf.keras.models.Sequential(layers)
    model.build(np.shape(x_train))
    # NOTE: This is very sensitive to randomness :(
    model.compile(
        #optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.025375482208110867, decay=0.14532385839041545,momentum=0,rho=0,epsilon=10**-7),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.007763,decay=0.0003867, beta_1=0.77275,beta_2 = 0.997,epsilon = 10**-7),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
        )
    cb_loss = opt.LossCallback.LossCallback(early_stopping = False)
    print("Num of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    hist = model.fit(
        x_train, y_train,
        epochs=20,
        verbose=1,
        validation_data=(x_test,y_test),
        batch_size=256,
        callbacks=[cb_loss],
    )
    """
    #model.save("red-wine-model.h5",overwrite=False)
    opt.utils.print_model(model,learning_metrics=cb_loss.learning_metric)
    
    #cb_loss.plot_loss(new_figure=True, show=False)
    opt.set_default_misc(epochs=20,batch_size=256,learning_metric="NEG_VALIDATION_ACCURACY",verbose=0,validation_split=0.25)
    model = opt.opt_optimizer_parameter(model, x_train, y_train, ["learning_rate","decay","beta_1","beta_2","amsgrad"],algo="Nelder-Mead")
    model = opt.opt_all_layer_params(model, x_train, y_train, "units")
    model = opt.opt_all_layer_params(model, x_train, y_train, "rate")
    model = opt.opt_all_layer_params(model, x_train, y_train, "activation")
    #model = opt.opt_optimizer_parameter(model, x_train, y_train, ["learning_rate","decay","beta_1","beta_2","amsgrad"])
    
    model.compile(optimizer=model.optimizer,loss=model.loss,metrics=["accuracy"])

    cb = opt.LossCallback.LossCallback()
    hist = model.fit(
        x_train,y_train,
        epochs=20,
        batch_size=256,
        callbacks=[cb],
        validation_data=(x_test,y_test)
    )
    #cb.plot_loss(show=False,new_figure=True)
    #"""
    
    preds = pd.DataFrame(model.predict(x_test))
    preds = preds.idxmax(axis=1)
    y_test = pd.DataFrame(y_test)
    y_test = y_test.idxmax(axis=1)
    #print(preds)
    #print(y_test)
    pred_count = Counter(preds)
    obs_count = Counter(y_test)
    print("Neural network pedictions",pred_count.items())
    print("Actual values",obs_count.items())
    count = 0
    for obs, pred in zip(y_test,preds):
        if obs == pred:
            count += 1
    print("Accuracy of model: ", round(count/len(preds),3))
    errs = preds - y_test
    fig,ax = plt.subplots()
    ax.bar(list(obs_count.keys()),list(obs_count.values()),label="Observations",align="edge",width=0.5)
    ax.bar(pred_count.keys(),pred_count.values(),label="Predictions",width=0.5)
    ax.legend()
    #print(errs)
    #ax.scatter(y_test,errs)
    fig,ax = plt.subplots()
    ax.hist(errs)
    opt.utils.print_model(model,learning_metrics=cb_loss.learning_metric)
    plt.show()
    
    
    
