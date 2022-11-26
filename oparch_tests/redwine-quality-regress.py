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
import random
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import smogn

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
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([3/8]),ntimes=8)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([4/8]),ntimes=2)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([1]),ntimes=7)
    train = multip_rows(train,mask_cond=lambda df : df["quality"].isin([7/8]),ntimes=1)
    y_train = train.pop("quality")
    x_train = train
    return x_train, y_train

def add_noise(y_train,y_test):
    def func(x):
        pm = 1 if random.random() > 0.5 and x < 1 else -1
        noise = random.random() / 80
        return x + pm*noise
    random.seed(42)
    y_train = y_train.apply(func)
    y_test = y_test.apply(func)
    return y_train, y_test

def do_smogn(train : pd.DataFrame ,y_train : pd.Series = None,y_header = "",smoter_kwargs = {}):
    # If there is no y data specified
    print("Train: \n",train)
    if y_train is None:
        if not y_header:
            raise ValueError("Specify the y data by y_header or y_train data")
    # If there is y data specified
    else:
        y_train = y_train.reset_index(drop=True)
        print("y data: \n", y_train)
        # If the y data is not a series
        if not isinstance(y_train,pd.Series):
            # If the y data is not a series and no y_header given
            if not y_header:
                raise ValueError("y_header can only be inferred from the y_train data if y_train is a Series")
            y_train = pd.Series(y_train,name=y_header,index=0)
        
        y_header = y_train.name
        train = pd.concat([train,y_train],axis=1)
    #x_train = x_train.reindex()
    print("trainset:\n",train)
    print(train.columns)
    print("'quality' loc: ",train.columns.get_loc(y_header))
    
    if not smoter_kwargs:
        smoter_kwargs = {
            "k" : 256,
            "samp_method" : "balance",
            "rel_thres":0.7,
        }
    train = smogn.smoter(data=train, y=y_header,**smoter_kwargs,)
    return train, train.pop(y_header)
    
def accuracy_info(y_test, preds):
    obs_count = Counter(y_test.to_numpy().flatten())
    hits = {}#0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}
    for obs, pred in zip(y_test.values,preds.values):
        if obs == pred:
            obs = int(obs)
            if obs not in hits:
                hits[obs] = 0
            hits[obs] += 1
    corrects = 0
    for qual,count in hits.items():
        corrects += count
        print(f"Accuracy of model for quality {qual}: ", round(count/obs_count[qual],3))
    print(f"Total accuracy of model: {corrects/len(y_test)}")

if __name__ == "__main__":
    opt.__reset_random__()
    # Read and handle data
    #df = pd.read_csv("/home/ilmari/python/Late-tyokurssi/viinidata/winequality-red.csv",sep=";")
    try:
        df = pd.read_csv("C:\\Users\\ivaht\\Desktop\\PYTHON\\Python_scripts\\Late-tyokurssi\\viinidata\\winequality-red.csv",sep=";")
    except FileNotFoundError:
        df = pd.read_csv("/home/ilmari/python/Late-tyokurssi/viinidata/winequality-red.csv",sep=";")
    #weirds = df[(np.abs(scipy.stats.zscore(df)) >= 3).any(axis=1)]
    #print(f"Found {len(weirds)} weird rows.")
    #weirds = pd.concat([weirds,weirds.copy(),weirds.copy()])
    df = max_norm(df)
    endog = df.pop("quality")
    exog = df
    # Create partition
    x_train,x_test, y_train, y_test = train_test_split(exog,endog,test_size=0.25,random_state=42)
    
    scaler = StandardScaler()
    cols = list(x_train.columns)
    scaler.fit_transform(x_train)
    x_train = pd.DataFrame(scaler.transform(x_train),columns=cols)
    x_test = pd.DataFrame(scaler.transform(x_test),columns=cols)
    print("xtrain",x_train)
    # multiple values in train set
    #x_train, y_train = add_rows(x_train, y_train)
    print("X train size: ",np.shape(x_train))
    
    y_train, y_test = add_noise(y_train,y_test)
    print(y_train)
    fig,ax = plt.subplots()
    ax.hist(y_train,10)
    ax.set_title("Distribution after adding noise")
    print("y _train: \n", y_train)
    x_train, y_train = do_smogn(x_train,y_train,y_header="quality")
    print(x_train.describe())
    fig,ax = plt.subplots()
    ax.hist(y_train,10,label="Distribution after SMOGN")
    ax.set_title("Distribution after SMOGN")
    print("X train size after SMOGN: ",np.shape(x_train))
    #y_train, y_test = add_noise(y_train,y_test)
    plt.show()
    
    
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    
    layers=[
        tf.keras.layers.Dense(13,activation="tanh"),
        tf.keras.layers.Dense(168,activation="relu"),
        tf.keras.layers.Dense(40,activation="relu"),
        tf.keras.layers.Dense(128,activation="relu"),
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0087, decay=3*10**-5, amsgrad=False),
        ##loss=tf.keras.losses.Hinge()
        #loss=tf.keras.losses.LogCosh() #Gives perhaps a little better results on the rare values
        #loss = tf.keras.losses.MeanAbsoluteError(),
        loss = tf.keras.losses.MeanSquaredError(),
        )
    cb_loss = opt.LossCallback.LossCallback(early_stopping = False)
    print("Num of GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    hist = model.fit(
        x_train, y_train,
        epochs=100,
        verbose=1,
        validation_data=(x_test,y_test),
        batch_size=128,
        callbacks=[cb_loss],
    )
    cb_loss.plot_loss(new_figure=True, show=False)
    #model.save("red-wine-model.h5",overwrite=False)
    """
    #model.save("red-wine-model.h5",overwrite=False)
    opt.utils.print_model(model,learning_metrics=cb_loss.learning_metric)
    
    opt.set_default_misc(epochs=20,batch_size=128,learning_metric="VALIDATION_LOSS",verbose=0,validation_split=0.25)

    #model = opt.opt_optimizer_parameter(model, x_train, y_train, ["learning_rate","decay"],maxiters=30)#,"momentum","rho","epsilon"
    #model = opt.opt_loss_fun(model,x_train,y_train,categrical=False,return_metric=None)
    for i in range(1):
        model = opt.opt_optimizer_parameter(model, x_train, y_train, ["learning_rate","decay","amsgrad"],algo="Nelder-Mead")
        model = opt.opt_all_layer_params(model, x_train, y_train, "activation")
        model = opt.opt_all_layer_params(model, x_train, y_train, "units")
    #model = opt.opt_all_layer_params(model, x_train, y_train, "rate")
    
    model.compile(optimizer=model.optimizer,loss=model.loss,metrics=["accuracy"])

    cb = opt.LossCallback.LossCallback()
    hist = model.fit(
        x_train,y_train,
        epochs=100,
        batch_size=128,
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
    print("Neural network pedictions",pred_count.items())
    print("Actual values",obs_count.items())
    count = 0
    accuracy_info(y_test, preds)
    errs = preds - y_test
    fig,ax = plt.subplots()
    #print("Errors:",errs)
    #ax.scatter(y_test,errs)
    bins = list(range(9))
    ax.bar(list(obs_count.keys()),list(obs_count.values()),label="Observations",alpha=0.75)
    ax.bar(pred_count.keys(),pred_count.values(),label="Predictions",alpha=0.75)
    #ax.hist(y_test,bins=bins,label="Observations")
    #ax.hist(preds,bins=bins,label="Predictions")
    ax.legend()
    fig, ax = plt.subplots()
    ax.hist(errs,label="Errors")
    ax.legend()
    opt.utils.print_model(model,learning_metrics=cb_loss.learning_metric)
    plt.show()
    
    
    
