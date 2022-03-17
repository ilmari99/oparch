Oparch allows you to do some rudimentary optimization of a Sequential tensorflow model.
All you need is a rough idea of a sequential model, data, and some know-how.
Due to the lack of documentation, refer to oparch_tests/test_abalone_script or the functions themselves

Note: This is a personal project, and thats what it looks like.

DOCS:
oparch.optimize:
Commonly supported kwargs in functions in oparch.optimize
--------------
    return_metric(str) : Which metric to use to evaluate performance, default is value in oparch.configurations. LAST_LOSS usually outperforms others.
    "return_model" : true or false, whether to return the model (true), or to return the tried values and results in a list of tuples (false): [(value,result),     (value1,result1)...]
    samples : How many samples to use for the evaluation. Defaults to all.
    validation_split : how many percent of the data to use for validation (only applicable if using an evaluation based on a metric calculated from validation set). Default 0.2.
    epochs : How many times to go through the training data. Defaults to 5, however 1 should be used with slow computers or large datasets or models
    batch_size : How many data pieces should be considered before calculating gradient. Default 32. If running out of memory, consider decreasing.
    verbose : What should be reported to stdout. (Not implemented)
    decimals : Up to how many decimal points should the evaluation metric be optimized. (Should perhaps be changed to apply to the metric, not to the result). Defaults to 5.
    metrics(list) : which metrics should be used to compile the model. Defaults to ["accuracy"]


    opt_all_layer_params(model,X,y,param,**kwargs)
    Loops over all layers and optimizes the given parameter with a global grid search (because often has multiple minimas)
    if the layer has that parameter attribute.
    Doesn't optimize the last layer.

    Args:
        model (Sequential): The model which layers are looped through
        X (feature data): a numpy array
        y (observed output): a numpy arra
        param (string): a string identifier
        **kwargs : 
                    param : list with values to be tested, if not specified, uses defaults from configurations #ex. [2,4,8,16...]

    Returns:
        model: model with the optimal layers
    example usage:

    x = np.array([[0.45, 0.66, 0.77, 0.69], [0.2, 0.44, 0.55, 0.22] ....])
    y = [0.5, 0.66 ....] #Could also be categorical
    model = [Flatten(),Dense(1),Dense(1)]
    #...build and compile model
    opt_all_layer_params(model,x,y,"units")
    # returns model, where the first Dense (index=1) units have been optimized with grid search or removed if better performance



    opt_optimizer_parameter(model: tf.keras.models.Sequential,
                        X: np.ndarray,
                        y: np.ndarray,
                        param: list,
                        **kwargs) -> tf.keras.models.Sequential or list:
    Uses a specified (default "TNC") algorithm from the scipy library, to optimize the list of parameters.
    If an optimizer doesn't have a specified parameter, it is reported, removed, and then resumed with out the incorrect parameter.
    All optimizer parameters are between (0,1), so uses these bounds for the optimization


    Args:
        model (Sequential) : model, which optimizer is optimized
        X (array) :
        y (array) : 
        param (list) : list of the optimizer hyperparameters to tune
        **kwargs : 
            maxiter : maximum number of iterations to do to find the optimal combination. Defaults to 50.
            algo : must be one of: "TNC","L-BFGS-B" or "SLSQP"
        Returns:
            model: model with the optimal layers

