py -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ oparch

I also got a good idea for the future of this project, and I wonder why I hadn't thought of it before. Make the functions less general, because too much generalization isn't good. The future: user has a model structure, and oparch package provides functions that only optimize a part of the structure separately, and the user should be the one to keep track of the optimized variables and their usage. The functions shouldn't return a model where the optimized thing is already implemented. For example 

opt_dense(model : Sequential, index : int, X : ndarray, y : ndarray, **kwargs) -> dict, Dense:

opt_learning_rate(model: Sequential, X : ndarray, y : ndarray) -> float:

opt_batch_size(model: Sequential, X : ndarray, y : ndarray) -> int:


The package could and should have higher levels of abstraction too, but only after these basics are well-implemented.

