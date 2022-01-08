import oparch
print(help(oparch))
layer = oparch.model_optimizer.get_dense_layer([1,"relu"])
print(layer)