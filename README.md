py -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ oparch

I also got a good idea for the future of this project, and I wonder why I hadn't thought of it before. Make the functions less general, because too much generalization isn't good. The future: user has a model structure, and oparch package provides functions that only optimize a part of the structure separately, and the user should be the one to keep track of the optimized variables and their usage. The functions shouldn't return a model where the optimized thing is already implemented.

