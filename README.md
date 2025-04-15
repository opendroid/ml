# ml
My machine learning examples.


# Performance benckmarks


This table illustatesL:
1. Creation of two vectors each two billion random numbers.
2. Addition of two vectors

On Apple M3 MAX CPU and Apple Silicon Metal Performance Shaders (MPS) PyTorch API. The results are:


| Where | Description | Time (seconds) |
|-------|-------------|----------------|
| Python | Create Two 1 Billion elements lists | 184.156 |
| NumPy | np.array | 16.291 |
| Torch ("mps") | t1, torch.randn(vector_size, device="mps") | 0.024 |
| Python | Add two lists | 46.538 |
| NumPy | c = a + b | 4.677 |
| Torch ("mps") | t3 = t1 + t2 | 0.026 |


## Installed packages
After the virtual environment is created and activated, create the requirements.txt file.


1. Parameters are used in the model. And are learned from the data and used to make predictions.
2. Hyperparameters are used to train the model. Are set before the training and are not learned from the data. These are to be tuned.
    - Learning rate
    - Batch size
    - Epochs
    - Number of hidden layers
    - Number of neurons in each hidden layer
    - Activation function
    - Loss function
    - Regularization parameter: L1, L2

```shell
pip3 freeze > requirements.txt

```

References:
1. [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
2. [PyTorch Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
3. [Zillow dataset](https://www.zillow.com/research/data/)
