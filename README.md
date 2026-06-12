# ml

My machine learning examples.

## Performance benckmarks

This table illustatesL:

1. Creation of two vectors each two billion random numbers.
2. Addition of two vectors

On Apple M3 MAX CPU and Apple Silicon Metal Performance Shaders (MPS) PyTorch API. The results are:

| Where                     | Description                                | Time (seconds) |
|---------------------------|--------------------------------------------|----------------|
| Python                    | Create Two 1 Billion elements lists        | 184.156        |
| NumPy                     | np.array                                   | 16.291         |
| Torch ("mps")             | t1, torch.randn(vector_size, device="mps") | 0.024          |
| Python                    | Add two lists                              | 46.538         |
| NumPy                     | c = a + b                                  | 4.677          |
| Torch ("mps")             | t3 = t1 + t2                               | 0.026          |
| mlx                       | m.random.randn(vector_size, device="mlx")  |                |

## Conda Environment Setup

This project uses Conda to manage environments and dependencies, integrating both `conda` and `pip` package managers. The environment configuration is defined in [environment.yml](environment.yml).

### 1. Create or Update the Environment

To create or update the `mlx` environment using the configuration file:

```shell
conda env update -f environment.yml --prune
```

### 2. Activate the Environment

To activate the environment:

```shell
conda activate mlx
```

### 3. Managing Packages

- **Source of Truth:** [environment.yml](environment.yml) is the primary definition for all dependencies.
  - Conda packages (e.g., `numpy`, `scikit-learn`, ...)
  - Pip packages (e.g., `mlx`, `torch`, `transformers`, ...)

- **Exporting Dependencies:** If you need to export or check the current packages:

```shell
pip freeze > requirements.txt
# or export the full conda env:
conda env export --no-builds > environment_freeze.yml
```

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

References:

1. [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
2. [MLX](https://mlxpy.github.io/)
3. [MLX Examples](https://ml-explore.github.io/mlx/build/html/index.html)
4. [PyTorch Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
5. [Zillow dataset](https://www.zillow.com/research/data/)
