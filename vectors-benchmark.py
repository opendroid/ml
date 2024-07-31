import torch
import numpy as np
import time
import random
from device import device


vector_size = 1000_000_000
unit = "billion"


def setup_two_lists(size):
    """Create two lists with random numbers
    """
    a = [random.uniform(0, 1000) for _ in range(size)]
    b = [random.uniform(0, 1000) for _ in range(size)]
    return a, b


def set_two_np_lists(size):
    """Create two NumPy lists with random numbers
    """
    np1 = np.array([np.random.uniform(low=0, high=1000, size=vector_size)])
    np2 = np.array([np.random.uniform(low=0, high=1000, size=vector_size)])
    return np1, np2


# Create vector_size array in python
start = time.time()
a, b = setup_two_lists(vector_size)
end = time.time()
print(f"Python {unit} rows creation: {end-start} seconds")

# Benchmark NP
start = time.time()
np1, np2 = set_two_np_lists(vector_size)
end = time.time()
print(f"NP {unit} rows creation: {end-start} seconds")

# Benchmark torch
start = time.time()
t1 = torch.randn(vector_size, device=device)
t2 = torch.randn(vector_size, device=device)
end = time.time()
print(f"torch {unit} rows creation: {end-start} seconds")

# Add two vectors
start = time.time()
c = [a[i] + b[i] for i in range(vector_size)]
end = time.time()
print(f"Python {unit} rows addition: {end-start} seconds")

# Add using NumPy
start = time.time()
cnp1 = np1 + np2
end = time.time()
print(f"Numpy {unit} rows addition: {end-start} seconds")

# Add using PyTorch
start = time.time()
t3 = t1 + t2
end = time.time()
print(f"torch MPS {unit} rows addition: {end-start} seconds")
