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


References:
1. [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
2. [PyTorch Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
3. [Zillow dataset](https://www.zillow.com/research/data/)
