"""Examples for the PyTorch Tensors
"""
import torch
from device import device

# By default, tensors are created on the CPU, create on device if available
shape = (3, 4,)
t1 = torch.randn(shape, device=device)
print(f"t1 = {t1.shape} of data type '{t1.dtype}' on '{t1.device}'")
t2 = torch.randn(shape, device=device)
t3 = torch.zeros_like(t1, device=device)

torch.add(t1, t2, alpha=1, out=t3)
print(f"Added values:\n {t1.T}\n + \n {t2.T} \n =\n {t3.T}")
# If needed clear torch.mps.empty_cache() and
# the empty cache by forcing garbage collection: gc.collect()
agg = t3.sum()
print(f"agg={agg}")
