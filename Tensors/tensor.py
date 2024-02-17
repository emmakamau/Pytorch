"""
Tensors are a specialized data structure that are very similar to arrays and matrices.
In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model’s parameters.
"""

import torch
import numpy as np


# Tensors can be created directly from data
def create_tensor():
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
    return data


# Can be created from Numpy Arrays
def create_tensor_from_numpy_array():
    data = create_tensor()
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
    return x_np


# shape is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.
def dimensionality_of_tensors():
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)

    print(f"Random Tensor: \n {rand_tensor} \n")
    print(f"Ones Tensor: \n {ones_tensor} \n")
    print(f"Zeros Tensor: \n {zeros_tensor}")


def tensor_attributes():
    tensor = torch.rand(3, 4)

    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")


def indexing_slicing():
    tensor = torch.ones(4, 4)
    print(f"First row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Last column: {tensor[..., -1]}")
    tensor[:, 1] = 0
    print(tensor)


def joining_tensors():
    x = torch.ones(4, 4)
    y = torch.ones(4, 3)
    z = torch.ones(4, 2)
    t1 = torch.cat([x, y, z], dim=1)
    print(t1)


def tensor_arithmetics():
    x = torch.ones(4, 4)
    y = torch.ones(4, 4)

    # This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
    # ``tensor.T`` returns the transpose of a tensor
    y1 = x @ y.T
    y2 = x.matmul(y.T)
    print("y1:", y1)
    print("y2:", y2)

    y3 = torch.rand_like(y1)
    torch.matmul(x, y.T, out=y3)
    print("y3:", y3)

    # This computes the element-wise product. z1, z2, z3 will have the same value
    z1 = x * y
    z2 = x.mul(y)
    print("z1:", z1)
    print("z2:", z2)

    z3 = torch.rand_like(x)
    torch.mul(x, y, out=z3)
    print("z3:", z3)


tensor_arithmetics()
