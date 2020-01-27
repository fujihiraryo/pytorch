from __future__ import print_function
import numpy as np
import torch
x = torch.empty(5, 3)
print(x)

x = torch.rand(5, 3)
print(x)
print(x[2, 1].item())

# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)

# x = torch.zeros(5, 3)
# print(x)

# x = torch.tensor([5.5, 3.0000000])
# print(x)

# x = x.new_ones(5, 3, dtype=torch.double)
# print(x)

# x = torch.ones(5, 3, dtype=torch.int)
# print(x)

# x = torch.rand_like(x, dtype=torch.float)
# print(x)

# print(x.size())

# x = torch.rand(5, 3)
# y = torch.rand(5, 3)
# z = torch.empty(5, 3)
# torch.add(x, y, out=z)
# print(x, y, z)

# print(x[:, 1])
# print(x[1, :])

# x = torch.randn(4, 4)
# print(x)
# x = torch.rand(4, 4)
# print(x)

# y = x.view(16, -1)
# z = x.view(2, 2, 2, -1)
# print(y)
# print(z)

# x = torch.rand(1)
# print(x)
# print(x.item())

# a = torch.ones(5)
# b = a.numpy()
# c = a
# a.add_(1)
# print(a)
# print(b)
# print(c)

# a = np.ones(5)
# b = torch.from_numpy(a)
# np.add(a, 1, out=a)
# print(a)
# print(b)

# print(torch.cuda.is_available())
