import torch

x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = 3 * y**2
out = z.mean()
print(z, out)

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a**2).sum()
print(b.grad_fn)
b.backward()
print(a.grad)

x = torch.ones(1, 2, requires_grad=True)
y = x[0, 0] + x[0, 1]
y.backward()
print(x.grad)

x = torch.ones(3, requires_grad=True)
y = x * x
v = torch.tensor([1, 2, 3], dtype=torch.float)
y.backward(v)
print(x.grad)
