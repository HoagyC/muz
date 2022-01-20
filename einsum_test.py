import torch

a = torch.arange(9).reshape(3, 3)
b = torch.arange(9, 0, -1).reshape(3, 3)

c = torch.arange(3)
d = torch.arange(3)
print(a, "\n", b)
e = torch.einsum("ij,ij->ij", [a, b])
f = torch.einsum("ij,ij -> ", [a, b])


print(e, f, sum(e))
