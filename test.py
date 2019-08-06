import numpy as np
import mcvisPy

n = 20
p = 5

x = np.random.normal(0, 1, n * p).reshape(n, p)
x[:, 0] = x[:, 1] + np.random.normal(0, 0.5, n)
res = mcvisPy.mcvisPy(x)
print(res)