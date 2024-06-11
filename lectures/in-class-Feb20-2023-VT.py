from skewness_object02 import skeness
import numpy as np

N = 1000
a = np.random.normal(0,1,N)
test = skeness(a,2)
m2 = test.sk_ew()

test = skeness(a,3)
m3 = test.sk_ew()

g1 = m3/m2**(3/2)
G1 = np.sqrt(N*(N-1)/(N-2))
print(f'the skewness of the data is {g1:.2f}')
print(f'the adjusted skewness of the data is {G1:.2f}')