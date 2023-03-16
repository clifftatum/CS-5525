import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
from sklearn.preprocessing import StandardScaler
import numpy as np

x = np.array([13,16,19,22,23,38,47,56,58,63,65,70,71])

z = (x - np.mean(x))/np.std(x)
print(z)

zz = (x - np.min(x))/(np.max(x) - np.min(x))
dx = pd.DataFrame(x)
zzz = (x - dx.quantile(.5)[0])/(dx.quantile(.75)[0] - dx.quantile(.25)[0])

df1 = pd.DataFrame(z, columns=['standard'])
df2 = pd.DataFrame(zz, columns=['normalized'])
df3 = pd.DataFrame(zzz, columns=['IQ transformation'])
df = pd.concat([df1,df2,df3],axis=1)

df.plot()
plt.show()
# scalar = StandardScaler()
# scalar.fit(x.reshape(-1,1))
# scalar_transform = scalar.transform(x.reshape(-1,1))
# print(f'standardized data \n :{np.round(scalar_transform,2)}')
# mean = 194
# std = 11.2
# critical1 = 225
# critical2 = 175
# print(f'Sample mean is  {mean:.2f} ')
# print(f'Sample std is {std:.2f} ')
#
# print(f'The probability that '
#       f'the observation be between'
#       f'{critical2} and {critical1}'
#       f'is {(st.norm(mean , std).cdf(critical1)-st.norm(mean , std).cdf(critical2))*100:.2f}%')