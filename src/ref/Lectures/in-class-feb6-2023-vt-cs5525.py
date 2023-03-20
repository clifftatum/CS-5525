import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.preprocessing import normalize
# url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/Train_UWu5bXk.csv'
# df = pd.read_csv(url)
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


df = pd.read_csv('stock prices.csv')

# n_features = 20
# X, y = make_classification(n_samples = 1000, # Default 100
#                         n_features = n_features, # Default 100
#                         n_informative = 15, # default=2 The number of informative features, i.e., the number of
#                         # features used to build the linear model used to generate the output.
#                         n_redundant = 5,
#                         n_repeated = 0,
#                         random_state = 5525,
#                         n_classes=4,
#                         )
#
# X_2 = pd.DataFrame(X, columns=[f'feature{x}' for x in range(1,X.shape[1]+1)])
# y_2 = pd.DataFrame(y, columns=['target'])
#
# df = pd.concat([X_2,y_2], axis=1)
# # Normalization
# scaler = MinMaxScaler()
# X_std = scaler.fit_transform(df.select_dtypes(include = np.number))
# X = X_std[:,:-1]
#
#
# # Perform PCA analysis
# pca = PCA(n_components = n_features ,svd_solver='full')
# pca.fit(X)
# X_PCA = pca.transform(X)
# print("Original Dim", X.shape)
# print("Transformed Dim", X_PCA.shape)
# print(f'explained variance ratio {pca.explained_variance_ratio_}')
#
# plt.plot(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1),
#          np.cumsum(pca.explained_variance_ratio_))
# plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1))
# plt.grid()
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.show()


# Random forest
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=1,
                              max_depth=100,
                              n_estimators=100)

model.fit(X,y)
features = df.columns
importance = model.feature_importances_
indices = np.argsort(importance)[-20:]
plt.barh(range(len(indices)), importance[indices], color='b',align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()
# age = [10,11,13,32,34,40,72,73,75]
# df = pd.DataFrame(np.array(age), columns=['age'])
# print(df)
#
# def change(x):
#     if x<20:
#         return 'young'
#     elif (x<50) and (x>=20):
#         return 'mature'
#     else:
#         return 'old'
# df['description'] = df['age'].apply(change)
# print(df)
# N = 50000
# x = np.random.normal(0,np.sqrt(2),N)
# y = np.random.normal(0,np.sqrt(3),N)
#
# X = np.vstack((x,y)).T
# df = pd.DataFrame(X)
# df = df - df.mean()
#
# cov = (df.values.T @ df.values)/(len(df)-1)
# print(cov)



# print(f'The total number of missing entries are {df.isna().sum().sum()}')
# print('Raw Dataset')
# print(df.isna().sum()/len(df)*100)
# # a = df.isna().sum()/len(df)*100
# # threshold = 15
# # variable =[]
# # col = df.columns
# # for i in range(0, len(df.columns)):
# #     if a[i]<= threshold:
# #         variable.append(col[i])
# #
# # df_clean = df[variable]
# # print('Cleaned Dataset')
# # print(df_clean.isna().sum()/len(df_clean)*100)
# num_type = ['int16','int32', 'int64','float16','float32', 'float64']
# numeric = df.select_dtypes(include=num_type)
# normalize = normalize(numeric)
# numeric_normalized = pd.DataFrame(normalize, columns=[f'feature{i} ' for i in range(1,5)])
#
# var = numeric_normalized.var()
# var_normalized = numeric_normalized.var()*100/np.max(var)
# threshold = 1e-3
#
# variable =[]
# col = df.columns
# for i in range(len(numeric_normalized.columns)):
#     if var_normalized[i]<= threshold:
#         variable.append(i)
# normalized_clean = numeric.drop(numeric.columns[variable], aixs=1)
# normalize = scaler.fit_transform(numeric)
# numeric_normalized = pd.DataFrame(normalize)

