import pandas as pd
from svd import SVD_Helper
import torch

df = pd.read_csv('utils/data/iris.csv')
header = list(df.columns.values)
df = df.drop(columns=header[len(header)-1])
print('Data Frame:', df)
X = torch.tensor(df.values)
print('Tensor:', X.size())

svd_helper = SVD_Helper(m=X.size(0), n=X.size(1))
svd_helper.train(X=X, no_steps=10000)


