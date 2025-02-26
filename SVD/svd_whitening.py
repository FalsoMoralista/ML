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

U, S, V_T = torch.linalg.svd(X)

print('S Size:', S.size())
print(S)

u, s, v_T = svd_helper.SVD()

print('S Size:', s.size())
print(s)

print('V Size:', v_T.size())
print(V_T)



print('difference:', torch.linalg.matrix_norm(U - u, ord='fro'))
print('difference:', torch.linalg.matrix_norm(S - s, ord='fro'))
print('difference:', torch.linalg.matrix_norm(V_T - v_T, ord='fro'))



