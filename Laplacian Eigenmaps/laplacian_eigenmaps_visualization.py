import torch
import torch.nn.functional as F

import faiss



data_path = '/home/rtcalumby/adam/luciano/DeepCluster/cache/plantnet_300k/dinov2_vit_large'
proj_embed_dim = 1024
k=5

def load_data(path, proj_embed_dim):
    cached_data = torch.load(path + f'/cached_features_{proj_embed_dim}_epoch_0.pt')
    return cached_data

def build_tensor(data):

    tensor_list = [tensor for cls in data.keys() for tensor in data[cls]]
    data_as_tensor = torch.stack(tensor_list)
    return data_as_tensor

data = load_data(data_path, proj_embed_dim)
data_matrix = build_tensor(data)
data_matrix = F.normalize(data_matrix, p=2, dim=1)
print(data_matrix.size())

# IP stands for "inner product". If you have normalized vectors, the inner product becomes cosine similarity.
index = faiss.IndexFlatIP(proj_embed_dim) 
index.add(data_matrix)  

similarities, neighbors = index.search(data_matrix, k + 1)  # k+1 to exclude self

# Convert to PyTorch tensors
similarities = torch.tensor(similarities[:, 1:])  # Exclude self-similarity
neighbors = torch.tensor(neighbors[:, 1:], dtype=torch.long)

# Convert inner product to cosine distance
cosine_distances = 1 - similarities  # d_cos(x, y) = 1 - cos(x, y)
print(cosine_distances.size())