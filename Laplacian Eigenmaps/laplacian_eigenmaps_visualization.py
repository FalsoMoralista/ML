import torch
import torch.nn.functional as F

import faiss
import faiss.contrib.torch_utils



# Settings
data_path = '/home/rtcalumby/adam/luciano/DeepCluster/cache/plantnet_300k/dinov2_vit_large'
proj_embed_dim = 1024
k=5
device = 'cuda:0'
############################################################################################################

def load_data(path, proj_embed_dim):
    cached_data = torch.load(path + f'/cached_features_{proj_embed_dim}_epoch_0.pt')
    return cached_data

def build_tensor(data):
    tensor_list = [tensor for cls in data.keys() for tensor in data[cls]]
    data_as_tensor = torch.stack(tensor_list)
    return data_as_tensor

def build_sparse_graph(neighbors, k=10):

    N = neighbors.shape[0]
    row_idx = []
    col_idx = []
    values = []

    for i in range(N):
        for j in neighbors[i][1:]:  # skip self-connection
            row_idx.append(i)
            col_idx.append(j)
            values.append(1.0)  # binary adjacency

    # Convert to PyTorch sparse tensor
    adjacency_matrix = torch.sparse_coo_tensor(
        torch.tensor([row_idx, col_idx]),
        torch.tensor(values),
        (N, N),
        dtype=torch.float32        
    )
    return adjacency_matrix

def graph_laplacian(adjacency_matrix, normalize=True):
    # Compute Degree Matrix (Diagonal with node degrees)
    degrees = torch.sparse.sum(adjacency_matrix, dim=1).to_dense()
    D = torch.diag(degrees)  # Dense degree matrix

    # Unnormalized Laplacian: L = D - W
    L = D - adjacency_matrix.to_dense()

    if normalize:
        # Compute D^(-1/2)
        D_sqrt_inv = torch.diag(1.0 / torch.sqrt(degrees + 1e-8))  # Avoid division by zero
        L = D_sqrt_inv @ L @ D_sqrt_inv  # Normalized Laplacian
    return L  # return sparse matrix

data = load_data(data_path, proj_embed_dim)
data_matrix = build_tensor(data).to(device)
data_matrix = F.normalize(data_matrix, p=2, dim=1)
#print(data_matrix.size())

# Setup Faiss with gpu
# IP stands for "inner product". If you have normalized vectors, the inner product becomes cosine similarity.
index = faiss.IndexFlatIP(proj_embed_dim) 
res = faiss.StandardGpuResources()  
index = faiss.index_cpu_to_gpu(res, 0, index)  # Move index to GPU

############################################################################################################
index.add(data_matrix)  
similarities, neighbors = index.search(data_matrix, k + 1)  # k+1 to exclude self

adjacency_matrix = build_sparse_graph(neighbors=neighbors, k=k)
L = graph_laplacian(adjacency_matrix, normalize=False)

print(L.size())

# Convert inner product to cosine distance
#cosine_distances = 1 - similarities  # d_cos(x, y) = 1 - cos(x, y)
#print(cosine_distances.size())