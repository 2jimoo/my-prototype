import torch
import numpy as np

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def cosine_search(query, k, documents, vector_dim=768):
    data_tensor = torch.cat(documents, dim=0).to(device)
    query_tensor = torch.tensor(query, dtype=torch.float32).to(device)
    # print(f"data_tensor shape: {data_tensor.shape}")
    # print(f"query_tensor shape: {query_tensor.shape}")
    similarities = torch.matmul(query_tensor, data_tensor.T)
    _, indices = torch.topk(similarities, k, dim=1)
    indices = indices.cpu().numpy()[0]
    # print(f'cosine_search indices:{indices}')
    return indices


def ncl_lsh(query, k, data, vector_dim=768):
    pass
