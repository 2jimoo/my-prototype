import torch
from .similarities import calculate_term, calculate_term_regl

torch.autograd.set_detect_anomaly(True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def cosine_search(query, k, documents, vector_dim=768):
    data_tensor = torch.cat(documents, dim=0).to(device)
    query_tensor = query.to(device)
    # print(f"data_tensor shape: {data_tensor.shape}")
    # print(f"query_tensor shape: {query_tensor.shape}")
    similarities = torch.matmul(query_tensor, data_tensor.T)
    _, indices = torch.topk(similarities, k, dim=1)
    indices = indices.cpu().numpy()[0]
    if len(indices) != len(set(indices)):
        print(f"cosine_search indices:{indices}")
    return indices


def term_search(query, k, documents, vector_dim=768):
    query_tensor = query.to(device)
    similarities = []
    for doc in documents:
        similarities.append(calculate_term(E_q=query_tensor, E_d=doc))
    similarities_tensor = torch.tensor(similarities).to(device)
    _, indices = torch.topk(similarities_tensor, k, dim=0)
    indices = indices.cpu().numpy()
    if len(indices) != len(set(indices)):
        print(f"term_search indices:{indices}")
    return indices


def term_regl_search(query, k, documents, vector_dim=768):
    query_tensor = query.to(device)
    similarities = []
    for doc in documents:
        similarities.append(calculate_term_regl(E_q=query_tensor, E_d=doc))
    similarities_tensor = torch.tensor(similarities).to(device)
    _, indices = torch.topk(similarities_tensor, k, dim=0)
    indices = indices.cpu().numpy()
    if len(indices) != len(set(indices)):
        print(f"term_regl_search indices:{indices}")
    return indices


def term_lsh(query, k, data, vector_dim=768):
    pass


def term_regl_lsh(query, k, data, vector_dim=768):
    pass
