import torch
from .similarities import calculate_term, calculate_term_regl

torch.autograd.set_detect_anomaly(True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def cosine_search(query, k, documents, vector_dim=768):
    data_tensor = torch.cat(documents, dim=0).to(device)
    # 복사를 안 하고 inplace 연산을 하면,, 에러가 생겨요...
    query_tensor = query.clone()
    query_tensor = query_tensor.to(device)
    # print(f"data_tensor shape: {data_tensor.shape}")
    # print(f"query_tensor shape: {query_tensor.shape}")
    similarities = torch.matmul(query_tensor, data_tensor.T)
    _, indices = torch.topk(similarities, k, dim=1)
    indices = indices.cpu().numpy()[0]
    # print(f'cosine_search indices:{indices}')
    return indices


def term_search(query, k, documents, vector_dim=768):
    query_tensor = query.clone().to(device)
    similarities = []
    for doc in documents:
        similarities.append(calculate_term(E_q=query_tensor, E_d=doc))
    _, indices = torch.topk(similarities, k, dim=1)
    indices = indices.cpu().numpy()[0]


def term_regl_search(query, k, documents, vector_dim=768):
    query_tensor = query.clone().to(device)
    similarities = []
    for doc in documents:
        similarities.append(calculate_term_regl(E_q=query_tensor, E_d=doc))
    _, indices = torch.topk(similarities, k, dim=1)
    indices = indices.cpu().numpy()[0]


def term_lsh(query, k, data, vector_dim=768):
    pass


def term_regl_lsh(query, k, data, vector_dim=768):
    pass
