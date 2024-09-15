import torch
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def calculate_cosine_similarity(E_q, E_d):
    E_q = E_q.to(device)
    E_d = E_d.to(device)
    cosine_sim = F.cosine_similarity(E_q, E_d)
    return cosine_sim


def calculate_term(E_q, E_d):
    if isinstance(E_q, list):
        E_q = torch.stack(E_q, dim=0)
    if isinstance(E_d, list):
        E_d = torch.stack(E_d, dim=0)
    E_q = E_q.to(device)
    E_d = E_d.to(device)
    scores = torch.matmul(E_q, E_d.t())
    max_scores, _ = torch.max(scores, dim=1)
    S_qd = max_scores.sum().item()
    return S_qd


def calculate_term_regl(E_q, E_d):
    if isinstance(E_q, list):
        E_q = torch.stack(E_q, dim=0)
    if isinstance(E_d, list):
        E_d = torch.stack(E_d, dim=0)
    E_q = E_q.to(device)
    E_d = E_d.to(device)
    E_q_normalized = torch.nn.functional.normalize(E_q, p=2, dim=1)
    E_d_normalized = torch.nn.functional.normalize(E_d, p=2, dim=1)

    cosine_sim_matrix = torch.matmul(E_q_normalized, E_d_normalized.T)
    max_scores, _ = torch.max(cosine_sim_matrix, dim=1)
    S_qd_score = max_scores.mean().item()
    return S_qd_score
