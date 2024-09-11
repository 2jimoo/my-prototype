import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def info_nce_loss(anchor, positives, negatives, temperature=0.1):
    anchor = anchor.to(device)
    positives = positives.to(device)
    negatives = negatives.to(device)

    # Normalize embeddings to unit length(코사인 유사도를 계산하기 위해 임베딩을 단위 벡터로 정규화)
    anchor = F.normalize(anchor, p=2, dim=-1)
    positives = F.normalize(positives, p=2, dim=-1)
    negatives = F.normalize(negatives, p=2, dim=-1)

    # Compute positive similarities (N, K) and negative similarities (N, M)
    pos_sim = torch.bmm(anchor.unsqueeze(1), positives.permute(0, 2, 1)).squeeze(1)
    neg_sim = torch.bmm(anchor.unsqueeze(1), negatives.permute(0, 2, 1)).squeeze(1)

    # Combine positive and negative similarities
    logits = torch.cat([pos_sim, neg_sim], dim=-1)  # Shape: (N, K + M)

    # Create labels for positive samples
    labels = torch.zeros(anchor.size(0), dtype=torch.long).to(device)  # Shape: (N,)

    # Apply temperature scaling
    logits /= temperature

    # Compute loss
    loss = F.cross_entropy(logits, labels)

    return loss.item()


def ncl_compatible_info_nce_loss(
    anchor,
    positive_old,
    negative_old,
    positive_new,
    negative_new,
    positive_old_weights,
    negative_old_weights,
    temperature=0.1,
):
    # Move tensors to MPS device
    anchor = anchor.to(device)
    positive_old = positive_old.to(device)
    negative_old = negative_old.to(device)
    positive_new = positive_new.to(device)
    negative_new = negative_new.to(device)
    positive_old_weights = positive_old_weights.to(device)
    negative_old_weights = negative_old_weights.to(device)

    # Normalize embeddings to unit length
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive_old = F.normalize(positive_old, p=2, dim=-1)
    negative_old = F.normalize(negative_old, p=2, dim=-1)
    positive_new = F.normalize(positive_new, p=2, dim=-1)
    negative_new = F.normalize(negative_new, p=2, dim=-1)

    # Compute similarities for old samples
    pos_sim_old = torch.bmm(anchor.unsqueeze(1), positive_old.permute(0, 2, 1)).squeeze(
        1
    )  # (N, K)
    neg_sim_old = torch.bmm(anchor.unsqueeze(1), negative_old.permute(0, 2, 1)).squeeze(
        1
    )  # (N, M)

    # Compute weighted similarities for old samples
    weighted_pos_sim_old = pos_sim_old * positive_old_weights  # (N, K)
    weighted_neg_sim_old = neg_sim_old * negative_old_weights  # (N, M)

    # Compute similarities for new samples
    pos_sim_new = torch.bmm(anchor.unsqueeze(1), positive_new.permute(0, 2, 1)).squeeze(
        1
    )  # (N, K_new)
    neg_sim_new = torch.bmm(anchor.unsqueeze(1), negative_new.permute(0, 2, 1)).squeeze(
        1
    )  # (N, M_new)

    # Combine old and new similarities
    pos_sim = torch.cat([weighted_pos_sim_old, pos_sim_new], dim=-1)  # (N, K + K_new)
    neg_sim = torch.cat([weighted_neg_sim_old, neg_sim_new], dim=-1)  # (N, M + M_new)

    # Create labels
    # For old positives, the label will be based on the weighted positive samples
    # For new positives, the label will be 0 as they are the true positives in the new set
    labels = torch.cat(
        [
            torch.zeros(anchor.size(0), dtype=torch.long).to(
                device
            ),  # Labels for new positives
            torch.ones(anchor.size(0), dtype=torch.long).to(device)
            * (weighted_pos_sim_old.size(1)),  # Labels for old positives
        ],
        dim=-1,
    )

    # Apply temperature scaling
    pos_sim /= temperature
    neg_sim /= temperature

    # Compute logits by concatenating positive and negative similarities
    logits = torch.cat([pos_sim, neg_sim], dim=-1)  # (N, K + K_new + M + M_new)

    # Compute loss
    loss = F.cross_entropy(logits, labels)

    return loss.item()
