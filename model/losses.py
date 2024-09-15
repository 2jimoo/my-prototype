import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List


torch.autograd.set_detect_anomaly(True)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor: Tensor, positive: List[Tensor], negative: List[Tensor]):
        pos_sim = self.cosine_similarity(anchor, positive) / self.temperature
        neg_sim = self.cosine_similarity(anchor, negative) / self.temperature

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss


class CompatibleInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor, positive_old, negative_old, positive_new, negative_new):
        positive = torch.cat([positive_old, positive_new], dim=-1)
        negative = torch.cat([negative_old, negative_new], dim=-1)

        pos_sim = self.cosine_similarity(anchor, positive) / self.temperature
        neg_sim = self.cosine_similarity(anchor, negative) / self.temperature

        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss


class NCLCompatibleInfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(NCLCompatibleInfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor: Tensor,
        positives: List[Tensor],
        negatives: List[Tensor],
        positive_weights: List[float],
        negative_weights: List[float],
    ):
        # Ensure all tensors are on the same device
        device = anchor.device

        # Convert weights from lists to tensors and move to device
        positive_weights = torch.tensor(
            positive_weights, device=device, dtype=torch.float32
        )
        negative_weights = torch.tensor(
            negative_weights, device=device, dtype=torch.float32
        )

        # Normalize the anchor, positives, and negatives
        # print(f'# of positives :{len(positives)}')
        # print(f'# of negatives :{len(negatives)}')
        anchor = F.normalize(anchor, p=2, dim=-1)
        positives = torch.stack(
            [F.normalize(tensor, p=2, dim=-1) for tensor in positives], dim=0
        ).to(device)
        negatives = torch.stack(
            [F.normalize(tensor, p=2, dim=-1) for tensor in negatives], dim=0
        ).to(device)

        # Compute similarities for old positives and negatives
        # print(f"NCLCompatibleInfoNCELoss.forward")
        # print(f"anchor size: {anchor.shape}")
        # print(f"positives size: {positives.shape}")
        # print(f"negatives size: {negatives.shape}")

        # Expand dimensions for broadcasting
        anchor_expanded = anchor.clone().unsqueeze(1)  # [1, 1, 768]

        # Ensure positives and negatives are in the shape [num_samples, embedding_dim, 1]
        positives_permuted = positives.permute(1, 2, 0)  # [1, 768, num_positives]
        negatives_permuted = negatives.permute(1, 2, 0)  # [1, 768, num_negatives]

        # Compute similarity scores
        pos_sim = torch.bmm(
            anchor_expanded, positives_permuted
        )  # [1, 1, num_positives]
        neg_sim = torch.bmm(
            anchor_expanded, negatives_permuted
        )  # [1, 1, num_negatives]

        # Squeeze to remove the singleton dimension
        pos_sim = pos_sim.squeeze(1)  # [1, num_positives]
        neg_sim = neg_sim.squeeze(1)  # [1, num_negatives]

        # Apply weights to similarities
        weighted_pos_sim = pos_sim * positive_weights  # [1, num_positives]
        weighted_neg_sim = neg_sim * negative_weights  # [1, num_negatives]

        # Concatenate positive and negative similarities
        logits = (
            torch.cat([weighted_pos_sim, weighted_neg_sim], dim=-1) / self.temperature
        )
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=device)  # [1]

        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss
