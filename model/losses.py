import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

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
        # print("forward에 입력 텐서가 모두 require_grad=True여야 한답디다^^..")
        # print(anchor.requires_grad)
        # for tensor in positives:
        #     print(tensor.requires_grad)
        # for tensor in negatives:
        #     print(tensor.requires_grad)

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
        anchor = F.normalize(anchor, p=2, dim=-1)
        positives = torch.stack(
            [F.normalize(tensor, p=2, dim=-1) for tensor in positives], dim=0
        ).to(device)
        negatives = torch.stack(
            [F.normalize(tensor, p=2, dim=-1) for tensor in negatives], dim=0
        ).to(device)

        # Compute similarities for old positives and negatives
        pos_sim_old = torch.bmm(
            anchor.unsqueeze(1), positives.permute(0, 2, 1)
        ).squeeze(1)
        neg_sim_old = torch.bmm(
            anchor.unsqueeze(1), negatives.permute(0, 2, 1)
        ).squeeze(1)

        # Apply weights to similarities
        weighted_pos_sim = pos_sim_old * positive_weights
        weighted_neg_sim = neg_sim_old * negative_weights

        # Concatenate positive and negative similarities
        pos_sim = torch.cat([weighted_pos_sim], dim=-1)
        neg_sim = torch.cat([weighted_neg_sim], dim=-1)

        # Calculate logits and loss
        logits = torch.cat([pos_sim, neg_sim], dim=-1) / self.temperature
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=device)

        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        # print(f"loss type: {type(loss)}")
        # print(f"loss.requires_grad: {loss.requires_grad}")
        return loss
