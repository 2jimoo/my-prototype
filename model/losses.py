import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor, positive, negative):
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

    def forward(self, anchor, positives, negatives, positive_weights, negative_weights):
        anchor = F.normalize(anchor, p=2, dim=-1)
        positives = F.normalize(positives, p=2, dim=-1)
        negatives = F.normalize(negatives, p=2, dim=-1)

        pos_sim_old = torch.bmm(
            anchor.unsqueeze(1), positives.permute(0, 2, 1)
        ).squeeze(1)
        neg_sim_old = torch.bmm(
            anchor.unsqueeze(1), negatives.permute(0, 2, 1)
        ).squeeze(1)

        weighted_pos_sim = pos_sim_old * positive_weights
        weighted_neg_sim = neg_sim_old * negative_weights

        pos_sim = torch.cat([weighted_pos_sim], dim=-1)
        neg_sim = torch.cat([weighted_neg_sim], dim=-1)

        logits = torch.cat([pos_sim, neg_sim], dim=-1) / self.temperature
        labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)

        loss = F.cross_entropy(logits, labels)
        return loss
