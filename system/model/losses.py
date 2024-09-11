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

    def forward(
        self,
        anchor,
        positive_old,
        negative_old,
        positive_new,
        negative_new,
        positive_old_weights,
        negative_old_weights,
    ):
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive_old = F.normalize(positive_old, p=2, dim=-1)
        negative_old = F.normalize(negative_old, p=2, dim=-1)
        positive_new = F.normalize(positive_new, p=2, dim=-1)
        negative_new = F.normalize(negative_new, p=2, dim=-1)

        pos_sim_old = torch.bmm(
            anchor.unsqueeze(1), positive_old.permute(0, 2, 1)
        ).squeeze(1)
        neg_sim_old = torch.bmm(
            anchor.unsqueeze(1), negative_old.permute(0, 2, 1)
        ).squeeze(1)

        weighted_pos_sim_old = pos_sim_old * positive_old_weights
        weighted_neg_sim_old = neg_sim_old * negative_old_weights

        pos_sim_new = torch.bmm(
            anchor.unsqueeze(1), positive_new.permute(0, 2, 1)
        ).squeeze(1)
        neg_sim_new = torch.bmm(
            anchor.unsqueeze(1), negative_new.permute(0, 2, 1)
        ).squeeze(1)

        pos_sim = torch.cat([weighted_pos_sim_old, pos_sim_new], dim=-1)
        neg_sim = torch.cat([weighted_neg_sim_old, neg_sim_new], dim=-1)

        logits = torch.cat([pos_sim, neg_sim], dim=-1) / self.temperature
        labels = torch.zeros(anchor.size(0), dtype=torch.long).to(anchor.device)

        loss = F.cross_entropy(logits, labels)
        return loss
