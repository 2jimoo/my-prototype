from torch import Tensor
from typing import List
import numpy as np
from dataclasses import dataclass
import torch


torch.autograd.set_detect_anomaly(True)


@dataclass
class SamplingResult:
    positive_embeddings: List[Tensor]
    negative_embeddings: List[Tensor]
    positive_weights: List[float]
    negative_weights: List[float]


@dataclass
class ClusterInstance:
    id: int = -1
    passage: str = ""
    mean_emb: Tensor = None
    token_embs: List[Tensor] = None

    def __str__(self):
        return f"ClusterInstance(id: {self.id}, passage: {self.passage[:5]})"


class ActiveClusterFeatureVector:
    def __init__(
        self, centroid_id, current_time_step=0, centroid: ClusterInstance = None
    ):
        self.centroid_id = centroid_id
        self.u = 10
        if centroid:
            self.n = 1
            self.S1: Tensor = torch.sum(centroid.mean_emb, dim=0)
            self.S2: Tensor = torch.sum(centroid.mean_emb**2, dim=0)
            self.prototype = centroid
        self.t = current_time_step
        # print(f"ActiveClusterFeatureVector __init__: {self.S2}")

    def __str__(self):
        return f"ActiveClusterFeatureVector(centroid_id: {self.centroid_id}, n: {self.n}, t:{self.t})"

    def update_prototype(self, prototype: ClusterInstance):
        self.prototype = prototype

    def get_centroid_id(self):
        return self.centroid_id

    def update(self, embedding: Tensor, t):
        self.n += 1
        self.S1 = self.S1.clone() + torch.mean(embedding, dim=0)
        self.S2 = self.S2.clone() + torch.mean(embedding**2, dim=0)
        self.t = t

    def get_weight(self, current_time):
        weight = np.exp((self.t - current_time) / self.u)
        # print(f'centroid_id: {self.centroid_id} | weight: {weight}')
        return weight

    def get_mean(self):
        # (num_sample=1, vector_size=768)로 맞춰주기 위해 unsqueeze(0)추가
        return (self.S1 / self.n).unsqueeze(0)

    def get_rms(self):
        # TODO 이것도 norm? torch.sqrt(self.S2 / self.n)
        rms = torch.norm(torch.sqrt(self.S2 / self.n))
        print(f"centroid_id: {self.centroid_id} | rms: {rms}")
        return rms

    def get_std_norm(self):
        std = self.get_std()
        mean_distance = torch.norm(std).item()
        return mean_distance

    def get_std(self):
        mean = self.get_mean()
        variance = (self.S2 / self.n) - (mean**2)
        std = torch.sqrt(variance)
        return std


class DeactiveClusterFeatureVector:
    def __init__(
        self, centroid_id, n, S1: Tensor, S2: Tensor, prototype: ClusterInstance
    ):
        self.centroid_id = centroid_id
        self.n = n
        self.S1 = S1.clone()
        self.S2 = S2.clone()
        self.prototype = prototype

    def get_centroid_id(self):
        return self.centroid_id

    def get_mean(self):
        return self.S1 / self.n

    def get_std(self):
        mean = self.get_mean()
        variance = (self.S2 / self.n) - (mean**2)
        std = torch.sqrt(variance)
        return std

    def get_std_norm(self):
        std = self.get_std()
        mean_distance = torch.norm(std).item()
        return mean_distance

    def build_ACFV(self) -> ActiveClusterFeatureVector:
        reactivated = ActiveClusterFeatureVector()
        reactivated.centroid_id = self.centroid_id
        reactivated.n = self.n
        reactivated.S1 = self.S1.clone()
        reactivated.S2 = self.S2.clone()
        reactivated.t = self.t
        reactivated.prototype = self.prototype
        return reactivated
