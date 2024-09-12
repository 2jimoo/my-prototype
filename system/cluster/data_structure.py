from torch import Tensor
from typing import List
import numpy as np
from dataclasses import dataclass


@dataclass
class SamplingResult:
    positive_embeddings: List[Tensor]
    negative_embeddings: List[Tensor]
    positive_weights: List[float]
    negative_weights: List[float]


@dataclass
class ClusterInstance:
    id: int
    passage: str
    mean_emb: Tensor
    token_embs: List[Tensor]
    cluster_id: int


class ActiveClusterFeatureVector:
    def __init__(self, current_time_step=None, centroid: ClusterInstance = None):
        if centroid:
            self.centroid_id = centroid.id
            self.n = 1
            self.S1 = np.zeros(centroid.mean_emb)
            self.S2 = np.zeros(centroid.mean_emb)
            self.prototype = centroid
        if current_time_step:
            self.t = current_time_step
        self.u = 0.1

    def update_prototype(self, prototype: ClusterInstance):
        self.prototype = prototype

    def get_centroid_id(self):
        return self.centroid_id

    def update(self, embedding, t):
        self.n += 1
        self.S1 += embedding
        self.S2 += embedding**2
        self.t = t

    def get_weight(self, current_time):
        return np.exp((self.t - current_time) / self.u)

    def get_mean(self):
        return self.S1 / self.n

    def get_rms(self):
        return np.sqrt(self.S2 / self.n)

    def get_std(self):
        mean = self.get_mean()
        variance = (self.S2 / self.n) - (mean**2)
        std = np.sqrt(variance)
        return std


class DeactiveClusterFeatureVector:
    def __init__(self, centroid_id, n, S1, S2, prototype: ClusterInstance):
        self.centroid_id = centroid_id
        self.n = n
        self.S1 = S1
        self.S2 = S2
        self.prototype = prototype

    def get_centroid_id(self):
        return self.centroid_id

    def get_mean(self):
        return self.S1 / self.n

    def get_std(self):
        mean = self.get_mean()
        variance = (self.S2 / self.n) - (mean**2)
        std = np.sqrt(variance)
        return std

    def build_ACFV(self) -> ActiveClusterFeatureVector:
        reactivated = ActiveClusterFeatureVector()
        reactivated.centroid_id = self.centroid_id
        reactivated.n = self.n
        reactivated.S1 = self.S1
        reactivated.S2 = self.S2
        reactivated.t = self.t
        reactivated.prototype = self.prototype
        return reactivated
