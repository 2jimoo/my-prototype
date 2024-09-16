from torch import Tensor
from typing import List
import numpy as np
from dataclasses import dataclass
import torch
from functions import calculate_term, calculate_term_regl

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
        self,
        centroid_id,
        current_time_step=0,
    ):
        self.centroid_id: int = centroid_id
        self.u = 10
        self.t = current_time_step

    def get_centroid_id(self):
        return self.centroid_id

    def __str__(self):
        return f"ActiveClusterFeatureVector(centroid_id: {self.centroid_id}, n: {self.n}, t:{self.t})"

    def update_prototype(self, prototype: ClusterInstance):
        self.prototype = prototype

    def get_weight(self, current_time):
        weight = np.exp((self.t - current_time) / self.u)
        return weight

    def update(self, x: ClusterInstance, t):
        pass

    def get_mean(self):
        pass

    def get_rms(self):
        pass

    def get_std_norm(self):
        pass

    def get_std(self):
        pass


class MeanEmbActiveClusterFeatureVector(ActiveClusterFeatureVector):
    def __init__(
        self, centroid_id, current_time_step=0, centroid: ClusterInstance = None
    ):
        super().__init__(centroid_id=centroid_id, current_time_step=current_time_step)
        if centroid:
            self.n = 1
            self.V1: Tensor = torch.sum(centroid.mean_emb, dim=0)
            self.V2: Tensor = torch.sum(centroid.mean_emb**2, dim=0)
            self.prototype = centroid

    def update(self, x: ClusterInstance, t):
        self.n += 1
        self.V1 = self.V1 + torch.mean(x.mean_emb, dim=0)
        self.V2 = self.V2 + torch.mean(x.mean_emb**2, dim=0)
        self.t = t

    def get_mean(self):
        # (num_sample=1, vector_size=768)로 맞춰주기 위해 unsqueeze(0)추가
        return (self.V1 / self.n).unsqueeze(0)

    def get_rms(self):
        # TODO 이것도 norm? torch.sqrt(self.V2 / self.n)
        rms = torch.norm(torch.sqrt(self.V2 / self.n))
        # print(f"centroid_id: {self.centroid_id} | rms: {rms}")
        return rms

    def get_std_norm(self):
        std = self.get_std()
        mean_distance = torch.norm(std).item()
        return mean_distance

    def get_std(self):
        mean = self.get_mean()
        variance = (self.V2 / self.n) - (mean**2)
        std = torch.sqrt(variance)
        return std


class TokenEmbsActiveClusterFeatureVector(ActiveClusterFeatureVector):
    def __init__(
        self, centroid_id, current_time_step=0, centroid: ClusterInstance = None
    ):
        super().__init__(centroid_id=centroid_id, current_time_step=current_time_step)
        if centroid:
            self.n = 1
            # 헉 나 자신이면 mean, var 초기화할 값이 없어..
            self.V1 = calculate_term(centroid.token_embs, centroid.token_embs) / 2.0
            self.V2 = self.V1**2
            self.prototype = centroid

    def update(self, x: ClusterInstance, t):
        self.n += 1
        self.V1 = self.V1 + calculate_term(x.token_embs, self.prototype.token_embs)
        self.V2 = self.V2 + calculate_term(x.token_embs, self.prototype.token_embs) ** 2
        self.t = t

    def get_mean(self):
        return self.V1 / self.n

    def get_rms(self):
        rms = np.sqrt(self.V2 / self.n)
        print(f"centroid_id: {self.centroid_id} | rms: {rms}")
        return rms

    def get_std_norm(self):
        return self.get_std()

    def get_std(self):
        mean = self.get_mean()
        variance = (self.V2 / self.n) - (mean**2)
        std = np.sqrt(variance)
        return std


class DeactiveClusterFeatureVector:
    def __init__(self, centroid_id, n, prototype: ClusterInstance, t):
        self.centroid_id = centroid_id
        self.n = n
        self.prototype = prototype
        self.t = t

    def get_centroid_id(self):
        pass

    def get_mean(self):
        pass

    def get_std(self):
        pass

    def get_std_norm(self):
        pass

    def build_ACFV(self) -> ActiveClusterFeatureVector:
        pass


class MeanDeactiveClusterFeatureVector(DeactiveClusterFeatureVector):
    def __init__(
        self, centroid_id, n, V1: Tensor, V2: Tensor, prototype: ClusterInstance, t
    ):
        super().__init__(centroid_id=centroid_id, n=n, prototype=prototype, t=t)
        self.V1 = V1
        self.V2 = V2

    def get_centroid_id(self):
        return self.centroid_id

    def get_mean(self):
        return self.V1 / self.n

    def get_std(self):
        mean = self.get_mean()
        variance = (self.V2 / self.n) - (mean**2)
        std = torch.sqrt(variance)
        return std

    def get_std_norm(self):
        std = self.get_std()
        mean_distance = torch.norm(std).item()
        return mean_distance

    def build_ACFV(self) -> MeanEmbActiveClusterFeatureVector:
        reactivated = MeanEmbActiveClusterFeatureVector()
        reactivated.centroid_id = self.centroid_id
        reactivated.n = self.n
        reactivated.V1 = self.V1
        reactivated.V2 = self.V2
        reactivated.t = self.t
        reactivated.prototype = self.prototype
        return reactivated


class TokenEmbsDeactiveClusterFeatureVector(DeactiveClusterFeatureVector):
    def __init__(self, centroid_id, n, V1, V2, prototype: ClusterInstance, t):
        super().__init__(centroid_id=centroid_id, n=n, prototype=prototype, t=t)
        self.V1 = V1
        self.V2 = V2

    def get_centroid_id(self):
        return self.centroid_id

    def get_mean(self):
        return self.V1 / self.n

    def get_std(self):
        mean = self.get_mean()
        variance = (self.V2 / self.n) - (mean**2)
        std = np.sqrt(variance)
        return std

    def get_std_norm(self):
        return self.get_std()

    def build_ACFV(self) -> TokenEmbsActiveClusterFeatureVector:
        reactivated = TokenEmbsActiveClusterFeatureVector()
        reactivated.centroid_id = self.centroid_id
        reactivated.n = self.n
        reactivated.V1 = self.V1
        reactivated.V2 = self.V2
        reactivated.t = self.t
        reactivated.prototype = self.prototype
        return reactivated
