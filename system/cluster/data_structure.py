from torch import Tensor
from typing import List
import numpy as np
from dataclasses import dataclass


@dataclass
class ClusterInstance:
    id: int
    passage: str
    mean_emb: Tensor
    token_embs: List[Tensor]


class ActiveClusterFeatureVector:
    def __init__(self, centroid_id, centroid_mean_emb, current_time_step):
        self.centroid_id = centroid_id
        self.n = 1  # 클러스터 내 객체 수
        self.S1 = np.zeros(centroid_mean_emb)  # 클러스터 내 객체 임베딩 선형 합
        self.S2 = np.zeros(centroid_mean_emb)  # 클러스터 내 객체 임베딩 제곱 합
        self.t = current_time_step  # 마지막으로 데이터가 도착한 시간
        self.u = 0.1

    def get_centroid_id(self):
        return self.centroid_id

    def update(self, embedding, t):
        """클러스터 특성 벡터를 업데이트 (객체 추가 시)"""
        self.n += 1
        self.S1 += embedding
        self.S2 += embedding**2
        self.t = t

    def get_weight(self, current_time):
        return np.exp((self.t - current_time) / self.u)

    def get_mean(self):
        """클러스터의 평균 임베딩(centroid) 계산"""
        return self.S1 / self.n

    def get_std(self):
        """클러스터의 표준편차(평균적으로 centroid에서 얼마나 떨어졌는가) 계산"""
        mean = self.get_mean()
        variance = (self.S2 / self.n) - (mean**2)
        std = np.sqrt(variance)
        return std


class DeactiveClusterFeatureVector:
    def __init__(self, centroid_id, n, S1, S2, prototype, E0, w):
        self.centroid_id = centroid_id
        self.n = n
        self.S1 = S1
        self.S2 = S2
        self.prototype = prototype
        self.E0 = E0
        self.w = w

    def get_centroid_id(self):
        return self.centroid_id

    def get_mean(self):
        return self.S1 / self.n

    def get_std(self):
        mean = self.get_mean()
        variance = (self.S2 / self.n) - (mean**2)
        std = np.sqrt(variance)
        return std


# class NCLActiveClusterFeatureVector:
#     def __init__(self,centroid_id, centroid_mean_emb):
#         self.centroid_id = centroid_id
#         self.n = 0
#         self.S1 = np.zeros(centroid_mean_emb)
#         self.S2 = np.zeros(centroid_mean_emb)
#         self.t = 0
#         self.u = 0.1
#         self.T1= 0
#         self.T2=0
