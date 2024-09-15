from functions import (
    calculate_term,
    calculate_term_regl,
    cosine_search,
)
from cluster import ClusterInstance, ActiveClusterFeatureVector
from model import DenseEncoder
from typing import List
import torch

torch.autograd.set_detect_anomaly(True)


class Strategy:
    def __init__(self, encoder: DenseEncoder) -> None:
        self.encoder = encoder

    def get_repr_drift(self, prototype: ClusterInstance):
        pass

    def get_closest_cluster_indice(
        self,
        query: ClusterInstance,
        k,
        data: List[ActiveClusterFeatureVector],
        vector_dim=768,
    ):
        pass

    def get_closet_instance_indice(
        self,
        query: ActiveClusterFeatureVector,
        k,
        data: List[ClusterInstance],
        vector_dim=768,
    ):
        pass

    def get_distance(self, x: ClusterInstance, centroid: ActiveClusterFeatureVector):
        pass


class MeanPoolingCosineSimilartyStrategy(Strategy):
    def __init__(self, encoder: DenseEncoder) -> None:
        super().__init__(encoder)

    def get_repr_drift(self, prototype: ClusterInstance):
        E0, E_cur = (
            prototype.mean_emb.clone(),
            self.encoder.encode(prototype.passage)[0].clone(),
        )
        # 유클리드 거리(2-norm)로 비교? 코사인으로 비교?
        distance = torch.norm(E0 - E_cur).item()
        # print(f'get_repr_drift distance:{distance}')
        return distance

    def get_closest_cluster_indice(
        self,
        query: ClusterInstance,
        k,
        data: List[ActiveClusterFeatureVector],
        vector_dim=768,
    ):
        query = query.mean_emb.clone()
        embeddings = [c.get_mean() for c in data]
        return cosine_search(
            query=query, k=k, documents=embeddings, vector_dim=vector_dim
        )

    def get_closet_instance_indice(
        self,
        query: ActiveClusterFeatureVector,
        k,
        data: List[ClusterInstance],
        vector_dim=768,
    ):
        query = query.get_mean()
        embeddings = [x.mean_emb.clone() for x in data]
        # print(f"get_closet_instance_indice query shape: {query.shape}")
        # print(f"get_closet_instance_indice embeddings len: {len(embeddings)}")
        # print(f"get_closet_instance_indice embeddings shape: {embeddings[0].shape}")
        return cosine_search(
            query=query, k=k, documents=embeddings, vector_dim=vector_dim
        )

    def get_distance(self, x: ClusterInstance, centroid: ActiveClusterFeatureVector):
        # calculate_cosine_similarity
        # print(f"get_distance | x: {x.mean_emb.shape}, centoid centor vector:{centroid.get_mean().shape}")
        return torch.norm(x.mean_emb.clone() - centroid.get_mean())


class TokenEmbeddingsTermSimilartyStrategy(Strategy):
    def __init__(self) -> None:
        self.similirity_func = calculate_term


class TokenEmbeddingsTermReglSimilartyStrategy(Strategy):
    def __init__(self) -> None:
        self.similirity_func = calculate_term_regl
