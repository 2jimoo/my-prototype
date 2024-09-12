from functions import (
    calculate_cosine_similarity,
    calculate_term,
    calculate_term_regl,
    faiss_search,
    ncl_lsh,
)
from cluster import ClusterInstance, ActiveClusterFeatureVector
from model import DenseEncoder
from typing import List
import numpy as np


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
        super.__init__(self, encoder)

    def get_repr_drift(self, prototype: ClusterInstance):
        E0, E_cur = (
            prototype.mean_emb,
            self.encoder.encode(prototype.passage)[0],
        )
        return calculate_cosine_similarity(E0, E_cur)

    def get_closest_cluster_indice(
        self,
        query: ClusterInstance,
        k,
        data: List[ActiveClusterFeatureVector],
        vector_dim=768,
    ):
        query = query.mean_emb
        embeddings = np.array([c.get_mean() for c in data])
        return faiss_search(query=query, k=k, data=embeddings, vector_dim=vector_dim)

    def get_closet_instance_indice(
        self,
        query: ActiveClusterFeatureVector,
        k,
        data: List[ClusterInstance],
        vector_dim=768,
    ):
        query = query.get_mean()
        embeddings = np.array([x.mean_emb for x in data])
        return faiss_search(query=query, k=k, data=embeddings, vector_dim=vector_dim)

    def get_distance(self, x: ClusterInstance, centroid: ActiveClusterFeatureVector):
        return calculate_cosine_similarity(x.mean_emb, centroid.get_mean())


class TokenEmbeddingsTermSimilartyStrategy(Strategy):
    def __init__(self) -> None:
        self.similirity_func = calculate_term


class TokenEmbeddingsTermReglSimilartyStrategy(Strategy):
    def __init__(self) -> None:
        self.similirity_func = calculate_term_regl
