from functions import (
    calculate_term,
    calculate_term_regl,
    cosine_search,
    term_search,
    term_regl_search,
)
from cluster import (
    ActiveClusterFeatureVector,
    DeactiveClusterFeatureVector,
    MeanEmbActiveClusterFeatureVector,
    MeanDeactiveClusterFeatureVector,
    TokenEmbsActiveClusterFeatureVector,
    TokenEmbsDeactiveClusterFeatureVector,
)
from cluster import ClusterInstance, ActiveClusterFeatureVector
from model import DenseEncoder
from typing import List
import torch

torch.autograd.set_detect_anomaly(True)


class Strategy:
    def __init__(self, encoder: DenseEncoder) -> None:
        self.encoder = encoder

    def get_search_func(self):
        return self.search_function

    def get_embedding(self, x):
        pass

    def is_assigned(self, centroid: ActiveClusterFeatureVector, distance):
        pass

    def is_repr_drifted(
        self, centroid: ActiveClusterFeatureVector, prototype: ClusterInstance
    ):
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

    def get_distances(self, x: ClusterInstance, instances: List[ClusterInstance]):
        pass

    def build_ActiveClusterFeatureVector(
        self, centroid_id, centroid: ClusterInstance, current_time_step
    ):
        pass

    def build_DeactiveClusterFeatureVector(
        self, centroid_id, n, V1, V2, prototype: ClusterInstance, t
    ):
        pass


class MeanPoolingCosineSimilartyStrategy(Strategy):
    def __init__(self, encoder: DenseEncoder) -> None:
        super().__init__(encoder)
        self.search_function = cosine_search

    def get_embedding(self, x):
        if isinstance(x, ClusterInstance):
            return x.mean_emb
        elif isinstance(x, ActiveClusterFeatureVector):
            return x.get_mean()
        else:
            raise TypeError("Unsupported type for 'x'")

    def is_assigned(self, centroid: ActiveClusterFeatureVector, distance):
        return centroid.get_rms() > distance

    def is_repr_drifted(
        self, centroid: ActiveClusterFeatureVector, prototype: ClusterInstance
    ):
        return centroid.get_std_norm() <= self.get_repr_drift(prototype)

    def get_repr_drift(self, prototype: ClusterInstance):
        E0, E_cur = (
            prototype.mean_emb,
            self.encoder.encode(prototype.passage)[0],
        )
        # 유클리드 거리(2-norm)로 비교? 코사인으로 비교?
        distance = torch.norm(E0 - E_cur).item()
        return distance

    def get_closest_cluster_indice(
        self,
        query: ClusterInstance,
        k,
        data: List[ActiveClusterFeatureVector],
        vector_dim=768,
    ):
        query = query.mean_emb
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
        embeddings = [x.mean_emb for x in data]
        return cosine_search(
            query=query, k=k, documents=embeddings, vector_dim=vector_dim
        )

    def get_distance(self, x: ClusterInstance, centroid: ActiveClusterFeatureVector):
        return torch.norm(x.mean_emb - centroid.get_mean())

    def get_distances(self, x: ClusterInstance, instances: List[ClusterInstance]):
        return [torch.norm(x.mean_emb - instance.mean_emb) for instance in instances]

    def build_ActiveClusterFeatureVector(
        self, centroid_id, centroid: ClusterInstance, current_time_step
    ):
        return MeanEmbActiveClusterFeatureVector(
            centroid_id=centroid_id,
            current_time_step=current_time_step,
            centroid=centroid,
        )

    def build_DeactiveClusterFeatureVector(
        self, centroid_id, n, V1, V2, prototype: ClusterInstance, t
    ):
        return MeanDeactiveClusterFeatureVector(centroid_id, n, V1, V2, prototype, t)


class TokenEmbeddingsTermSimilartyStrategy(Strategy):
    def __init__(self, encoder) -> None:
        super().__init__(encoder)
        self.search_function = term_search

    def get_embedding(self, x):
        if isinstance(x, ClusterInstance):
            return x.token_embs
        elif isinstance(x, ActiveClusterFeatureVector):
            return x.prototype.token_embs
        else:
            raise TypeError("Unsupported type for 'x'")

    def is_assigned(self, centroid: ActiveClusterFeatureVector, score):
        return centroid.get_rms() < score

    def is_repr_drifted(
        self, centroid: ActiveClusterFeatureVector, prototype: ClusterInstance
    ):
        return centroid.get_std_norm() > self.get_repr_drift(prototype)

    def get_repr_drift(self, prototype: ClusterInstance):
        E0, E_cur = (
            prototype.token_embs,
            self.encoder.encode(prototype.passage)[1],
        )
        score = calculate_term(E0, E_cur)
        return score

    def get_closest_cluster_indice(
        self,
        query: ClusterInstance,
        k,
        data: List[ActiveClusterFeatureVector],
        vector_dim=768,
    ):
        query = query.token_embs
        embeddings = [c.prototype.token_embs for c in data]
        return term_search(
            query=query, k=k, documents=embeddings, vector_dim=vector_dim
        )

    def get_closet_instance_indice(
        self,
        query: ActiveClusterFeatureVector,
        k,
        data: List[ClusterInstance],
        vector_dim=768,
    ):
        query = query.prototype.token_embs
        embeddings = [c.token_embs for c in data]
        return term_search(
            query=query, k=k, documents=embeddings, vector_dim=vector_dim
        )

    def get_distance(self, x: ClusterInstance, centroid: ActiveClusterFeatureVector):
        return torch.tensor(calculate_term(x.token_embs, centroid.prototype.token_embs))

    def get_distances(self, x: ClusterInstance, instances: List[ClusterInstance]):
        return [
            torch.tensor(calculate_term(x.token_embs, instance.token_embs))
            for instance in instances
        ]

    def build_ActiveClusterFeatureVector(
        self, centroid_id, centroid: ClusterInstance, current_time_step
    ):
        return TokenEmbsActiveClusterFeatureVector(
            centroid_id=centroid_id,
            current_time_step=current_time_step,
            centroid=centroid,
        )

    def build_DeactiveClusterFeatureVector(
        self, centroid_id, n, V1, V2, prototype: ClusterInstance, t
    ):
        return TokenEmbsDeactiveClusterFeatureVector(
            centroid_id, n, V1, V2, prototype, t
        )


class TokenEmbeddingsTermReglSimilartyStrategy(Strategy):
    def __init__(self, encoder) -> None:
        super().__init__(encoder)
        self.search_function = term_regl_search

    def get_embedding(self, x):
        if isinstance(x, ClusterInstance):
            return x.token_embs
        elif isinstance(x, ActiveClusterFeatureVector):
            return x.prototype.token_embs
        else:
            raise TypeError("Unsupported type for 'x'")

    def is_assigned(self, centroid: ActiveClusterFeatureVector, score):
        return centroid.get_rms() <= score

    def is_repr_drifted(
        self, centroid: ActiveClusterFeatureVector, prototype: ClusterInstance
    ):
        return centroid.get_std_norm() > self.get_repr_drift(prototype)

    def get_repr_drift(self, prototype: ClusterInstance):
        E0, E_cur = (
            prototype.token_embs,
            self.encoder.encode(prototype.passage)[1],
        )
        score = calculate_term_regl(E0, E_cur)
        return score

    def get_closest_cluster_indice(
        self,
        query: ClusterInstance,
        k,
        data: List[ActiveClusterFeatureVector],
        vector_dim=768,
    ):
        query = query.token_embs
        embeddings = [c.prototype.token_embs for c in data]
        return term_regl_search(
            query=query, k=k, documents=embeddings, vector_dim=vector_dim
        )

    def get_closet_instance_indice(
        self,
        query: ActiveClusterFeatureVector,
        k,
        data: List[ClusterInstance],
        vector_dim=768,
    ):
        query = query.prototype.token_embs
        embeddings = [c.token_embs for c in data]
        return term_regl_search(
            query=query, k=k, documents=embeddings, vector_dim=vector_dim
        )

    def get_distance(self, x: ClusterInstance, centroid: ActiveClusterFeatureVector):
        return torch.tensor(
            calculate_term_regl(x.token_embs, centroid.prototype.token_embs)
        )

    def get_distances(self, x: ClusterInstance, instances: List[ClusterInstance]):
        return [
            torch.tensor(calculate_term_regl(x.token_embs, instance.token_embs))
            for instance in instances
        ]

    def build_ActiveClusterFeatureVector(
        self, centroid_id, centroid: ClusterInstance, current_time_step
    ):
        return TokenEmbsActiveClusterFeatureVector(
            centroid_id=centroid_id,
            current_time_step=current_time_step,
            centroid=centroid,
        )

    def build_DeactiveClusterFeatureVector(
        self, centroid_id, n, V1, V2, prototype: ClusterInstance, t
    ):
        return TokenEmbsDeactiveClusterFeatureVector(
            centroid_id, n, V1, V2, prototype, t
        )
