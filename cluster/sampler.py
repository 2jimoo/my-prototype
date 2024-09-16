import torch
from cluster import (
    ClusterManager,
    SamplingResult,
    ClusterInstance,
    ActiveClusterFeatureVector,
)
from config import Strategy
from typing import List
import random
from utils import print_dict, print_dicts

torch.autograd.set_detect_anomaly(True)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class Sampler:
    def __init__(self, cluster_manager: ClusterManager):
        self.cluster_manager: ClusterManager = cluster_manager
        self.strategy: Strategy = cluster_manager.strategy

    def get_samples(self, anchor_mean_emb, anchor_token_embs, k) -> SamplingResult:
        pass


class RandomSampler(Sampler):
    def __init__(self, cluster_manager: ClusterManager):
        self.cluster_manager: ClusterManager = cluster_manager
        self.strategy: Strategy = cluster_manager.strategy

    def get_samples(self, anchor_mean_emb, anchor_token_embs, k) -> SamplingResult:
        x: ClusterInstance = ClusterInstance()
        x.mean_emb = anchor_mean_emb
        x.token_embs = anchor_token_embs

        positive_centroid_id, negative_centroid_id = (
            self.cluster_manager.find_closest_centroid_ids(x=x, k=2)
        )
        # print(
        #     f"random | positive_centroid_id:{positive_centroid_id}, negative_centroid_id:{negative_centroid_id}"
        # )
        positive_cand_indice = self.cluster_manager.assignment_table[
            positive_centroid_id
        ]
        positive_cand_indice = (
            random.sample(positive_cand_indice, k)
            if k <= len(positive_cand_indice)
            else positive_cand_indice
        )
        positive_embeddings = [
            self.strategy.get_embedding(self.cluster_manager.instance_memory[idx])
            for idx in positive_cand_indice
        ]

        negative_cand_indice = self.cluster_manager.assignment_table[
            negative_centroid_id
        ]
        negative_cand_indice = (
            random.sample(negative_cand_indice, k)
            if k <= len(negative_cand_indice)
            else negative_cand_indice
        )
        negative_embeddings = [
            self.strategy.get_embedding(self.cluster_manager.instance_memory[idx])
            for idx in negative_cand_indice
        ]
        # print(
        #     f"random | positive_cand_indice:{positive_cand_indice}, negative_cand_indice:{negative_cand_indice}"
        # )
        return SamplingResult(
            positive_embeddings=positive_embeddings,
            positive_weights=[1.0] * (len(positive_embeddings)),
            negative_embeddings=negative_embeddings,
            negative_weights=[1.0] * (len(negative_embeddings)),
        )


class NCLSampler(Sampler):
    def __init__(self, cluster_manager: ClusterManager):
        self.cluster_manager: ClusterManager = cluster_manager
        self.strategy: Strategy = cluster_manager.strategy

    def get_samples(self, anchor_mean_emb, anchor_token_embs, k) -> SamplingResult:
        x: ClusterInstance = ClusterInstance()
        x.mean_emb = anchor_mean_emb
        x.token_embs = anchor_token_embs

        positive_centroid_id, negative_centroid_id = (
            self.cluster_manager.find_closest_centroid_ids(x=x, k=2)
        )
        current_time = self.cluster_manager.time_step
        # print(f"positive_centroid_id:{positive_centroid_id}, negative_centroid_id:{negative_centroid_id}")

        positive_centroid: ActiveClusterFeatureVector = (
            self.cluster_manager.centroid_memory[positive_centroid_id]
        )
        positive_cand_indice = self.cluster_manager.assignment_table[
            positive_centroid_id
        ]
        positive_sample = self._find_bottom_k_positive_samples(
            anchor=x,
            k=k,
            centroid=positive_centroid,
            instances=[
                self.cluster_manager.instance_memory[idx]
                for idx in positive_cand_indice
            ],
        )
        positive_embeddings = [self.strategy.get_embedding(x) for x in positive_sample]
        positive_weight = positive_centroid.get_weight(current_time)
        positive_weights = [positive_weight] * len(positive_embeddings)

        negative_centroid: ActiveClusterFeatureVector = (
            self.cluster_manager.centroid_memory[negative_centroid_id]
        )
        negative_cand_indice = self.cluster_manager.assignment_table[
            negative_centroid_id
        ]
        negative_samples = self._find_top_k_negative_samples(
            anchor=x,
            k=k,
            centroid=negative_centroid,
            instances=[
                self.cluster_manager.instance_memory[idx]
                for idx in negative_cand_indice
            ],
        )
        negative_embeddings = [self.strategy.get_embedding(x) for x in negative_samples]
        negative_weight = negative_centroid.get_weight(current_time)
        negative_weights = [negative_weight] * len(negative_samples)

        return SamplingResult(
            positive_embeddings=positive_embeddings,
            positive_weights=positive_weights,
            negative_embeddings=negative_embeddings,
            negative_weights=negative_weights,
        )

    def _find_top_k_negative_samples(
        self,
        anchor: ClusterInstance,
        k: int,
        centroid: ActiveClusterFeatureVector,
        instances: List[ClusterInstance],
    ):
        if len(instances) <= k:
            return instances

        sim_anchor_instances = self.strategy.get_distances(anchor, instances)  # (n,)
        sim_anchor_instances_tensor = torch.tensor(sim_anchor_instances).to(device)
        sim_anchor_centroid = self.strategy.get_distance(anchor, centroid).to(
            device
        )  # (1,)

        closer_to_anchor_mask = sim_anchor_instances_tensor > sim_anchor_centroid
        filtered_similarities = sim_anchor_instances_tensor[closer_to_anchor_mask]
        filtered_indices = torch.nonzero(closer_to_anchor_mask, as_tuple=True)[
            0
        ]  # (n,)

        if filtered_similarities.size(0) > k:
            top_k_similarities, top_k_indices = torch.topk(filtered_similarities, k)
        else:
            top_k_similarities = filtered_similarities
            top_k_indices = torch.arange(filtered_similarities.size(0), device=device)

        top_k_indices = filtered_indices[top_k_indices].cpu().tolist()
        top_k_similarities = top_k_similarities.cpu().tolist()

        return [instances[idx] for idx in top_k_indices]

    def _find_bottom_k_positive_samples(
        self,
        anchor: ClusterInstance,
        k,
        centroid: ActiveClusterFeatureVector,
        instances: List[ClusterInstance],
    ):
        if len(instances) <= k:
            return instances

        sim_anchor_instances = self.strategy.get_distances(anchor, instances)  # (n,)
        sim_anchor_instances_tensor = torch.tensor(sim_anchor_instances).to(device)
        sim_anchor_centroid = self.strategy.get_distance(anchor, centroid).to(
            device
        )  # (1,)

        closer_to_anchor_mask = sim_anchor_instances_tensor < sim_anchor_centroid
        filtered_similarities = sim_anchor_instances_tensor[closer_to_anchor_mask]
        filtered_indices = torch.nonzero(closer_to_anchor_mask, as_tuple=True)[0]

        if filtered_similarities.size(0) > k:
            bottom_k_similarities, bottom_k_indices = torch.topk(
                filtered_similarities, k, largest=False
            )
        else:
            bottom_k_similarities, bottom_k_indices = (
                filtered_similarities,
                torch.arange(filtered_similarities.size(0), device=device),
            )

        bottom_k_indices = filtered_indices[bottom_k_indices].cpu().tolist()
        bottom_k_similarities = bottom_k_similarities.cpu().tolist()

        return [instances[idx] for idx in bottom_k_indices]
