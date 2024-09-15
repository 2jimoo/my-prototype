import torch
from cluster import (
    ClusterManager,
    SamplingResult,
    ClusterInstance,
    ActiveClusterFeatureVector,
)
from config import Strategy
from typing import List

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class NCLSampler:
    def __init__(self, cluster_manager: ClusterManager):
        self.cluster_manager: ClusterManager = cluster_manager
        self.strategy: Strategy = cluster_manager.strategy

    def get_weak_samples(self, anchor_mean_emb, anchor_token_embs, k) -> SamplingResult:
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
        positive_embeddings = [x.mean_emb for x in positive_sample]
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
        negative_embeddings = [x.mean_emb for x in negative_samples]
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
        k,
        centroid: ActiveClusterFeatureVector,
        instances: List[ClusterInstance],
    ):
        if len(instances) <= k:
            # print(f"return {instances}")
            return instances

        anchor = anchor.to(device)
        centroid = centroid.to(device)
        instances = instances.to(device)

        sim_anchor_instances = self.strategy.get_distance(anchor, instances)  # (n,)
        sim_anchor_centroid = self.strategy.get_distance(anchor, centroid)  # (1,)

        closer_to_anchor_mask = sim_anchor_instances > sim_anchor_centroid
        filtered_similarities = sim_anchor_instances[closer_to_anchor_mask]
        filtered_indices = torch.nonzero(closer_to_anchor_mask).squeeze()

        if filtered_similarities.size(0) > k:
            top_k_similarities, top_k_indices = torch.topk(filtered_similarities, k)
        else:
            top_k_similarities, top_k_indices = filtered_similarities, torch.arange(
                filtered_similarities.size(0)
            )

        top_k_indices = (
            filtered_indices[top_k_indices].cpu().tolist()
        )  # 인덱스는 CPU로 변환 후 반환
        top_k_similarities = (
            top_k_similarities.cpu().tolist()
        )  # 유사도 값도 CPU로 변환 후 반환

        # print(f"NCLSampler._find_top_k_negative_samples result")
        # print(f"top_k_similarities: {top_k_similarities}")
        # print(f"top_k_indices: {top_k_indices}")
        # print(f"instances[top_k_indices]: {instances[top_k_indices]}")
        return instances[top_k_indices]

    def _find_bottom_k_positive_samples(
        self,
        anchor: ClusterInstance,
        k,
        centroid: ActiveClusterFeatureVector,
        instances: List[ClusterInstance],
    ):
        if len(instances) <= k:
            # print(f"return {instances}")
            return instances

        anchor = anchor.to(device)
        centroid = centroid.to(device)
        instances = instances.to(device)

        sim_anchor_instances = self.strategy.get_distance(anchor, instances)  # (n,)
        sim_anchor_centroid = self.strategy.get_distance(anchor, centroid)  # (1,)

        closer_to_anchor_mask = sim_anchor_instances < sim_anchor_centroid
        filtered_similarities = sim_anchor_instances[closer_to_anchor_mask]
        filtered_indices = torch.nonzero(closer_to_anchor_mask).squeeze()

        if filtered_similarities.size(0) > k:
            bottom_k_similarities, bottom_k_indices = torch.topk(
                filtered_similarities, k, largest=False
            )
        else:
            bottom_k_similarities, bottom_k_indices = (
                filtered_similarities,
                torch.arange(filtered_similarities.size(0)),
            )

        bottom_k_indices = filtered_indices[bottom_k_indices].cpu().tolist()
        bottom_k_similarities = bottom_k_similarities.cpu().tolist()

        # print(f"NCLSampler._find_bottom_k_positive_samples result")
        # print(f"bottom_k_similarities: {bottom_k_similarities}")
        # print(f"bottom_k_indices: {bottom_k_indices}")
        # print(f"instances[bottom_k_indices]: {instances[bottom_k_indices]}")
        return instances[bottom_k_indices]
