import torch
from cluster import ClusterManager, SamplingResult

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class NCLSampler:
    def __init__(self, cluster_manager: ClusterManager, similarity_func):
        self.cluster_manager = cluster_manager
        self.similarity_func = similarity_func

    def get_weak_samples(self, anchor, k) -> SamplingResult:
        positive_centroid_id, negative_centroid_id = (
            self.cluster_manager.find_closest_centroids(x_mean_emb=anchor, k=2)
        )
        current_time = self.cluster_manager.time_step

        positive_centroid = self.cluster_manager.centroid_memory[positive_centroid_id]
        positive_centroid_emb = positive_centroid.get_mean()
        positive_candidates = [
            x.mean_emb
            for x in self.cluster_manager.assignment_table[positive_centroid_id]
        ]
        positive_samples = self.find_bottom_k_positive_samples(
            anchor, k, positive_centroid_emb, positive_candidates
        )
        positive_weight = positive_centroid.get_weight(current_time)
        positive_weights = [positive_weight] * len(positive_samples)

        negative_centroid = self.cluster_manager.centroid_memory[negative_centroid_id]
        negative_centroid_emb = negative_centroid.get_mean()
        negative_candidates = [
            x.mean_emb
            for x in self.cluster_manager.assignment_table[negative_centroid_id]
        ]
        negative_samples = self.find_top_k_negative_samples(
            anchor, k, negative_centroid_emb, negative_candidates
        )
        negative_weight = negative_centroid.get_weight(current_time)
        negative_weights = [negative_weight] * len(negative_samples)

        return SamplingResult(
            positive_embeddings=positive_samples,
            positive_weights=positive_weights,
            negative_embeddings=negative_samples,
            negative_weights=negative_weights,
        )

    def find_top_k_negative_samples(self, anchor, k, centroid, instances):
        anchor = anchor.to(device)
        centroid = centroid.to(device)
        instances = instances.to(device)

        sim_anchor_instances = self.similarity_func(anchor, instances)  # (n,)
        sim_anchor_centroid = self.similarity_func(anchor, centroid)  # (1,)

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
        return instances[top_k_indices]

    def find_bottom_k_positive_samples(self, anchor, k, centroid, instances):
        anchor = anchor.to(device)
        centroid = centroid.to(device)
        instances = instances.to(device)

        sim_anchor_instances = self.similarity_func(anchor, instances)  # (n,)
        sim_anchor_centroid = self.similarity_func(anchor, centroid)  # (1,)

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
        return instances[bottom_k_indices]
