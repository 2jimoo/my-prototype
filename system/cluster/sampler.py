import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def find_top_k_negative_samples(anchor, centroid, instances, k, similarity_func):
    anchor = anchor.to(device)
    centroid = centroid.to(device)
    instances = instances.to(device)

    sim_anchor_instances = similarity_func(anchor, instances)  # (n,)
    sim_anchor_centroid = similarity_func(anchor, centroid)  # (1,)

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

    return top_k_indices, top_k_similarities


def find_bottom_k_positive_samples(anchor, centroid, instances, k, similarity_func):
    anchor = anchor.to(device)
    centroid = centroid.to(device)
    instances = instances.to(device)

    sim_anchor_instances = similarity_func(anchor, instances)  # (n,)
    sim_anchor_centroid = similarity_func(anchor, centroid)  # (1,)

    closer_to_anchor_mask = sim_anchor_instances < sim_anchor_centroid
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

    return top_k_indices, top_k_similarities
