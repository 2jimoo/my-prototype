import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DenseEncoder, InfoNCELoss, NCLCompatibleInfoNCELoss
from cluster import (
    ClusterManager,
    NCLSampler,
    RandomSampler,
    Sampler,
    SamplingResult,
)
from functions import cosine_search
from config import MeanPoolingCosineSimilartyStrategy
from my_data import read_doc_dataset, read_query_dataset
from evaluate import evaluate_dataset
from collections import defaultdict
from utils import print_dict, print_dicts

torch.autograd.set_detect_anomaly(True)
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def save_model(encoder: DenseEncoder, model_weights_path):
    torch.save(encoder.state_dict(), model_weights_path)


def load_model(model_weights_path):
    model = DenseEncoder()
    model.load_state_dict(torch.load(model_weights_path))
    return model


def train_model(
    encoder: DenseEncoder,
    loss_fn,
    cluster_manager: ClusterManager,
    sampler: NCLSampler,
    optimizer,
    dataloader,
    max_samples,
    init_cluster_num,
    model_weights_path,
    epochs=1,
):
    # for name, param in encoder.named_parameters():
    #     print(f"Parameter: {name} - requires_grad: {param.requires_grad}")
    encoder.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for batch in dataloader:
            id, text = int(batch["doc_id"].item()), batch["text"][0]
            anchor_mean_embedding, anchor_token_embedding = encoder.encode(text)
            anchor_mean_embedding, anchor_token_embedding = (
                anchor_mean_embedding.clone(),
                anchor_token_embedding.clone(),
            )
            cluster_manager.assign(
                x_id=id,
                x_passage=text,
                mean_embedding=anchor_mean_embedding,
                token_embedding=anchor_token_embedding,
            )
            # print(f'anchor_mean_embedding.requires_grad: {anchor_mean_embedding.requires_grad}')
            # print(f'anchor_token_embedding.requires_grad: {anchor_token_embedding.requires_grad}')

            if init_cluster_num <= len(cluster_manager.centroid_memory.keys()):
                print(
                    f"init_cluster_num: {init_cluster_num}, # of cluster: {len(cluster_manager.centroid_memory.keys())}"
                )
                # print_dict(cluster_manager.centroid_memory)
                # print_dicts(cluster_manager.assignment_table)
                sampling_result: SamplingResult = sampler.get_samples(
                    anchor_mean_emb=anchor_mean_embedding.clone(),
                    anchor_token_embs=anchor_token_embedding.clone(),
                    k=max_samples,
                )

                loss = loss_fn(
                    anchor=anchor_mean_embedding.clone(),
                    positives=sampling_result.positive_embeddings,
                    negatives=sampling_result.negative_embeddings,
                    positive_weights=sampling_result.positive_weights,
                    negative_weights=sampling_result.negative_weights,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader)}")
        cluster_manager.time_step += 1
    save_model(encoder, model_weights_path)


def generate_rank_file(
    cluster_manager: ClusterManager, rank_file_path, k=3, session_t=0
):
    # 현재 세션의 모든 문서에서 각 query의 top-k pid rank순서대로 저장
    encoder: DenseEncoder = load_model(encoder, model_weights_path)
    encoder = encoder.to(device)
    encoder.eval()

    test_queries = read_query_dataset()
    all_docs = cluster_manager.instance_memory.values()
    doc_embs = [doc.mean_emb for doc in all_docs]
    result = defaultdict(list)

    with torch.no_grad():
        for query in test_queries:
            qid = query["qid"]
            q_emb, _ = encoder.encode(query["text"])
            indices = cosine_search(query=q_emb, k=k, documents=doc_embs)
            doc_ids = [all_docs[idx].id for idx in indices]
            result[qid].extend(doc_ids)

    with open(rank_file_path, "w") as f:
        for key, values in result.items():
            line = f"{key} " + " ".join(map(str, values)) + "\n"
            f.write(line)


if __name__ == "__main__":
    init_cluster_num = 5
    encoder = DenseEncoder().to(device)
    loss_fn = NCLCompatibleInfoNCELoss().to(device)

    strategy = MeanPoolingCosineSimilartyStrategy(encoder=encoder)
    cluster_manager = ClusterManager(
        strategy=strategy, init_cluster_num=init_cluster_num
    )
    sampler = RandomSampler(cluster_manager=cluster_manager)

    optimizer = optim.Adam(encoder.parameters(), lr=1e-5)
    dataset = read_doc_dataset()
    dataloader = DataLoader(dataset, batch_size=1)

    max_samples = 3
    model_weights_path = "./data/model/model_weights.pth"
    train_model(
        encoder=encoder,
        loss_fn=loss_fn,
        cluster_manager=cluster_manager,
        sampler=sampler,
        optimizer=optimizer,
        dataloader=dataloader,
        max_samples=max_samples,
        init_cluster_num=init_cluster_num,
        model_weights_path=model_weights_path,
    )
    rank_file_path = "./data/result"
    generate_rank_file(rank_file_path=rank_file_path, cluster_manager=cluster_manager)
    # evaluate_dataset()
