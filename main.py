import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DenseEncoder, InfoNCELoss, NCLCompatibleInfoNCELoss
from cluster import (
    ClusterManager,
    NCLSampler,
    SamplingResult,
)
from functions import cosine_search
from config import MeanPoolingCosineSimilartyStrategy
from my_data import read_doc_dataset, read_query_dataset
from .evaluate import evaluate_dataset
from collections import defaultdict

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
    epochs=3,
):
    encoder.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for batch in dataloader:
            id, text = batch
            anchor_mean_embedding, anchor_token_embedding = encoder.encode(text)
            cluster_manager.assign(
                x_id=id,
                x_passage=text,
                mean_embedding=anchor_mean_embedding,
                token_embedding=anchor_token_embedding,
            )

            if init_cluster_num <= len(cluster_manager.centroid_memory.keys()):
                sampling_result: SamplingResult = sampler.get_weak_samples(
                    anchor_mean_emb=anchor_mean_embedding,
                    anchor_token_embs=anchor_token_embedding,
                    k=max_samples,
                )

                loss = loss_fn(
                    anchor=anchor_mean_embedding,
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
    save_model(encoder, model_weights_path)


def generate_rank_file(
    cluster_manager: ClusterManager, rank_file_path, k=3, session_t=0
):
    # 현재 세션의 모든 문서에서 각 query의 top-k pid rank순서대로 저장
    encoder: DenseEncoder = save_model(encoder, model_weights_path).to(device)
    encoder.eval()
    test_queries = read_query_dataset()
    all_docs = cluster_manager.instance_memory.values()
    doc_embs = [doc.mean_emb for doc in all_docs]
    result = defaultdict(list)

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
    init_cluster_num = 10
    encoder = DenseEncoder().to(device)
    loss_fn = NCLCompatibleInfoNCELoss().to(device)

    strategy = MeanPoolingCosineSimilartyStrategy(encoder=encoder)
    cluster_manager = ClusterManager(
        strategy=strategy, init_cluster_num=init_cluster_num
    )
    sampler = NCLSampler(cluster_manager=cluster_manager)

    optimizer = optim.Adam(encoder.parameters(), lr=1e-5)
    dataset = read_doc_dataset()
    dataloader = DataLoader(dataset, batch_size=4)

    max_samples = 3
    model_weights_path = "./data/model/model_weights.pth"
    train_model(
        encoder,
        loss_fn,
        cluster_manager,
        sampler,
        optimizer,
        dataloader,
        init_cluster_num,
        max_samples,
        model_weights_path,
    )
    rank_file_path = "./data/result"
    generate_rank_file(cluster_manager)
    # evaluate_dataset()
