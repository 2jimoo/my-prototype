import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from model import DenseEncoder, InfoNCELoss, NCLCompatibleInfoNCELoss
from cluster import (
    ClusterManager,
    NCLSampler,
    SamplingResult,
)
from config import MeanPoolingCosineSimilartyStrategy
from my_data import read_doc_dataset

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def train_model(
    encoder: DenseEncoder,
    loss_fn,
    cluster_manager: ClusterManager,
    sampler: NCLSampler,
    optimizer,
    dataloader,
    max_samples,
    init_cluster_num,
    epochs=3,
):
    encoder.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for batch in dataloader:
            id, text = batch
            anchor_mean_embedding, anchor_token_embedding = encoder.encode(text)
            print(f"anchor_mean_embedding shape: {anchor_mean_embedding.shape}")
            print(f"anchor_token_embedding shape: {anchor_token_embedding.shape}")
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


def inference(encoder: DenseEncoder, cluster_manager: ClusterManager):

    dummy_query = {"qid": 4, "query": "are alpha and beta glucose geometric isomers?"}
    id, text = dummy_query["qid"], dummy_query["query"]
    anchor_mean_embedding, anchor_token_embedding = encoder.encode(text)
    cluster_manager.assign(id, text, anchor_mean_embedding, anchor_token_embedding)


if __name__ == "__main__":
    init_cluster_num = 2
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
    train_model(
        encoder,
        loss_fn,
        cluster_manager,
        sampler,
        optimizer,
        dataloader,
        init_cluster_num,
        max_samples,
    )
    inference(encoder, cluster_manager)
