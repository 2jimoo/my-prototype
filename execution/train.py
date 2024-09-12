import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from system import (
    read_doc_dataset,
    DenseEncoder,
    InfoNCELoss,
    NCLCompatibleInfoNCELoss,
    ClusterManager,
    NCLSampler,
    SamplingResult,
    MeanPoolingCosineSimilartyStrategy,
)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def train_model(
    encoder,
    loss_fn,
    cluster_manager,
    sampler,
    optimizer,
    dataloader,
    max_samples,
    epochs=3,
):
    encoder.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for batch in dataloader:
            id, text = batch
            anchor_mean_embedding, anchor_token_embedding = encoder.encode(text)

            cluster_manager.assign(
                id, text, anchor_mean_embedding, anchor_token_embedding
            )
            sampling_result: SamplingResult = sampler.get_weak_samples(
                anchor=anchor_mean_embedding, k=max_samples
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
    cluster_manager.assign()


if __name__ == "__main__":
    encoder = DenseEncoder().to(device)
    loss_fn = NCLCompatibleInfoNCELoss().to(device)

    strategy = MeanPoolingCosineSimilartyStrategy()
    cluster_manager = ClusterManager(strategy=strategy)
    sampler = NCLSampler(cluster_manager=cluster_manager)

    optimizer = optim.Adam(encoder.parameters(), lr=1e-5)
    dataset = read_doc_dataset()
    dataloader = DataLoader(dataset, batch_size=4)

    max_samples = 3
    train_model(
        encoder, loss_fn, cluster_manager, sampler, optimizer, dataloader, max_samples
    )
    inference(encoder, cluster_manager)
