import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from system import (
    DenseEncoder,
    InfoNCELoss,
    NCLCompatibleInfoNCELoss,
    ClusterManager,
    NCLSampler,
    SamplingResult,
    read_doc_dataset,
    calculate_cosine_similarity,
    calculate_term,
    calculate_term_regl,
)

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


if __name__ == "__main__":
    encoder = DenseEncoder().to(device)
    loss_fn = NCLCompatibleInfoNCELoss().to(device)

    cluster_manager = ClusterManager(
        encoder=encoder, similarity_func=calculate_cosine_similarity
    )
    sampler = NCLSampler(
        cluster_manager=cluster_manager, similarity_func=calculate_cosine_similarity
    )

    optimizer = optim.Adam(encoder.parameters(), lr=1e-5)
    dataset = read_doc_dataset()
    dataloader = DataLoader(dataset, batch_size=4)

    max_samples = 3
    train_model(
        encoder, loss_fn, cluster_manager, sampler, optimizer, dataloader, max_samples
    )
