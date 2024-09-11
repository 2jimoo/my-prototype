import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from system import DenseEncoder, InfoNCELoss, CompatibleInfoNCELoss

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

encoder = DenseEncoder().to("cuda")
loss_fn = InfoNCELoss().to("cuda")
optimizer = optim.Adam(encoder.parameters(), lr=1e-5)


# 학습 루프 예시
def train_model(encoder, loss_fn, optimizer, data_loader, epochs=3):
    encoder.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for batch in data_loader:
            texts, positives, negatives = batch  # (anchor, positive, negative)

            # 모델 출력 (GPU로 이동)
            anchor_enc = encoder(texts).to("cuda")[:, 0, :]  # [CLS] 토큰 사용
            positive_enc = encoder(positives).to("cuda")[:, 0, :]
            negative_enc = encoder(negatives).to("cuda")[:, 0, :]

            # 손실 계산
            loss = loss_fn(anchor_enc, positive_enc, negative_enc)

            # 역전파 및 옵티마이저 스텝
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(data_loader)}")
