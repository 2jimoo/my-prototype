import torch

from transformers import BertTokenizer, BertModel
import torch.nn as nn

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class DenseEncoder(nn.Module):
    def __init__(self, model="bert-base-uncased"):
        super(DenseEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.model = BertModel.from_pretrained(model).to(device)

    def forward(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # .squeeze(0) (batch_size, hidden_size)에서 배치 사이즈(1)없애기
        mean_embedding = outputs.last_hidden_state.mean(dim=1)
        # outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰만 추출
        return mean_embedding

    def encode(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        # with torch.no_grad(): 멍청아...
        outputs = self.model(**inputs)

        mean_embedding = outputs.last_hidden_state.mean(dim=1)
        token_embedding = outputs.last_hidden_state.squeeze(0)
        return mean_embedding, token_embedding
