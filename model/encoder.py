import torch

from transformers import BertTokenizer, BertModel
import torch.nn as nn


torch.autograd.set_detect_anomaly(True)
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
        outputs = self.model(**inputs)
        # .squeeze(0) (batch_size, hidden_size)에서 배치 사이즈(1)없애기
        hidden_states = outputs.last_hidden_state.clone()
        mean_embedding = hidden_states.mean(dim=1)
        # outputs.last_hidden_state[:, 0, :]  # [CLS] 토큰만 추출
        return mean_embedding

    def encode(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).to(device)
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state.clone()
        mean_embedding = hidden_states.mean(dim=1)
        token_embedding = hidden_states.squeeze(0)
        return mean_embedding, token_embedding
