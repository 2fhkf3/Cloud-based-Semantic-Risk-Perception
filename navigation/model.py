"""Strict checkpoint-compatible 4x20 / hidden-512 / 12-action GRU-IQN."""
from pathlib import Path
import hashlib
import torch
from torch import nn
from .numerics import require_finite


class GRUIQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(20, 512)
        self.recurrent = nn.GRU(512, 512, batch_first=True)
        self.cos_embedding = nn.Linear(64, 512)
        self.hidden_layer = nn.Linear(512, 512)
        self.hidden_layer_2 = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, 12)
        self.register_buffer('pis', torch.arange(64).float() * torch.pi, persistent=False)

    def encode(self, history):
        _, hidden = self.recurrent(torch.relu(self.head(history)))
        return hidden[-1]  # Zero hidden state for EACH four-frame sliding window.

    def quantiles(self, features, taus):
        cos = torch.cos(taus.unsqueeze(-1) * self.pis)
        embedding = torch.relu(self.cos_embedding(cos))
        x = (features.unsqueeze(1) * embedding).reshape(-1, 512)
        x = torch.relu(self.hidden_layer(x))
        x = torch.relu(self.hidden_layer_2(x))
        return self.output_layer(x).reshape(features.shape[0], taus.shape[1], 12)

    def forward(self, history, taus):
        return self.quantiles(self.encode(history), taus)


def load_model(checkpoint, device='cpu', expected_sha256=None):
    checkpoint = Path(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if expected_sha256 and digest != expected_sha256:
        raise ValueError('checkpoint SHA256 does not match the configured model identity')
    model = GRUIQN()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError('checkpoint must be a tensor state dictionary')
    for name, tensor in state.items():
        require_finite(tensor, 'checkpoint tensor ' + name)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), digest


class QuantilePredictor:
    def __init__(self, checkpoint, device='cpu', seed=37, expected_sha256=None):
        self.model, self.sha256 = load_model(checkpoint, device, expected_sha256)
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def __call__(self, history, alpha):
        h = torch.tensor([history], dtype=torch.float32, device=self.device)
        taus = torch.cat((torch.rand(1, 64, device=self.device, generator=self.generator) * alpha,
                          torch.arange(1, 9, device=self.device).reshape(1, 8) / 16), dim=1)
        with torch.inference_mode():
            values = self.model(h, taus)[0].cpu().tolist()
        q = [sum(row[a] for row in values[:64]) / 64 for a in range(12)]
        action = max(range(12), key=lambda a: q[a])
        rtv = sum((row[action] - values[-1][action]) ** 2 for row in values[64:]) / 8
        return action, q, rtv
