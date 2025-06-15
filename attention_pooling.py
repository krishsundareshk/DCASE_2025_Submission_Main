# attention_pooling.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    """
    Attention pooling with optional attribute‐bias.  
    patch_embeddings: (B, N, D)  
    attrs:            (B, attr_dim) or None  
    """
    def __init__(self, embed_dim, hidden_dim=128, attr_dim=0):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        if attr_dim > 0:
            self.attr_bias = nn.Linear(attr_dim, 1)
        else:
            self.attr_bias = None

    def forward(self, patch_embeddings, attrs=None):
        """
        Returns pooled: (B, D)
        """
        # 1) raw scores
        scores = self.attn_net(patch_embeddings)        # (B, N, 1)
        # 2) attribute bias
        if self.attr_bias is not None and attrs is not None:
            b = self.attr_bias(attrs)                   # (B, 1)
            scores = scores + b.unsqueeze(1)            # broadcast → (B, N, 1)
        # 3) weights + sum
        weights = F.softmax(scores, dim=1)              # (B, N, 1)
        pooled  = (weights * patch_embeddings).sum(dim=1)  # (B, D)
        return pooled