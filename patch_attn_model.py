# patch_attn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from attention_pooling import AttentionPooling

class ResNet34Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        b = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        b.fc = nn.Identity()
        self.backbone = b

    def forward(self, x):
        return self.backbone(x)  # (B, 512)

class PatchAttentionCLModel(nn.Module):
    """
    Contrastive-learning over patches with hybrid attribute strategy:
      1) attribute-conditioned attention pooling
      2) early-fusion concat after pooling
    Using ResNet-34 backbone.
    """
    def __init__(self, embed_dim=128, attr_dim=0):
        super().__init__()
        self.embed_dim = embed_dim
        # patch→512→embed_dim
        self.encoder = ResNet34Encoder()
        self.projector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim)
        )
        # attribute-conditioned pooling
        self.attn_pool = AttentionPooling(embed_dim, hidden_dim=128, attr_dim=attr_dim)
        # early-fusion attribute MLP
        if attr_dim > 0:
            self.attr_mlp = nn.Sequential(
                nn.Linear(attr_dim, 32),
                nn.ReLU(),
                nn.Linear(32, embed_dim)
            )
            self.final_dim = 2 * embed_dim
        else:
            self.attr_mlp = None
            self.final_dim = embed_dim
        self.fusion = nn.Identity()

    def encode_patches(self, patches):
        # patches: (B*N,3,224,224)
        f = self.encoder(patches)           # (B*N,512)
        return self.projector(f)            # (B*N,embed_dim)

    def forward(self, patches, batch_size, num_patches, attrs=None):
        # 1) encode & reshape
        B, N, C, H, W = patches.shape
        flat = patches.view(B * N, C, H, W)
        proj = self.encode_patches(flat)    # (B*N, embed_dim)
        proj = proj.view(B, N, -1)          # (B, N, embed_dim)
        # 2) attribute-conditioned pooling
        pooled = self.attn_pool(proj, attrs)  # (B, embed_dim)
        # 3) early-fusion
        if self.attr_mlp is not None and attrs is not None:
            a = self.attr_mlp(attrs)           # (B, embed_dim)
            fused = torch.cat([pooled, a], dim=1)  # (B, 2*embed_dim)
            return self.fusion(fused)          # (B, 2*embed_dim)
        return pooled                          # (B, embed_dim)

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N  = z1.size(0)
        z  = torch.cat([z1, z2], dim=0)           # (2N, D)
        sim = torch.mm(z, z.T) / self.temperature # (2N,2N)
        exp = torch.exp(sim)
        mask = ~torch.eye(2*N, device=z.device, dtype=torch.bool)
        exp = exp * mask
        pos = torch.exp((z1 * z2).sum(dim=1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)       # (2N,)
        denom = exp.sum(dim=1)                   # (2N,)
        loss  = -torch.log(pos / denom)
        return loss.mean()