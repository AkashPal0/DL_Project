from ..base_model import BaseModel
import torch
import torchvision.transforms as transforms
from torch import Tensor
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size = 4):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # break-down the block of size 16 x 64 in 8 x 8 patches and flat them
            Rearrange('b t l (h p1) (w p2) -> b t l (h w) (p1 p2)', p1=patch_size, p2=patch_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x
    
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.q = torch.nn.Linear(dim, dim)
        self.k = torch.nn.Linear(dim, dim)
        self.v = torch.nn.Linear(dim, dim)
        self.multihead = nn.Sequential(Rearrange('b t l p (h d) -> b t l h p d', h=n_heads))
        self.scale = (dim/n_heads) ** -0.5

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        qx = self.multihead(q)
        kx = self.multihead(k)
        vx = self.multihead(v)
        dots = einsum('b t l h p d, b t l h q d -> b t l h p q', qx, kx) * self.scale
        attn = dots.softmax(dim=-1)
        out = einsum('b t l h p q, b t l h q d -> b t l h p d', attn, vx)
        out = rearrange(out, 'b t l h p d -> b t l p (h d)')
        return out
    
class AutoFormer(BaseModel):
    def __init__(self, cfgs, training):
        super(AutoFormer, self).__init__(cfgs, training)

    def build_network(self, model_cfg):
        # Attributes
        block_height=16
        block_width=64
        patch_size=4
        emb_dim = 16
        dropout=0.
        heads=8
        self.height = block_height
        self.width = block_width
        self.patch_size = patch_size

        # Patching
        self.patch_embedding = PatchEmbedding(patch_size=patch_size)
    
        # Learnable params
        num_patches = (block_height // patch_size) * (block_width // patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, 1, num_patches, emb_dim))

        # Self Attention
        self.attention = Attention(dim = emb_dim, n_heads = heads, dropout = dropout)


    def forward(self, x):
        ipts, labs, _, _, seqL = x
        ipts = ipts[0]
        x = rearrange(ipts, 'b t (l h) w -> b t l h w', h=self.height)
        x = self.patch_embedding(x)
        b, t, l, p, _ = x.shape
        x += self.pos_embedding
        x = self.attention(x)
        x = rearrange(x, 'b t l p d -> b t (l p) d')
        print(x.shape)
        return x