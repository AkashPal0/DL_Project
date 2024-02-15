import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, t, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b t n (h d) -> b t h n d', h = h), qkv)
        #print("Shape of q: ", q.shape)
        #print("Shape of k: ", k.shape)
        dots = einsum('b t h i d, b t h j d -> b t h i j', q, k) * self.scale
        #print("Shape of dots: ", dots.shape)
        attn = dots.softmax(dim=-1)
        out = einsum('b t h i j, b t h j d -> b t h i d', attn, v)
        out = rearrange(out, 'b t h n d -> b t n (h d)')
        out =  self.to_out(out)
        return out, attn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        # self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        self.pre_norm = PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))

        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
        #         PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        #     ]))

    def forward(self, x):
        # for attn, ff in self.layers:
        #     attntn, adj_mat = attn(x)
        #     x = attntn + x
        #     # x = ff(x) + x
        attntn, adj_mat = self.pre_norm(x)
        x = attntn + x
        return self.norm(x), adj_mat

class ViViT(nn.Module):
    def __init__(self, image_size, patch_size, num_frames, tub_depth, depth = 1, heads = 8, dropout = 0.,
                 emb_dropout = 0., scale_dim = 4, ):
        super().__init__()

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2    # Number of tubulets per frame
        num_tub = num_frames//tub_depth   # 6               # Number of tubulets in temporal dimension
        patch_dim = tub_depth * patch_size ** 2 #320          # DImension of one tubulet
        dim = 128       #embedding_size
        dim_head = dim//heads
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (t T) (h p1) (w p2) -> b t (h w) (p1 p2 T)', p1 = patch_size, p2 = patch_size, T = tub_depth),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tub, num_patches + 1, dim)) #[1, 6, 65, 384]
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, dim*scale_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)


    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', t=t, b=b)
        x = torch.cat((cls_space_tokens, x), dim=2)
        y = self.pos_embedding[:, :, :(n + 1)]
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)


        x, space_adj_mat = self.space_transformer(x)
       
        x = rearrange(x, 'b t n d -> b n t d')
        r = x.shape[1]

        cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b n t d', b=b, n=r)
        x = torch.cat((cls_temporal_tokens, x), dim=2)
        x, temporal_adj_mat = self.temporal_transformer(x)

        x = rearrange(x, 'b n t d -> b t n d')



        out_cls_token = x[:, 0, 0]
        none2 = x[:, 1:, 1:]
        return out_cls_token, x[:, 1:, 1:], space_adj_mat, temporal_adj_mat


frame_size = 64
patch_size = 8
max_frame = 30
tubulet_depth = 5

vivit = ViViT(frame_size, patch_size, max_frame, tubulet_depth)

x = torch.rand(4,30,64,64)

out_cls_token, x, space_adj_mat, temporal_adj_mat = vivit(x)
