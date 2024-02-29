import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks

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
        inner_dim = dim_head *  heads ## dim_head = 16, heads = 8, inner_dim = 128, dim = 128
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()

    def forward(self, x):
        b, t, n, _, h = *x.shape, self.heads
        qkv0 = self.to_qkv(x)
        qkv = qkv0.chunk(3, dim = -1)
        del qkv0
        q, k, v = map(lambda m: rearrange(m, 'b t n (h d) -> b t h n d', h = h), qkv)
        #print("Shape of q: ", q.shape)
        #print("Shape of k: ", k.shape)
        dots = einsum('b t h i d, b t h j d -> b t h i j', q, k) * self.scale
        #print("Shape of dots: ", dots.shape)
        attn = dots.softmax(dim=-1)
        out = einsum('b t h i j, b t h j d -> b t h i d', attn, v)
        out = rearrange(out, 'b t h n d -> b t n (h d)')
        # out =  self.to_out(out)
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
        self.feed_forward = nn.Sequential(nn.Linear(dim, dim))

        # for _ in range(depth):
        #     self.layers.append(nn.ModuleList([
        #         PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
        #         PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
        #     ]))

    def forward(self, x):
        attntn, adj_mat = self.pre_norm(x)
        x = attntn + x
        x = self.norm(x)
        ff = self.feed_forward(x)
        x = x + ff
        return x, adj_mat

class ViViT(BaseModel):
    def __init__(self, cfgs, training):
        super(ViViT, self).__init__(cfgs, training)
        

        

    def build_network(self, model_cfg):
        self.image_size = 64
        self.patch_size = 8
        self.num_frames = 30
        self.tub_depth = 5
        self.depth = 1
        self.heads = 8
        self.dropout = 0.
        self.emb_dropout = 0.
        self.scale_dim = 4
        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (self.image_size // self.patch_size) ** 2    # Number of tubulets per frame
        num_tub = self.num_frames//self.tub_depth   # 6               # Number of tubulets in temporal dimension
        patch_dim = self.tub_depth * self.patch_size ** 2 #320          # DImension of one tubulet
        dim = 128       #embedding_size
        dim_head = dim//self.heads ## dim_head = 128//8 == 16
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b (t T) (h p1) (w p2) -> b t (h w) (p1 p2 T)', p1 = patch_size, p2 = patch_size, T = self.tub_depth),
        #     nn.Linear(patch_dim, dim),
        # )

        self.to_patch_embedding = nn.Sequential(nn.Conv3d(in_channels=1, out_channels=128, kernel_size=(5,8,8), stride=(5,8,8)),
                                                Rearrange('b c t h w -> b t (h w) c'))

        self.pos_embedding = nn.Parameter(torch.randn(1, num_tub, num_patches, dim)) #[1, 6, 64, 128]
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(dim, self.depth, self.heads, dim_head, dim*self.scale_dim, self.dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, self.depth, self.heads, dim_head, dim*self.scale_dim, self.dropout)

        self.dropout = nn.Dropout(self.emb_dropout)

    def forward(self, x):
        ipts, labs, _, _, seqL = x
        ipts = ipts[0].unsqueeze(1) #ipts == [6, 1, 30, 64, 64]
        x = self.to_patch_embedding(ipts) ## ipts == [6, 1, 30, 64, 44] ==> [6, 6, 64, 128]
        del ipts
        b, t, n, _ = x.shape

        for i in range(3):
            if(i == 0):
                # cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', t=t, b=b) ## [1, 1, 128] -> [6, 6, 1, 128]
                # x = torch.cat((cls_space_tokens, x), dim=2)
                y = self.pos_embedding[:, :, :n] ## [1, 6, 64, 128]
                
                x += self.pos_embedding[:, :, :n] ## [6, 6, 64, 128]
                x = self.dropout(x)


            x, space_adj_mat = self.space_transformer(x) ## x = [6, 6, 64, 128]
        
            x = rearrange(x, 'b t n c -> b n t c') ## [6, 65, 6, 128]
            # r = x.shape[1]

            # if(i == 0):
                # cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b n t d', b=b, n=r) ## [1, 1, 128] -> [6, 65, 1, 128]
                # x = torch.cat((cls_temporal_tokens, x), dim=2) ##[6, 65, 7, 128]
            x, temporal_adj_mat = self.temporal_transformer(x)

            x = rearrange(x, 'b n t c -> b t n c') ## x = [6, 7, 65, 128]



        out_cls_token = x[:, 0, 0]
        none2 = x[:, 1:, 1:]
        return out_cls_token, x[:, 1:, 1:], space_adj_mat, temporal_adj_mat


frame_size = 64
patch_size = 8
max_frame = 30
tubulet_depth = 5

# vivit = ViViT(frame_size, patch_size, max_frame, tubulet_depth)

# # x = torch.rand(4,30,64,64)

# out_cls_token, x, space_adj_mat, temporal_adj_mat = vivit(x)