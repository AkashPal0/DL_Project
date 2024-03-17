import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, einsum
from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks
from utils import visualisation as TSNE_Visual

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

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout = 0.):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, dim),
#             nn.Dropout(dropout)
#         )
#     def forward(self, x):
#         return self.net(x)

class FeedForward(nn.Module):
    def __init__(self, dim, cls_num):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, cls_num)
        )
    def forward(self, x):
        return self.net(x)

class HorizontalPool(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.HP = nn.MaxPool3d(kernel_size = (1, 1, patch_size), stride=(1, 1, patch_size), padding=0, dilation=1, return_indices=False, ceil_mode=False)
    
    def forward(self, x):
        return self.HP(x)

class VerticalPool(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.VP = nn.AvgPool3d(kernel_size = (1, patch_size, 1), stride=(1, patch_size, 1), padding=0, ceil_mode=False)
    
    def forward(self, x):
        return self.VP(x)

class TemporalPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_ = nn.Conv3d(in_channels= 6, out_channels= 1,kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
        self.max_ = nn.MaxPool3d(kernel_size = (6, 1, 1), stride=(6, 1, 1))
        self.avg_ = nn.AvgPool3d(kernel_size = (6, 1, 1), stride=(6, 1, 1))

    def forward(self, x):
        x1 = self.avg_(x)
        x1 = x1.squeeze()
        x2 = self.max_(x)
        x2 = x2.squeeze()
        x = rearrange(x, 'b c t h w -> b t c h w') ## [b 1 c h w]
        x3 = self.conv_(x)
        x3 = x3.squeeze()
        return ((x1+x2)+x3) ## [b c h w]

        

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
        x1 = self.norm(x)
        ff = self.feed_forward(x1)
        del x1
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
        self.cls_num = 74
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
        self.make_query = nn.Parameter(torch.randn(1, 1, dim, dim))
        self.space_transformer = Transformer(dim, self.depth, self.heads, dim_head, dim*self.scale_dim, self.dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, self.depth, self.heads, dim_head, dim*self.scale_dim, self.dropout)
        self.dropout = nn.Dropout(self.emb_dropout)
        self.HP = HorizontalPool(self.patch_size)
        self.VP = VerticalPool(self.patch_size)
        self.FCNN = FeedForward(dim, self.cls_num)
        self.TP = TemporalPool()
        self.FF = nn.Sequential(nn.Linear(128, 128))
        self.ce_input = nn.Sequential(Rearrange('b n c -> b c n'),
                                      nn.Conv2d(in_channels=128, out_channels=74, kernel_size=(1,1), stride=(1,1)),
                                      Rearrange('b c n -> b n c'))
    

    def forward(self, x):
        ipts, labs, _, _, seqL = x
        ipts = ipts[0].unsqueeze(1) #ipts == [6, 1, 30, 64, 64]
        x = self.to_patch_embedding(ipts) ## ipts == [6, 1, 30, 64, 64] ==> [6, 6, 64, 128]
        b, t, n, _ = x.shape

        for i in range(3):
            if(i == 0):
                # cls_space_tokens = repeat(self.space_token, '() n d -> b t n d', t=t, b=b) ## [1, 1, 128] -> [6, 6, 1, 128]
                # x = torch.cat((cls_space_tokens, x), dim=2)
                y = self.pos_embedding[:, :, :n] ## [1, 6, 64, 128]
                
                x += self.pos_embedding[:, :, :n] ## [6, 6, 64, 128]
                x = self.dropout(x)


            x, space_adj_mat = self.space_transformer(x) ## x = [6, 6, 64, 128]
        
            x = rearrange(x, 'b t n c -> b n t c') ## [6, 64, 6, 128]
            # r = x.shape[1]

            # if(i == 0):
                # cls_temporal_tokens = repeat(self.temporal_token, '() t d -> b n t d', b=b, n=r) ## [1, 1, 128] -> [6, 65, 1, 128]
                # x = torch.cat((cls_temporal_tokens, x), dim=2) ##[6, 65, 7, 128]
            x, temporal_adj_mat = self.temporal_transformer(x)

            x = rearrange(x, 'b n t c -> b t n c') ## x = [6, 6, 64, 128]

        x = rearrange(x, 'b t (h w) c -> b c t h w', h = self.patch_size, w = self.patch_size) ## x = [6, 128, 6, 8, 8]

        '''
        Implementing for CE Loss
        
        '''
        x1 = self.HP(x) ## x = [6, 128, 6, 8, 1]
        x1 = self.VP(x1) ## x = [6, 128, 6, 1, 1]
        x1 = x1.squeeze()## x = [6, 128, 6]        
        # TSNE_Visual.visual_summary(x, labs)
        fcnn = rearrange(x1, 'b c t -> b (c t)')
        embed1 = self.FCNN(fcnn)

        '''
        Implementing for CA Loss
        
        ''' 
        CA = self.TP(x)
        CA = rearrange(CA, 'b c h w -> b h w c')
        embed2 = rearrange(self.FF(CA), 'b h w c -> b (h w) c')
        
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed2, 'labels': labs},
                'softmax': {'logits': embed1, 'labels': labs}
                # 'temporal_attn': {'attn': temporal_adj_mat, 'labels': labs},
                # 'spatial_attn': {'attn': space_adj_mat, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(ipts, 'n c t h w -> (n c t) 1 h w') #ipts == [6, 1, 30, 64, 64]
            },
            'inference_feat': {
                'embeddings': x
            }
        }

        return retval


frame_size = 64
patch_size = 8
max_frame = 30
tubulet_depth = 5

# vivit = ViViT(frame_size, patch_size, max_frame, tubulet_depth)

# # x = torch.rand(4,30,64,64)

# out_cls_token, x, space_adj_mat, temporal_adj_mat = vivit(x)