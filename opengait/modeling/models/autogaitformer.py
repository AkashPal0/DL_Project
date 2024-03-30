from ..base_model import BaseModel
import torch
import numpy as np
from einops import rearrange
from ..modules import PatchEmbedding, Attention, PositionalEncodings
from torch import nn

    
class AutoGaitFormer(BaseModel):
    def __init__(self, cfgs, training):
        super(AutoGaitFormer, self).__init__(cfgs, training)
        

    def build_network(self, model_cfg):

        #Parameters
        self.block_width = 64
        self.block_height = model_cfg['block_height']
        self.patch_size = model_cfg['patch_size']
        self.dim = np.array((self.patch_size[0]**2,
                             np.array(self.patch_size[1])**2,
                             np.array(self.patch_size[2])**2),
                             dtype=object)
        self.emb_dim = 16
        self.dropout=0.
        self.heads=8
        self.height = self.block_height
        self.width = self.block_width
        self.patch_size = self.patch_size

        # Patching
        self.patch_embedding_L1 = PatchEmbedding(self.dim[0], self.patch_size[0])
        self.pos_encoding_L1 = PositionalEncodings(self.dim[0])
        self.attention_L1 = Attention(self.dim[0], self.heads)

        # self.patch_embedding_L2 = PatchEmbedding(self.dim[1], self.patch_size[0])
    
        # # Learnable params
        # num_patches = (self.block_height // self.patch_size) * (self.block_width // self.patch_size)
        # self.pos_embedding = nn.Parameter(torch.randn(1, 1, 1, num_patches, self.emb_dim))

        # # Self Attention
        # self.attention = Attention(dim = self.emb_dim, n_heads = self.heads, dropout = self.dropout)


    def forward(self, x):
        ipts, labs, _, _, seqL = x
        ipts = ipts[0] ## ipts = [6, 32, 64, 64]

        '''Block 1'''
        x = rearrange(ipts, 'b t (s h) w -> b t s h w', h=self.block_height[0]) ## block_height= 4, x = [6, 32, 16, 4, 64]
        x = self.patch_embedding_L1(x)
        b, t, s, n, _ = x.shape  # s is number of strips
        time = torch.linspace(start = 0, end = s*n -1, steps = s*n).to('cuda')
        y = self.pos_encoding_L1(time) ##
        y = y.unsqueeze(0).unsqueeze(0)
        y = rearrange(y, 'b t (n s) d -> b t n s d', s = s)
        x = x + y ## [6, 32, 16, 16, 16]
        del y
        x = self.attention_L1(x) ## [b t s n d]
        x = rearrange(x, 'b t s (h w) (p1 p2) -> b t s h (w p1 p2)', h = self.patch_size[0], p1 = self.patch_size[0])
        x = rearrange(x, 'b t s h w -> b t (s h) w')
        y = x

        '''Block 2'''
        