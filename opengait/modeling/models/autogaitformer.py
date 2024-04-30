from ..base_model import BaseModel
import torch
import numpy as np
from einops import rearrange
from ..modules import PatchEmbedding, Attention, PositionalEncodings, CustomConv2d
from torch import nn
from ..layers.Embed import DataEmbedding_wo_pos
from ..layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from ..layers.Autoformer_EncDec import Encoder, EncoderLayer

    
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
        self.emb_dim = 64 # Emdedding dimension is 64 x 1
        self.dropout=0.
        self.heads= model_cfg['num_heads']
        self.height = 64
        self.width = 64
        self.width = self.block_width
        self.patch_size = self.patch_size
        self.factor = model_cfg['factor']
        self.moving_avg = model_cfg['moving_avg']

        # Patching
        # Block 1
        self.patch_embedding_L1 = PatchEmbedding(self.dim[0], self.patch_size[0])
        self.pos_encoding_L1 = PositionalEncodings(self.dim[0])
        self.attention_L1 = Attention(self.dim[0], self.heads)

        # Block 2
        self.patch_embedding_L2_first = PatchEmbedding(self.dim[1][0], self.patch_size[1][0])
        self.pos_encoding_L2_first = PositionalEncodings(self.dim[1][0])
        self.attention_L2_first = Attention(self.dim[1][0], self.heads)

        self.patch_embedding_L2_rest = PatchEmbedding(self.dim[1][1], self.patch_size[1][1])
        self.pos_encoding_L2_rest = PositionalEncodings(self.dim[1][1])
        self.attention_L2_rest = Attention(self.dim[1][1], self.heads)

        # Block 3
        self.patch_embedding_L3_first = PatchEmbedding(self.dim[2][0], self.patch_size[2][0])
        self.pos_encoding_L3_first = PositionalEncodings(self.dim[2][0])
        self.attention_L3_first = Attention(self.dim[2][0], self.heads)

        self.patch_embedding_L3_sec = PatchEmbedding(self.dim[2][1], self.patch_size[2][1])
        self.pos_encoding_L3_sec = PositionalEncodings(self.dim[2][1])
        self.attention_L3_sec = Attention(self.dim[2][1], self.heads)

        self.patch_embedding_L3_rest = PatchEmbedding(self.dim[2][2], self.patch_size[2][2])
        self.pos_encoding_L3_rest = PositionalEncodings(self.dim[2][2])
        self.attention_L3_rest = Attention(self.dim[2][2], self.heads)

        # 1x1 Conv
        self.custom_conv = CustomConv2d(self.width, option="MaxPool2d")

        # self.patch_embedding_L2 = PatchEmbedding(self.dim[1], self.patch_size[0])
    
        # # Learnable params
        # num_patches = (self.block_height // self.patch_size) * (self.block_width // self.patch_size)
        # self.pos_embedding = nn.Parameter(torch.randn(1, 1, 1, num_patches, self.emb_dim))

        # # Self Attention
        # self.attention = Attention(dim = self.emb_dim, n_heads = self.heads, dropout = self.dropout)

        # AutoFormer Encoder
        self.enc_embedding = DataEmbedding_wo_pos(c_in = self.emb_dim, d_model = self.emb_dim, dropout = self.dropout)

        self.encoder = Encoder([
            EncoderLayer(
                attention = AutoCorrelationLayer(
                    correlation = AutoCorrelation(False, self.factor),
                    d_model = self.emb_dim,
                    n_heads = self.heads
                ),
                d_model = self.emb_dim,
                d_ff = 4 * self.emb_dim,
                moving_avg = self.moving_avg,
                dropout = self.dropout,
                activation = "relu"
            )
            for _ in range(1)
        ])

        self.Conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.ff = nn.Sequential(nn.Linear(in_features=128, out_features=128), nn.ReLU())
        self.ff2 = nn.Sequential(nn.Linear(in_features=128, out_features=74), nn.ReLU())

    def forward(self, x):
        ipts, labs, _, _, seqL = x
        sils = ipts[0].unsqueeze(1)
        n1, _, s1, h1, w1 = sils.shape  # s is number of strips
        
        ipts = ipts[0] ## ipts = [6, 32, 64, 64]

        if s1 < 3:
            repeat = 3 if s1 == 1 else 2
            ipts = ipts.repeat(1, repeat, 1, 1)
            
        '''Block 1'''
        x = rearrange(ipts, 'b t (s h) w -> b t s h w', h=self.block_height[0]) ## block_height= 4, x = [6, 32, 16, 4, 64]
        x = self.patch_embedding_L1(x)
        b, t, s, n, _ = x.shape  # s is number of strips
        time = torch.linspace(start = 0, end = s*n -1, steps = s*n).to('cuda')    #positional embedding
        y = self.pos_encoding_L1(time) ##
        y = y.unsqueeze(0).unsqueeze(0)
        y = rearrange(y, 'b t (n s) d -> b t s n d', s = s)
        x = x + y ## [6, 32, 16, 16, 16]
        del y
        x = self.attention_L1(x) ## [b t s n d]
        x = rearrange(x, 'b t s (h w) (p1 p2) -> b t s (h p1) (w p2)', h = self.block_height[0]//self.patch_size[0], p1 = self.patch_size[0])
        x = rearrange(x, 'b t s h w -> b t (s h) w')
        # y = x

        '''Block 2'''
        x = rearrange(x, 'b t (s h) w -> b t s h w', h=self.block_height[1]) ## block_height= 8, x = [6, 32, 8, 8, 64]
        y1, y2 = torch.split(x, [1, (self.height//self.block_height[1])-1], dim=2)
        y1 = self.patch_embedding_L2_first(y1)
        b, t, s, n, _ = y1.shape
        time = torch.linspace(start = 0, end = s*n -1, steps = s*n).to('cuda')
        z1 = self.pos_encoding_L2_first(time)
        z1 = z1.unsqueeze(0).unsqueeze(0)
        z1 = rearrange(z1, 'b t (n s) d -> b t s n d', s = s)
        y1 = y1 + z1
        del z1
        y1 = self.attention_L2_first(y1)
        y1 = rearrange(y1, 'b t s (h w) (p1 p2) -> b t s (h p1) (w p2)', h = self.block_height[1]//self.patch_size[1][0], p1 = self.patch_size[1][0])

        y2 = self.patch_embedding_L2_rest(y2)
        b, t, s, n, _ = y2.shape
        time = torch.linspace(start = 0, end = s*n -1, steps = s*n).to('cuda')
        z2 = self.pos_encoding_L2_rest(time)
        z2 = z2.unsqueeze(0).unsqueeze(0)
        z2 = rearrange(z2, 'b t (n s) d -> b t s n d', s = s)
        y2 = y2 + z2
        del z2
        y2 = self.attention_L2_rest(y2)
        y2 = rearrange(y2, 'b t s (h w) (p1 p2) -> b t s (h p1) (w p2)', h = self.block_height[1]//self.patch_size[1][1], p1 = self.patch_size[1][1])
        x = torch.cat((y1, y2), dim=2)
        x = rearrange(x, 'b t s h w -> b t (s h) w')

        '''Block 3'''
        x = rearrange(x, 'b t (s h) w -> b t s h w', h=self.block_height[2]) ## block_height= 16, x = [6, 32, 4, 16, 64]
        y1, y2 = torch.split(x, [1, (self.height//self.block_height[2])-1], dim=2)
        y21, y22 = torch.split(y2, [1, (self.height//self.block_height[2])-2], dim=2)
        y1 = self.patch_embedding_L3_first(y1)
        b, t, s, n, _ = y1.shape
        time = torch.linspace(start = 0, end = s*n -1, steps = s*n).to('cuda')
        z1 = self.pos_encoding_L3_first(time)
        z1 = z1.unsqueeze(0).unsqueeze(0)
        z1 = rearrange(z1, 'b t (n s) d -> b t s n d', s = s)
        y1 = y1 + z1
        del z1
        y1 = self.attention_L3_first(y1)
        y1 = rearrange(y1, 'b t s (h w) (p1 p2) -> b t s (h p1) (w p2)', h = self.block_height[2]//self.patch_size[2][0], p1 = self.patch_size[2][0])

        y21 = self.patch_embedding_L3_sec(y21)
        b, t, s, n, _ = y21.shape
        time = torch.linspace(start = 0, end = s*n -1, steps = s*n).to('cuda')
        z21 = self.pos_encoding_L3_sec(time)
        z21 = z21.unsqueeze(0).unsqueeze(0)
        z21 = rearrange(z21, 'b t (n s) d -> b t s n d', s = s)
        y21 = y21 + z21
        del z21
        y21 = self.attention_L3_sec(y21)
        y21 = rearrange(y21, 'b t s (h w) (p1 p2) -> b t s (h p1) (w p2)', h = self.block_height[2]//self.patch_size[2][1], p1 = self.patch_size[2][1])
        
        y22 = self.patch_embedding_L2_rest(y22)
        b, t, s, n, _ = y22.shape
        time = torch.linspace(start = 0, end = s*n -1, steps = s*n).to('cuda')
        z22 = self.pos_encoding_L3_rest(time)
        z22 = z22.unsqueeze(0).unsqueeze(0)
        z22 = rearrange(z22, 'b t (n s) d -> b t s n d', s = s)
        y22 = y22 + z22
        del z22
        y22 = self.attention_L3_rest(y22)
        y22 = rearrange(y22, 'b t s (h w) (p1 p2) -> b t s (h p1) (w p2)', h = self.block_height[2]//self.patch_size[2][2], p1 = self.patch_size[2][2])
        y2 = torch.cat((y21, y22), dim=2)
        del y21, y22
        x = torch.cat((y1, y2), dim=2)
        del y1, y2
        x = rearrange(x, 'b t s h w -> b t (s h) w')

        ''' 1x1 Conv '''
        x = self.custom_conv(x)
        
        ''' AutoFormer '''
        enc_out = self.enc_embedding(x)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        ''' Final Conv '''
        x = enc_out.unsqueeze(1) # [6, 1, 32, 64]
        del enc_out
        x = self.Conv1(x) # [6, 128, 32, 64]
        x = rearrange(x, 'b c t d -> b t d c') # [6, 32, 64, 128]
        x = self.ff(x) # [6, 32, 64, 128]
        x = rearrange(x, 'b t d c -> b c t d') # [6, 128, 32, 64]
        x = torch.max(x, dim=2)[0] # [6, 128, 64]
        
        embed = x
        
        x = rearrange(x, 'b c t -> b t c') # [6, 64, 128]
        x = self.ff2(x) # [6, 64, 74]
        logi = rearrange(x, 'b t c -> b c t') # [6, 74, 64]

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n1*s1, 1, h1, w1)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval  
    


