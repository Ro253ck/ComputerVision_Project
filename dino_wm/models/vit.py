# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn
from einops import rearrange, repeat

# helpers
NUM_FRAMES = 1
NUM_PATCHES = 1

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch) 
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i+1) + [zeros] * (nwindow - i-1), dim=1)  # gives a (npatch, nwindow*npatch) row block
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)  # stack rows into (nwindow*npatch, nwindow*npatch), then add batch+heads dims to match the attention matrix (B, heads, T, T)
    return mask

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)  # no projection needed if single head with inner dim already matching input dim

        self.heads = heads
        self.scale = dim_head ** -0.5  # divides the similarity matrix to keep the softmax from saturating

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)  # produces Q, K, V in one go

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()  # only project if needed, otherwise the dim is already right
        self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES).to('cuda')  # built once in init, reused every forward call, kept on GPU

    def forward(self, x):
        (
            B,  # batch size
            T,  # total tokens = num_patches x frames
            C,  # embedding dim
        ) = x.size()

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)  # (B, 588, 1536) -> 3 tensors of (B, 588, 512), chunk returns a tuple
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  # split each of Q/K/V into per-head chunks

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # apply causal mask
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # merge heads back together
        return self.to_out(out) 

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x  # residual connection
            x = ff(x) + x

        return self.norm(x)
    
class ViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * (num_patches), dim))  # nn.Parameter: a learnable tensor updated via backprop
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

    def forward(self, x): # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x) 
        x = self.transformer(x) 
        return x