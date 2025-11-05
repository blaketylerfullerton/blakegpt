""""
GPT model here, (rewrite)
Notalble features:
- rotary embeddings (and no positional)
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 5034
    n_layer: int = 12
    n_head: int = 6 #number of query heads
    n_kv_head: int = 6 # number of key/value heads (MQA (Multi query attention))
    n_embd: int = 768

def norm(x):
    #purely functoinal rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 #multi head attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up into last time into two halves
    y1 = x1 * cos + x2 * sin #rotate paids of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1,y2], 3) # re aseemble
    out = out.to(x.dtype) #ensure input / output d types match


class CasualSelfAttention(nn.module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size() #setting the size of x being passed to these vars

        # Project the input to get queries keys, and values
        
        #setting q to be c_q 
        # Q (Query): what this token wants to know
        # K (Key): what this token has to offer
        # V (Value): the actual information content to share  
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)   
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)
