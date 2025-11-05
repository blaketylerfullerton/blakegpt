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

        #apply KV Cache: insert current k,v into chace, to get the full view
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)

        Tq = q.size()
        Tk = k.size()

        enable_gqa = self.n_head != self.n_kv_head #group query attention (GQA): duplicate key /value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            #During training (no KV cache), attend as usual with casual attention
            #And even if there is KV cache, we can still use the simple version
            y = F.scaled_dot_product_attention(q, k , v, is_casual=False, enable_gqa=enable_gqa)
           elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0: # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


    class MLP(nn.Module):
        """
        This Class is the Multi Layer Perceptron
            - This expands the embedeeing dimensions by 4 Times  (1)
            - Applies the Relu activation function (2)
            - Then projects back to the origional dimension (3)
        """
        """
        The MLP serves as the position-wise feed-forward network in each transformer block. 
        After the attention mechanism processes the input (focusing on relationships between tokens), 
        the MLP provides additional non-linear transformation capacity to each position independently.
        """
        def __init__(self, config):
            super().__init__()
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

        def forward(self, x):
            x = self.c_fc(x) # (1)
            x = F.relu(x).square() # (2)
            x = self.c_proj(x) # (3)
            return x


    class Block(nn.Module):
        """
        Each Block performs two key operations in sequence using residual connections
        """
        def __init__(self, config, layer_idx):
            super().__init__()
            self.attn = CausalSelfAttention(config, layer_idx) #Self Attention, normalizing the input, and applies casual self attention
            self.mlp = MLP(config)

        """
        Feed-Forward MLP. 
        Normalizing output from step 1, applies MLP transformation, and adds result back
        """
        def forward(self, x, cos_sin, kv_cache):
            x = x + self.attn(norm(x), cos_sin, kv_cache)
            x = x + self.mlp(norm(x))
            return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
             """
                Token Embeddings
                    - wte: converts token IDs into embedding vectors
            """
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
            """
                Token Embeddings
                    - wte converts token IDs into embedding vectors
            """
        })
         """
         Lanuage Model head
            - projects final embeddings back into vocabulary digits
        """
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)




    