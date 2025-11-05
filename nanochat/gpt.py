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

