from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class CaserlSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        pass

    def forward(self, x):
        # Forward pass for CaserlSelfAttention
        pass

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd,4 *config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh') 
        self.c_proj =  nn.Linear(4* config.n_embd, config.n_enbd)
      

    def forward(self, x):
      x = self.c_fc(x)
      x = self.gelu(x)
      x = self.c_proj(x)
      return x
    

# Define the Block class
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CaserlSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# Define GPTConfig dataclass
@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

# Define the GPTk class
class GPTk(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, x):
      
        pass

# Example instantiation of the model
config = GPTConfig()
model = GPTk(config)
