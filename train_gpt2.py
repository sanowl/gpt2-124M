import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

class CaserlSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.register_buffer(
            "bias", 
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        print("Input to Self-Attention:", x.size())
        B, T, C = x.size()  # B: Batch size, T: Sequence length, C: Embedding dimension

        # Apply linear transformation and split into Q, K, V
        qkv = self.c_attn(x)
        print("QKV shape:", qkv.size())
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape and transpose for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        print("Q shape:", q.size())
        print("K shape:", k.size())
        print("V shape:", v.size())

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        print("Attention weights shape:", att.size())

        # Apply attention to V
        y = att @ v
        print("Output of attention:", y.size())

        # Concatenate heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Apply the final linear projection
        y = self.c_proj(y)
        print("Output from Self-Attention:", y.size())
        
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh') 
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        print("Input to MLP:", x.size())
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        print("Output from MLP:", x.size())
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CaserlSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        print("Input to Block:", x.size())
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        print("Output from Block:", x.size())
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

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
    
    def forward(self, idx):
        print("Input to GPTk:", idx.size())
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        print("Token and position embeddings combined:", x.size())
        
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        print("Output logits:", logits.size())

        return logits

# Example instantiation of the model
config = GPTConfig()
model = GPTk(config)

# Example input
input_ids = torch.randint(0, config.vocab_size, (1, 10))  # Batch size 1, sequence length 10
output = model(input_ids)
print("Model output:", output.size())
