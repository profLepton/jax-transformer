from dataclasses import dataclass


@dataclass
class Config():
    num_layers = 3
    vocab_size = 50_000
    hidden_size = 128
    context_length = 128
    num_heads = 4
    causal = True
    batch_size = 32
