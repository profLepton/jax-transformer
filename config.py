from dataclasses import dataclass


@dataclass
class Config():
    num_layers = 3
    vocab_size = 1000
    hidden_size = 16
    context_length = 8
    num_heads = 4
    causal = True
    batch_size = 1
