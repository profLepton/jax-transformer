import jax
from jax import grad
import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass
from functools import partial

@dataclass
class config():
    num_layers = 3 
    vocab_size = 100
    hidden_size = 128
    context_length = 128
    num_heads = 4
    causal = 

class nn:

    def __init__(self):
        self.seed = 42
        self.key = jax.random.key(self.seed)
        self.initializer = jax.nn.initializers.glorot_normal()

    def initialize_matrix(self, shape):
        self.key, self.subkey = jax.random.split(self.key)
        matrix = self.initializer(self.subkey, shape, jnp.float32)
        return matrix
    

    def forward(self):
        raise Exception("Forward method has not been implemented in the class")

    def __call__(self, *args):
        return self.forward(*args)

    def __type__(self):
        return "<class Neural Ne>t"


class MultiHeadSelfAttention(nn):
    
    def __init__(self, config):
        super().__init__()
        self.kqv_proj = Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.num_heads = config.num_heads
        self.causal = config.causal



class Blocks(nn):

    def __init__(self, config):
        super().__init__()
        

    def forward(self, x):
        return x

class Linear(nn):

    def __init__(self, input_size, output_size, bias):
        super().__init__()
        self.weights = self.initialize_matrix((input_size, output_size))
        if bias:
            self.bias = self.initialize_matrix((output_size,))
            self.forward = jit(lambda x: jnp.dot(x, self.weights) + self.bias)
        else:
            self.forward = jit(lambda x: jnp.dot(x, self.weights))


class LanguageModel(nn):

    def __init__(self, config):
        super().__init__()
        self.token_embeddings = self.initialize_matrix((config.vocab_size, config.hidden_size)) 
        self.positional_embeddings = self.initialize_matrix((config.context_length, config.hidden_size)) 
        self.num_layers = config.num_layers
        self.blocks = [Blocks(config) for _ in range(self.num_layers)]
        self.head = Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x):
        # x : tokenized inputs
        B, T = x.shape
    
        # Get token embeddings
        token_embeds = self.token_embeddings[x]
        
        # Add positional embeddings
        token_embeds = token_embeds + self.positional_embeddings[jnp.arange(T)]

        for block in self.blocks:
            token_embeds = block(token_embeds)        

        logits = self.head(token_embeds)

        return logits


conf = config()

model = LanguageModel(conf)
input_tokens = jnp.expand_dims(jnp.array([1, 2, 3]), axis=0)

out = model(input_tokens)

print(out.shape)
