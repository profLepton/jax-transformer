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
    causal = True


def softmax(x):
    return x

def gelu(x):
    return x


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
        return "<class Neural Net>"

class Linear(nn):

    def __init__(self, input_size, output_size, bias):
        super().__init__()
        self.weights = self.initialize_matrix((input_size, output_size))
        if bias:
            self.bias = self.initialize_matrix((output_size,))
            self.forward = jit(lambda x: jnp.dot(x, self.weights) + self.bias)
        else:
            self.forward = jit(lambda x: jnp.dot(x, self.weights))


class FFN(nn):

    def __init__(self, input_size, output_size, bias):
        super().__init__()
        self.layer1 = Linear(input_size, 4 * output_size, bias)
        self.layer2 = Linear(4* output_size, input_size, bias) 
        
    def forward(self, x): 
        x = gelu(self.layer1(x)) 
        x = gelu(self.layer2(x)) 
        return x 


class LayerNorm(nn): 
    def __init__(self, config): 
        pass 
    def forward(self, x): 
        return x

class MultiHeadSelfAttention(nn):
    
    def __init__(self, config):
        super().__init__()
        self.kqv_proj = Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.num_heads = config.num_heads
        self.causal = config.causal
        self.num_heads = config.num_heads

    def forward(self, x):
        B, T, C = x.shape
        kqv = self.kqv_proj(x)
        kqv = kqv.reshape(B, T, self.num_heads, -1)
        kqv = kqv.transpose(0, 2, 1, 3)
        k, q, v = jnp.split(kqv, 3, axis=-1)


        return x
        
         




class Blocks(nn):

    def __init__(self, config):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(config)
        self.ffn = FFN(config.hidden_size, config.hidden_size, False)
        self.ln1 = LayerNorm(config)
        self.ln2 = LayerNorm(config)

    def forward(self, x):
        
        x = self.ln1(x + self.mhsa(x))
        x = self.ln2(x + self.ffn(x))

        return x


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
