import jax
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class config():
    num_layers = 3 
    vocab_size = 100_000
    hidden_size = 128

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
        self.forward(*args)





class Blocks(nn):
    
    def __init__(self, config):
        super().__init__()



class LanguageModel(nn):

    def __init__(self, config):
        super().__init__()
        self.token_embeddings = self.initialize_matrix((config.vocab_size, config.hidden_size)) 
        self.num_layers = config.num_layers
        self.blocks = [Blocks(config) for _ in range(self.num_layers)]
        

    def forward(self, x):
        # x : tokenized inputs
        print(self.token_embeddings)
            

conf = config()

model = LanguageModel(conf)
model("a")
