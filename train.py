import os
from config import Config
import tiktoken
from model import LanguageModel
import jax
import jax.numpy as jnp
#implement one hot in the future


path = "data/"

text = ""

for file in os.listdir(path):
    with open(path + file, 'r') as f:
        text += " " + f.read()

enc = tiktoken.get_encoding("gpt2")

# Need to multi thread this

batches = []

encoded = enc.encode(text[:10000])

for i in range(len(encoded)//Config.batch_size):
    batch = [jnp.empty((1, Config.context_length), dtype=jnp.int16), jnp.empty((1, Config.context_length, Config.vocab_size), dtype=jnp.int16) ]
    for j in range(Config.batch_size):
        x = jnp.reshape(jnp.array(encoded[j*Config.context_length:(j+1)*Config.context_length], dtype=jnp.int16), (1, -1))
        target = jax.nn.one_hot(jnp.reshape(jnp.array(encoded[j*Config.context_length+1:(j+1)*Config.context_length+1], dtype=jnp.int16), (1, -1)), Config.vocab_size)
        jnp.stack([batch[0], x], axis=0)
        jnp.stack([batch[1],target] , axis=0)
    batches.append(batch)

print(f"{len(batches)} batches in dataset")

# Optimizer

model = LanguageModel(Config())


target = batches[0][1]

x = batches[0][0]


out = model(x, target)


print(out[0], " ---- ", out[1])


