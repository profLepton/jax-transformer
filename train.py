import os
from config import Config
import tiktoken
from model import LanguageModel
import jax
import jax.numpy as jnp

path = "data/"

text = ""

for file in os.listdir(path):
    with open(path + file, 'r') as f:
        text += " " + f.read()

enc = tiktoken.get_encoding("gpt2")

# Need to multi thread this

batches = []

encoded = enc.encode(text)

for i in range(len(encoded)//Config.batch_size):
    batch = [[], []]
    for j in range(Config.batch_size):
        batch[0].append(jnp.array(encoded[i*Config.context_length:(i+1)*Config.context_length]))
        batch[1].append(jnp.array(encoded[i*Config.context_length+1:(i+1)*Config.context_length+1]))
    batches.append(batch)

print(f"{len(batches)} batches in dataset")

# Optimizer

model = LanguageModel(Config())

target = batches[0][1]

x = batches[0][0]


print(model(x, y))


