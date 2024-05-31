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

enc = tiktoken.get_encoding("cl100k_base")

# Need to multi thread this

batches = []

encoded = enc.encode(text)

for i in range(len(encoded)//Config.batch_size):
    batch = []
    for j in range(Config.batch_size):
        batch.append(jnp.array(encoded[i*Config.context_length:(i+1)*Config.context_length]))
    batches.append(batch)

print(f"{len(batches)} batches in dataset")

# Optimizer

model = LanguageModel(Config())

print(model(batches[0]))



