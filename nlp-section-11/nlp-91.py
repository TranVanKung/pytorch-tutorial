import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

with open('./shakespeare.txt', 'r', encoding='utf8') as f:
    text = f.read()

# print(type(text))
# print(text[:1000])
print(len(text))
all_characters = set(text)
print(all_characters)
print(len(all_characters))

# number -> letter
decoder = dict(enumerate(all_characters))
# print(decoder)

# letter -> number
encoder = {char: ind for ind, char in decoder.items()}
# print(encoder)

encoded_text = np.array([encoder[char] for char in text])
print(encoded_text[:500])
print(decoder[3])


def one_hot_encoder(encoded_text, num_uni_chars):
    # encoded_text -> batch of encoded text
    # num_uni_chars -> len(set(text))
    one_hot = np.zeros((encoded_text.size, num_uni_chars))
    one_hot = one_hot.astype(np.float32)
    one_hot[np.arange(one_hot.shape[0]), encoded_text.flatten()] = 1.0
    one_hot = one_hot.reshape((*encoded_text.shape, num_uni_chars))

    return one_hot


# arr = np.array([1, 2, 0])
# print(one_hot_encoder(arr, 3))

example_text = np.arange(10)
example_text.reshape((5, -1))


def generate_batches(encoded_text, samp_per_batch=10, seq_len=50):
    # X: encoded text of length sed_len
    # [0, 1, 2]
    # [1, 2, 3]
    # Y: encoded test shifted by one
    # [1, 2, 3]
    # [2, 3, 4]
    # how many char per batch
    char_per_batch = samp_per_batch * seq_len
    # how many batches we make, givaen the len of encoded text
    num_batches_avail = int(len(encoded_text)/char_per_batch)

    # cut off the end of the encoded ext, that won't fit evenly into a batch
    encoded_text = encoded_text[:num_batches_avail*char_per_batch]

    for n in range(0, encoded_text.shape[1], seq_len):
        x = encoded_text[:, n:n+seq_len]
        # zeros array to the same shape as X
        y = np.zeros_like(x)

        try:
            y[:, :-1] = x[:, 1]
            y[:, -1] = encoded_text[:, n+seq_len]
        except:
            y[:, :-1] = x[:, 1:]
            y[:, -1] = encoded_text[:, 0]
        yield x, y
