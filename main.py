import torch
import matplotlib.pyplot as plt


words = open("names.txt", "r").read().splitlines()

n_bigrams = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set("".join(words))))
str_to_int = {s: i + 1 for i, s in enumerate(chars)}
str_to_int["."] = 0
int_to_str = {i: s for s, i in str_to_int.items()}

for word in words:
    chs = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        i = str_to_int[ch1]
        j = str_to_int[ch2]
        n_bigrams[i, j] += 1

g = torch.Generator().manual_seed(32)

probs = n_bigrams.float()
probs /= probs.sum(1, keepdim=True)

for i in range(10):
    outputs = []
    ix = 0
    while True:
        prob = probs[ix]
        ix = torch.multinomial(
            prob, num_samples=1, replacement=True, generator=g
        ).item()
        outputs.append(int_to_str[ix])
        if ix == 0:
            break
    print("".join(outputs))
