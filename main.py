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

plt.figure(figsize=(27, 27))
plt.imshow(n_bigrams, cmap="Blues")
for i in range(n_bigrams.shape[0]):
    for j in range(n_bigrams.shape[1]):
        ch_to_str = int_to_str[i] + int_to_str[j]
        plt.text(j, i, ch_to_str, ha="center", va="bottom", color="gray")
        plt.text(j, i, n_bigrams[i, j].item(), ha="center", va="top", color="gray")
plt.axis("off")
plt.show()
