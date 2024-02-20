import torch
import torch.nn.functional as F


# Create the dataset
words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
str_to_int = {s: i + 1 for i, s in enumerate(chars)}
str_to_int["."] = 0
int_to_str = {i: s for s, i in str_to_int.items()}

block_size = 3
inputs, targets = [], []
for word in words:
    context = [0] * block_size
    for ch in word + ".":
        y = str_to_int[ch]
        inputs.append(context)
        targets.append(y)
        context = context[1:] + [y]

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)

# Look-up table
lookup_table = torch.randn((27, 2))

# Weights and biases
generator = torch.Generator().manual_seed(32)
weights_1 = torch.randn((6, 100), generator=generator)
weights_2 = torch.randn((100, 27), generator=generator)
biases_1 = torch.randn(100, generator=generator)
biases_2 = torch.randn(27, generator=generator)
parameters = [lookup_table, weights_1, biases_1, weights_2, biases_2]

for p in parameters:
    p.requires_grad = True

for i in range(1000):
    # Minibatch
    x = torch.randint(0, inputs.shape[0], (32,))

    # Forward pass
    embed = lookup_table[inputs[x]]
    h = torch.tanh(embed.view(-1, 6) @ weights_1 + biases_1)
    logits = h @ weights_2 + biases_2
    loss = F.cross_entropy(logits, targets[x])
    print(f"iter: {i + 1}, loss: {loss}")

    # Backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update parameters
    for p in parameters:
        p.data += -0.1 * p.grad
