import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Create the dataset
words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set("".join(words))))
str_to_int = {s: i + 1 for i, s in enumerate(chars)}
str_to_int["."] = 0
int_to_str = {i: s for s, i in str_to_int.items()}

inputs, targets = [], []
for word in words:
    chs = ["."] + list(word) + ["."]
    for ch1, ch2 in zip(chs, chs[1:]):
        x = str_to_int[ch1]
        y = str_to_int[ch2]
        inputs.append(x)
        targets.append(y)

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)
n_inputs = inputs.nelement()

print(f"number of examples: {n_inputs}")

# Initialize the network
generator = torch.Generator().manual_seed(32)
weights = torch.randn((27, 27), generator=generator, requires_grad=True)

# Gradient descent
for iter in range(100):
    # Forward pass
    enc_inputs = F.one_hot(inputs, num_classes=27).float()
    logits = enc_inputs @ weights
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(n_inputs), targets].log().mean()
    print(f"iteration {iter + 1}, loss {loss.item()}")

    # Backward pass
    weights.grad = None
    loss.backward()

    # Update the parameters
    weights.data += -0.1 * weights.grad
