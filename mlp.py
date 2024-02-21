import random
import torch
import torch.nn.functional as F


# Hyperparameters
n_embed = 10  # The dimensionality of the character embedding vectors
n_hidden = 200  # The number of neurons in the hidden layer
block_size = 3
batch_size = 32
max_iter = 200_000


def build_dataset(words):
    X, y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            target = str_to_int[ch]
            X.append(context)
            y.append(target)
            context = context[1:] + [target]
    X = torch.tensor(X).to(device=device)
    y = torch.tensor(y).to(device=device)

    return X, y


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read the dataset file
    words = open("names.txt", "r").read().splitlines()

    chars = sorted(list(set("".join(words))))
    vocab_size = len(chars) + 1
    str_to_int = {s: i + 1 for i, s in enumerate(chars)}
    str_to_int["."] = 0
    int_to_str = {i: s for s, i in str_to_int.items()}

    # Build the dataset (train, dev, test)
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    X_train, y_train = build_dataset(words[:n1])
    X_dev, y_dev = build_dataset(words[n1:n2])
    X_test, y_test = build_dataset(words[n2:])

    # Look-up table
    lookup_table = torch.randn((vocab_size, n_embed)).to(device=device)

    # Weights and biases
    g = torch.Generator().manual_seed(32)
    w1 = (
        torch.randn((n_embed * block_size, n_hidden), generator=g).to(device=device)
        * 0.1
    )
    b1 = torch.randn(n_hidden, generator=g).to(device=device) * 0.01
    w2 = torch.randn((n_hidden, vocab_size), generator=g).to(device=device) * 0.01
    b2 = torch.randn(vocab_size, generator=g).to(device=device) * 0.01
    parameters = [lookup_table, w1, b1, w2, b2]

    for p in parameters:
        p.requires_grad = True

    for i in range(max_iter + 1):
        # Minibatch
        x = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)

        # Forward pass
        embed = lookup_table[X_train[x]]
        h = torch.tanh(embed.view(-1, 30) @ w1 + b1)
        logits = h @ w2 + b2
        loss = F.cross_entropy(logits, y_train[x])

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # Update parameters
        lr = 0.1 if i < 100_000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        # Track the training process
        if i % 1000 == 0:
            print(f"iter: {i}, loss: {loss}")

    # Evaluate the model
    embed = lookup_table[X_dev]
    h = torch.tanh(embed.view(-1, 30) @ w1 + b1)
    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, y_dev)
    print(f"eval loss: {loss}")

    # Sample from the model
    g = torch.Generator(device=device).manual_seed(42)
    for _ in range(20):
        output = []
        context = [0] * block_size
        while True:
            embed = lookup_table[torch.tensor([context])]
            h = torch.tanh(embed.view(1, -1) @ w1 + b1)
            logits = h @ w2 + b2
            probs = F.softmax(logits, dim=1)
            target = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [target]
            output.append(target)
            if target == 0:
                break

        print("".join(int_to_str[i] for i in output))
