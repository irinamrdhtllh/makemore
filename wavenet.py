import random
import torch
import torch.nn.functional as F

from layers import *


# Hyperparameters
n_embed = 10  # The dimensionality of the character embedding vectors
n_hidden = 200  # The number of neurons in the hidden layer
block_size = 8
batch_size = 32
max_iter = 200_000


def build_dataset(words, device):
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


@torch.no_grad()
def calculate_loss(mode, model):
    X, y = {
        "train": (X_train, y_train),
        "dev": (X_dev, y_dev),
        "test": (X_test, y_test),
    }[mode]
    logits = model(X)
    loss = F.cross_entropy(logits, y)

    print(f"{mode}, loss: {loss}")


if __name__ == "__main__":
    d = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=d).manual_seed(32)

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
    X_train, y_train = build_dataset(words[:n1], device=d)
    X_dev, y_dev = build_dataset(words[n1:n2], device=d)
    X_test, y_test = build_dataset(words[n2:], device=d)

    # Neural network layers
    model = Sequential(
        [
            Embedding(vocab_size, n_embed, generator=g, device=d),
            FlattenConsecutive(2),
            Linear(n_embed * 2, n_hidden, generator=g, device=d),
            BatchNorm1D(n_hidden, device=d),
            Tanh(),
            FlattenConsecutive(2),
            Linear(n_hidden * 2, n_hidden, generator=g, device=d),
            BatchNorm1D(n_hidden, device=d),
            Tanh(),
            FlattenConsecutive(2),
            Linear(n_hidden * 2, n_hidden, generator=g, device=d),
            BatchNorm1D(n_hidden, device=d),
            Tanh(),
            Linear(n_hidden, vocab_size, generator=g, device=d),
        ]
    )

    with torch.no_grad():
        # Make last layer less confident
        model.layers[-1].weight *= 0.1
        # Apply gain to all other layers
        for layer in model.layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= 5 / 3

    parameters = model.parameters()
    for p in parameters:
        p.requires_grad = True

    for i in range(max_iter + 1):
        # Construct minibatch
        ix = torch.randint(0, X_train.shape[0], (batch_size,), generator=g, device=d)
        X_batch, y_batch = X_train[ix], y_train[ix]

        # Forward pass
        logits = model(X_batch)
        loss = F.cross_entropy(logits, y_batch)

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # Update the parameters
        lr = 0.1 if i < 100_000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        # Track the training process
        if i % 10000 == 0:
            print(f"iter: {i}, loss: {loss}")

    # Evaluate the model
    for layer in model.layers:
        layer.training = False
    calculate_loss("train", model)
    calculate_loss("dev", model)

    # Sample from the model
    g_sample = torch.Generator(device=d).manual_seed(32 + 10)
    for _ in range(20):
        output = []
        context = [0] * block_size

        while True:
            # Forward pass
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            # Sample from the distribution
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            # Shift the context window
            context = context[1:] + [ix]
            output.append(ix)
            # If we sample '.', break
            if ix == 0:
                break

        print("".join(int_to_str[i] for i in output))
