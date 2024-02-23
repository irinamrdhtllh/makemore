import random
import torch
import torch.nn.functional as F

from layers import Linear, BatchNorm1D, Tanh, Embedding, Flatten


# Hyperparameters
n_embed = 10  # The dimensionality of the character embedding vectors
n_hidden = 200  # The number of neurons in the hidden layer
block_size = 3
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
def calculate_loss(mode, layers):
    X, y = {
        "train": (X_train, y_train),
        "dev": (X_dev, y_dev),
        "test": (X_test, y_test),
    }[mode]

    x = X
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y)

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
    layers = [
        Embedding(vocab_size, n_embed, generator=g, device=d),
        Flatten(),
        Linear(n_embed * block_size, n_hidden, generator=g, device=d),
        BatchNorm1D(n_hidden, device=d),
        Tanh(),
        Linear(n_hidden, n_hidden, generator=g, device=d),
        BatchNorm1D(n_hidden, device=d),
        Tanh(),
        Linear(n_hidden, n_hidden, generator=g, device=d),
        BatchNorm1D(n_hidden, device=d),
        Tanh(),
        Linear(n_hidden, n_hidden, generator=g, device=d),
        BatchNorm1D(n_hidden, device=d),
        Tanh(),
        Linear(n_hidden, n_hidden, generator=g, device=d),
        BatchNorm1D(n_hidden, device=d),
        Tanh(),
        Linear(n_hidden, vocab_size, generator=g, device=d),
        BatchNorm1D(vocab_size, device=d),
    ]

    with torch.no_grad():
        # Make last layer less confident
        # layers[-1].weight *= 0.1
        layers[-1].gamma *= 0.1
        # Apply gain to all other layers
        for layer in layers[:-1]:
            if isinstance(layer, Linear):
                layer.weight *= 5 / 3

    parameters = [p for layer in layers for p in layer.parameters()]
    for p in parameters:
        p.requires_grad = True

    for i in range(max_iter + 1):
        # Construct minibatch
        ix = torch.randint(0, X_train.shape[0], (batch_size,), generator=g, device=d)
        X_batch, y_batch = X_train[ix], y_train[ix]

        # Forward pass
        x = X_batch
        for layer in layers:
            x = layer(x)
        loss = F.cross_entropy(x, y_batch)

        # Backward pass
        for layer in layers:
            layer.out.retain_grad()
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
    for layer in layers:
        layer.training = False
    calculate_loss("train", layers)
    calculate_loss("dev", layers)

    # Sample from the model
    g_sample = torch.Generator(device=d).manual_seed(32 + 10)
    for _ in range(20):
        output = []
        context = [0] * block_size

        while True:
            # Forward pass
            x = torch.tensor([context])
            for layer in layers:
                x = layer(x)
            logits = x
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
