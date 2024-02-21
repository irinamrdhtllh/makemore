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


@torch.no_grad()
def calculate_loss(mode):
    X, y = {
        "train": (X_train, y_train),
        "dev": (X_dev, y_dev),
        "test": (X_test, y_test),
    }[mode]

    embed = C[X]
    embed_concat = embed.view(embed.shape[0], -1)
    h_preact = embed_concat @ w1 + b1
    h_preact = (
        batch_norm_gain * (h_preact - mean_running) / std_running + batch_norm_bias
    )
    h = torch.tanh(h_preact)
    logits = h @ w2 + b2
    loss = F.cross_entropy(logits, y)

    print(f"{mode}, loss: {loss}")


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
    C = torch.randn((vocab_size, n_embed)).to(device=device)

    # Weights and biases
    g = torch.Generator().manual_seed(32)
    w1 = torch.randn((n_embed * block_size, n_hidden), generator=g).to(
        device=device
    ) * ((5 / 3) / ((n_embed * block_size) ** 0.5))
    b1 = torch.randn(n_hidden, generator=g).to(device=device) * 0.01
    w2 = torch.randn((n_hidden, vocab_size), generator=g).to(device=device) * 0.01
    b2 = torch.randn(vocab_size, generator=g).to(device=device) * 0.01

    # Batch normalization parameters
    batch_norm_gain = torch.ones((1, n_hidden)).to(device=device)
    batch_norm_bias = torch.zeros((1, n_hidden)).to(device=device)
    mean_running = torch.zeros((1, n_hidden)).to(device=device)
    std_running = torch.ones((1, n_hidden)).to(device=device)

    parameters = [C, w1, b1, w2, b2, batch_norm_gain, batch_norm_bias]

    for p in parameters:
        p.requires_grad = True

    for i in range(max_iter + 1):
        # Construct minibatch
        ix = torch.randint(0, X_train.shape[0], (batch_size,), generator=g)
        X_batch, y_batch = X_train[ix], y_train[ix]

        # Forward pass
        embed = C[X_batch]  # Embed the characters into vectors
        embed = embed.view(embed.shape[0], -1)  # Concatenante the vectors
        h_preact = embed @ w1 + b1  # Pre-activated hidden layer
        mean = h_preact.mean(0, keepdim=True)
        std = h_preact.std(0, keepdim=True)
        h_preact = batch_norm_gain * (h_preact - mean) / std + batch_norm_bias

        with torch.no_grad():
            mean_running = 0.999 * mean_running + 0.001 * mean
            std_running = 0.999 * std_running + 0.001 * std

        h = torch.tanh(h_preact)  # Hidden layer
        logits = h @ w2 + b2  # Output layer
        loss = F.cross_entropy(logits, y_batch)  # Loss function

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # Update parameters
        lr = 0.1 if i < 100_000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        # Track the training process
        if i % 10000 == 0:
            print(f"iter: {i}, loss: {loss}")

    # Evaluate the model
    calculate_loss(mode="train")
    calculate_loss(mode="dev")

    # Sample from the model
    g = torch.Generator(device=device).manual_seed(42)
    for _ in range(20):
        output = []
        context = [0] * block_size
        while True:
            embed = C[torch.tensor([context])]
            embed = embed.view(embed.shape[0], -1)
            h_preact = embed @ w1 + b1
            h_preact = (
                batch_norm_gain * (h_preact - mean_running) / std_running
                + batch_norm_bias
            )
            h = torch.tanh(h_preact)
            logits = h @ w2 + b2
            probs = F.softmax(logits, dim=1)
            target = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [target]
            output.append(target)
            if target == 0:
                break

        print("".join(int_to_str[i] for i in output))
