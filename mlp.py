import random
import torch
import torch.nn.functional as F


random.seed(42)


def build_dataset(words):
    block_size = 3
    X, y = [], []
    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            target = str_to_int[ch]
            X.append(context)
            y.append(target)
            context = context[1:] + [target]
    X = torch.tensor(X)
    y = torch.tensor(y)

    return X, y


if __name__ == "__main__":
    # Read the dataset file
    words = open("names.txt", "r").read().splitlines()

    chars = sorted(list(set("".join(words))))
    str_to_int = {s: i + 1 for i, s in enumerate(chars)}
    str_to_int["."] = 0
    int_to_str = {i: s for s, i in str_to_int.items()}

    # Build the dataset (train, dev, test)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    X_train, y_train = build_dataset(words[:n1])
    X_dev, y_dev = build_dataset(words[n1:n2])
    X_test, y_test = build_dataset(words[n2:])

    # Look-up table
    lookup_table = torch.randn((27, 10))

    # Weights and biases
    generator = torch.Generator().manual_seed(32)
    weights_1 = torch.randn((30, 200), generator=generator)
    weights_2 = torch.randn((200, 27), generator=generator)
    biases_1 = torch.randn(200, generator=generator)
    biases_2 = torch.randn(27, generator=generator)
    parameters = [lookup_table, weights_1, biases_1, weights_2, biases_2]

    for p in parameters:
        p.requires_grad = True

    for i in range(10000):
        # Minibatch
        x = torch.randint(0, X_train.shape[0], (32,))

        # Forward pass
        embed = lookup_table[X_train[x]]
        h = torch.tanh(embed.view(-1, 30) @ weights_1 + biases_1)
        logits = h @ weights_2 + biases_2
        loss = F.cross_entropy(logits, y_train[x])
        print(f"iter: {i + 1}, loss: {loss}")

        # Backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # Update parameters
        for p in parameters:
            p.data += -0.1 * p.grad

    # Evaluate the model
    embed = lookup_table[X_dev]
    h = torch.tanh(embed.view(-1, 30) @ weights_1 + biases_1)
    logits = h @ weights_2 + biases_2
    loss = F.cross_entropy(logits, y_dev)
    print(f"eval loss: {loss}")
