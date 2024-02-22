import torch
import torch.nn.functional as F


def build_dataset(words, device):
    X, y = [], []
    for word in words:
        chs = ["."] + list(word) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            i_ch1 = str_to_int[ch1]
            i_ch2 = str_to_int[ch2]
            X.append(i_ch1)
            y.append(i_ch2)

    X = torch.tensor(X, device=device)
    y = torch.tensor(y, device=device)

    return X, y


if __name__ == "__main__":
    d = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device=d).manual_seed(32)

    # Create the dataset
    words = open("names.txt", "r").read().splitlines()

    chars = sorted(list(set("".join(words))))
    str_to_int = {s: i + 1 for i, s in enumerate(chars)}
    str_to_int["."] = 0
    int_to_str = {i: s for s, i in str_to_int.items()}

    X, y = build_dataset(words, device=d)
    n_input = X.nelement()

    # Initialize the network
    weights = torch.randn((27, 27), generator=g, device=d, requires_grad=True)

    # Gradient descent
    for iter in range(1000):
        # Forward pass
        X = F.one_hot(X, num_classes=27).float()  # Encode the input
        logits = X @ weights
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)
        loss = (
            -probs[torch.arange(n_input), y].log().mean() + 0.01 * (weights**2).mean()
        )
        print(f"iteration {iter + 1}, loss {loss.item()}")

        # Backward pass
        weights.grad = None
        loss.backward()

        # Update the parameters
        weights.data += -1 * weights.grad

    # Sample from the neural network model
    for i in range(5):
        output = []
        ix = 0
        while True:
            enc_x = F.one_hot(torch.tensor([ix]), num_classes=27).float()
            logits = enc_x @ weights
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdim=True)

            ix = torch.multinomial(
                probs, num_samples=1, replacement=True, generator=g
            ).item()
            output.append(int_to_str[ix])
            if ix == 0:
                break

        print("".join(output))
