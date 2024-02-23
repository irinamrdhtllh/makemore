import torch
import torch.nn.functional as F


class Linear:
    def __init__(self, fan_in, fan_out, generator=None, bias=True, device=None):
        self.weight = (
            torch.randn((fan_in, fan_out), generator=generator, device=device)
            / fan_in**0.5
        )
        self.bias = torch.zeros(fan_out, device=device) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        return [self.weight] + ([self.bias] if self.bias is not None else [])


class BatchNorm1D:
    def __init__(self, dim, epsilon=1e-5, momentum=0.1, device=None):
        self.epsilon = epsilon
        self.momentum = momentum
        self.training = True
        # Parameters (trained with back propagation)
        self.gamma = torch.ones(dim, device=device)
        self.beta = torch.zeros(dim, device=device)
        # Buffers (trained with a running momentum update)
        self.running_mean = torch.zeros(dim, device=device)
        self.running_var = torch.ones(dim, device=device)

    def __call__(self, x):
        # Forward pass
        if self.training:
            x_mean = x.mean(0, keepdim=True)  # Batch mean
            x_var = x.var(0, keepdim=True, unbiased=True)  # Batch variance
        else:
            x_mean = self.running_mean
            x_var = self.running_var
        x_hat = (x - x_mean) / torch.sqrt(
            x_var + self.epsilon
        )  # Normalize to unit variance
        self.out = self.gamma * x_hat + self.beta

        # Update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * x_mean
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * x_var

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []


class Embedding:
    def __init__(self, num_embeddings, embedding_dim, generator=None, device=None):
        self.weight = torch.randn(
            (num_embeddings, embedding_dim), generator=generator, device=device
        )

    def __call__(self, x):
        self.out = self.weight[x]
        return self.out

    def parameters(self):
        return [self.weight]


class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out

    def parameters(self):
        return []
