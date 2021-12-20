# reference: https://github.com/AshwinRJ/Federated-Learning-PyTorch

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import ToTensor

# Load MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    "dataset", download=True, transform=ToTensor()
)
test_dataset = torchvision.datasets.MNIST(
    "dataset", train=False, download=True, transform=ToTensor()
)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True
)


class SimpleMLP(nn.Module):
    """Simple MLP Model for federated learning."""

    def __init__(self, n_input=28 * 28, n_output=10):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(n_input, 512)
        self.linear2 = nn.Linear(512, n_output)
        self.optimizer = optim.Adam(self.parameters(), lr=0.002)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x = self.flatten(x)
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x))
        return x


def train_local_models(models, data_loader, n_train=15):
    """Let train local models from small data (user data)."""
    losses = deque(maxlen=len(models))
    for i, (image, label) in enumerate(data_loader):
        idx = i % len(models)
        model = models[idx]
        model.optimizer.zero_grad()
        output = model(image)
        loss = F.cross_entropy(output, label)
        loss.backward()
        model.optimizer.step()
        losses.append(loss.item())
        if idx == 9:
            print(f"Model Average Loss: {sum(losses) / len(losses):.2f}",)

        if i > len(models) * n_train:
            break


def get_avg_weight(models):
    """Get average weight from local models."""
    avg_weight = copy.deepcopy(models[0].state_dict())
    for k in avg_weight:
        for model in models:
            avg_weight[k] += model.state_dict()[k] / len(models)

    return avg_weight


if __name__ == "__main__":

    # Let there is a global model.
    global_model = SimpleMLP()

    # Let there are 10 clients.
    num_model = 10
    models = []
    for i in range(num_model):
        models.append(copy.deepcopy(global_model))

    # Train local model from each client, each data.
    train_local_models(models, train_data_loader)

    # Get average weight from local models. and Update global model.
    global_model.load_state_dict(get_avg_weight(models))

    # Test global and Local model.
    local_correct = [0] * num_model
    global_correct = 0
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    for img, label in test_data_loader:
        global_correct += (global_model(img).argmax(1) == label).numpy().sum()
        for i in range(num_model):
            local_correct[i] = (
                local_correct[i] + (models[i](img).argmax(1) == label).numpy().sum()
            )

    global_acc = global_correct / len(test_dataset)
    local_acc = [local_correct[i] / len(test_dataset) for i in range(num_model)]
    print(f"Global Accuracy: {global_acc}, Locals Accuracy: {local_acc}")
