import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.full((out_features, in_features), -2.0))
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        hard_gates = (gates > 0.1).float()
        gates = hard_gates.detach() + gates - gates.detach()
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores).detach()


class SelfPruningNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def prunable_layers(self):
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m


def sparsity_loss(model):
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        total += gates.mean()
    return total


def get_loaders(batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),
                             (0.2470,0.2435,0.2616))
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, lam, device):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        cls_loss = criterion(outputs, labels)
        spar_loss = sparsity_loss(model)

        loss = cls_loss + lam * spar_loss * 2

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return 100 * correct / total


@torch.no_grad()
def compute_sparsity(model, threshold=0.1):
    total = 0
    pruned = 0

    for layer in model.prunable_layers():
        gates = layer.get_gates()
        total += gates.numel()
        pruned += (gates < threshold).sum().item()

    return 100 * pruned / total


def collect_gates(model):
    gates = []
    for layer in model.prunable_layers():
        gates.append(layer.get_gates().cpu().numpy().ravel())
    return np.concatenate(gates)


def plot_gate_distribution(gates, lam, acc, sparsity):
    plt.figure(figsize=(8,4))
    plt.hist(gates, bins=100)
    plt.axvline(0.1, linestyle='--')
    plt.title(f"Gate Distribution (λ={lam})\nAcc={acc:.2f}% | Spar={sparsity:.2f}%")
    plt.xlabel("Gate Value")
    plt.ylabel("Count")
    plt.savefig(f"gate_dist_lambda_{lam}.png")
    plt.close()


def plot_tradeoff(results):
    lambdas = [str(r[0]) for r in results]
    accs = [r[1] for r in results]
    sparsities = [r[2] for r in results]

    x = np.arange(len(lambdas))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x - width/2, accs, width, label="Accuracy")
    ax.bar(x + width/2, sparsities, width, label="Sparsity")

    ax.set_xticks(x)
    ax.set_xticklabels(lambdas)
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Percentage")
    ax.legend()

    plt.title("Accuracy vs Sparsity Trade-off")
    plt.savefig("tradeoff.png")
    plt.close()


def run_experiment(lam, device):
    print(f"\nTraining with λ = {lam}")

    train_loader, test_loader = get_loaders()
    model = SelfPruningNet().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 31):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, lam, device)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            sparsity = compute_sparsity(model)
            print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}% | Sparsity: {sparsity:.2f}%")

    final_acc = evaluate(model, test_loader, device)
    final_sparsity = compute_sparsity(model)
    gates = collect_gates(model)

    print(f"Final Accuracy: {final_acc:.2f}%")
    print(f"Final Sparsity: {final_sparsity:.2f}%")

    return final_acc, final_sparsity, gates


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    lambda_values = [0.05, 0.2, 1.0]

    results = []

    for lam in lambda_values:
        acc, spar, gates = run_experiment(lam, device)
        results.append((lam, acc, spar))
        plot_gate_distribution(gates, lam, acc, spar)

    plot_tradeoff(results)

    print("\nFinal Results:")
    for lam, acc, spar in results:
        print(f"λ={lam} | Accuracy={acc:.2f}% | Sparsity={spar:.2f}%")


if __name__ == "__main__":
    main()