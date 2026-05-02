"""
Train a small MLP on FashionMNIST and export it to ONNX.

The architecture is intentionally tiny so that Marabou (an SMT-based
verifier) can finish queries in reasonable time.  We only use Gemm
(linear) and ReLU operations -- both well-supported by Marabou's
ONNX parser (see Marabou/resources/onnx/layer-zoo).

Outputs:
  models/fashion_mlp.pth   - PyTorch weights (for re-export / debugging)
  models/fashion_mlp.onnx  - ONNX graph (input to Marabou)
"""

import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# Pixels are scaled to [0, 1] (no mean/std normalisation).
# This makes the verification region "+/- eps in pixel space" easy to
# reason about and lets us safely clip to [0, 1] when building the
# L-infinity ball.
TRANSFORM = transforms.Compose([transforms.ToTensor()])

INPUT_DIM = 28 * 28          # 784 raw pixels
HIDDEN1 = 32
HIDDEN2 = 16
NUM_CLASSES = 10


class FashionMLP(nn.Module):
    """A 3-layer fully-connected ReLU network (784 -> 32 -> 16 -> 10)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN1)
        self.fc2 = nn.Linear(HIDDEN1, HIDDEN2)
        self.fc3 = nn.Linear(HIDDEN2, NUM_CLASSES)

    def forward(self, x):
        # x: (batch, 1, 28, 28) -> (batch, 784)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)       # raw logits; no softmax (Marabou-friendly)


def train(model, loader, device, epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        print(f"  epoch {epoch}: loss={loss_sum/total:.4f} acc={correct/total:.4f}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += x.size(0)
    return correct / total


def export_onnx(model, onnx_path):
    """Export to ONNX with opset 12; produces only Gemm + Relu nodes."""
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28)        # batch=1 for static shapes
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        opset_version=12,
        do_constant_folding=True,
        # No dynamic_axes: Marabou prefers a fully static input shape.
    )
    print(f"  exported -> {onnx_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./models")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_set = datasets.FashionMNIST(
        args.data_dir, train=True, download=True, transform=TRANSFORM)
    test_set = datasets.FashionMNIST(
        args.data_dir, train=False, download=True, transform=TRANSFORM)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = FashionMLP().to(device)
    print("training...")
    train(model, train_loader, device, args.epochs, args.lr)

    acc = evaluate(model, test_loader, device)
    print(f"test accuracy: {acc:.4f}")

    pth = os.path.join(args.out_dir, "fashion_mlp.pth")
    onnx_path = os.path.join(args.out_dir, "fashion_mlp.onnx")
    torch.save(model.state_dict(), pth)
    print(f"  saved -> {pth}")

    # Always export ONNX from CPU model for portability.
    export_onnx(model.cpu(), onnx_path)


if __name__ == "__main__":
    main()
