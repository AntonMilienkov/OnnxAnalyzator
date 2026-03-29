import copy
import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_dataset(
    n_samples: int,
    d_in: int = 8,
    d_out: int = 4,
    noise_std: float = 0.03,
    device: str = "cpu",
):

    x = torch.randn(n_samples, d_in, device=device)
    eps = noise_std * torch.randn(n_samples, d_out, device=device)
    y_clean = 2.0 * x[:, :d_out] - 1.0
    y = y_clean * (1.0 + eps)
    return x, y


def make_loader(x, y, batch_size=64, shuffle=True):
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# Модель A 
class BranchingModelA(nn.Module):
    def __init__(self, d_in=8, d_pre=12, d_hidden=16, d_out=4):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Linear(d_in, d_pre),
            nn.ReLU(),
        )

        self.branch1 = nn.Sequential(
            nn.Linear(d_pre, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

        self.branch2 = nn.Sequential(
            nn.Linear(d_pre, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_out),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(d_pre, 1),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        h = self.pre(x)
        y1 = self.branch1(h)
        y2 = self.branch2(h)
        g = self.gate_net(h)                 
        y = g * y1 + (1.0 - g) * y2
        y = self.post(y)
        return y, g



# Модель C = A + 1 дополнительный слой

class BranchingModelC(nn.Module):
    def __init__(self, d_in=8, d_pre=12, d_hidden=16, d_out=4):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Linear(d_in, d_pre),
            nn.ReLU(),
        )

        self.branch1 = nn.Sequential(
            nn.Linear(d_pre, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

        self.branch2 = nn.Sequential(
            nn.Linear(d_pre, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_out),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(d_pre, 1),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Linear(d_out, d_out),
        )

        # Дополнительный слой относительно модели A
        self.extra_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        h = self.pre(x)
        y1 = self.branch1(h)
        y2 = self.branch2(h)
        g = self.gate_net(h)
        y = g * y1 + (1.0 - g) * y2
        y = self.post(y)
        y = self.extra_layer(y)
        return y, g


# Модель D - другая топология, но слои похожи
class BranchingModelD(nn.Module):
    def __init__(self, d_in=8, d_pre=12, d_hidden=16, d_out=4):
        super().__init__()

        self.pre1 = nn.Sequential(
            nn.Linear(d_in, d_pre),
            nn.ReLU(),
        )

        self.pre2 = nn.Sequential(
            nn.Linear(d_in, d_pre),
            nn.Tanh(),
        )

        self.branch1 = nn.Sequential(
            nn.Linear(d_pre, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

        self.branch2 = nn.Sequential(
            nn.Linear(d_pre, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_out),
        )

        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_pre, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Linear(d_out, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        h1 = self.pre1(x)
        h2 = self.pre2(x)

        y1 = self.branch1(h1)
        y2 = self.branch2(h2)

        h_cat = torch.cat([h1, h2], dim=-1)
        g = self.gate_net(h_cat)

        y = g * y1 + (1.0 - g) * y2
        y = self.post(y)
        return y, g



@dataclass
class TrainConfig:
    lr: float = 1e-3
    epochs: int = 80
    batch_size: int = 64
    weight_decay: float = 1e-5
    device: str = "cpu"


def evaluate(model, loader, criterion, device="cpu"):
    model.eval()
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred, _ = model(xb)
            loss = criterion(pred, yb)

            bs = xb.shape[0]
            total_loss += loss.item() * bs
            total_count += bs

    return total_loss / max(total_count, 1)


def train_model(model, train_loader, val_loader=None, cfg: TrainConfig = TrainConfig()):
    device = cfg.device
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        total_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            bs = xb.shape[0]
            running_loss += loss.item() * bs
            total_count += bs

        train_loss = running_loss / max(total_count, 1)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, criterion, device=device)
            history["val_loss"].append(val_loss)
        else:
            val_loss = float("nan")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}/{cfg.epochs} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )

    return model, history


def clone_model_state(model: nn.Module):
    return copy.deepcopy(model.state_dict())


def load_model_state(model: nn.Module, state_dict):
    model.load_state_dict(copy.deepcopy(state_dict))
    return model


def main():
    set_seed(42)

    device = "cpu"

    d_in = 8
    d_out = 4

    x_A, y_A = generate_dataset(1500, d_in=d_in, d_out=d_out, noise_std=0.03, device=device)
    x_B_extra, y_B_extra = generate_dataset(200, d_in=d_in, d_out=d_out, noise_std=0.03, device=device)
    x_BB_extra, y_BB_extra = generate_dataset(500, d_in=d_in, d_out=d_out, noise_std=0.03, device=device)

    x_val, y_val = generate_dataset(400, d_in=d_in, d_out=d_out, noise_std=0.03, device=device)

    train_loader_A = make_loader(x_A, y_A, batch_size=64, shuffle=True)
    train_loader_B_extra = make_loader(x_B_extra, y_B_extra, batch_size=64, shuffle=True)
    train_loader_BB_extra = make_loader(x_BB_extra, y_BB_extra, batch_size=64, shuffle=True)
    val_loader = make_loader(x_val, y_val, batch_size=128, shuffle=False)

    # train A
    model_A = BranchingModelA(d_in=d_in, d_pre=12, d_hidden=16, d_out=d_out)
    cfg_A = TrainConfig(
        lr=1e-3,
        epochs=80,
        batch_size=64,
        weight_decay=1e-5,
        device=device,
    )

    print("\n=== Training model A on 1500 samples ===")
    model_A, hist_A = train_model(model_A, train_loader_A, val_loader, cfg_A)
    criterion = nn.MSELoss()
    loss_A = evaluate(model_A, val_loader, criterion, device=device)
    print(f"Final validation loss A: {loss_A:.6f}")

    
    state_A = clone_model_state(model_A)

    # Длообучаем B
    model_B = BranchingModelA(d_in=d_in, d_pre=12, d_hidden=16, d_out=d_out)
    load_model_state(model_B, state_A)

    cfg_B = TrainConfig(
        lr=5e-4,
        epochs=30,
        batch_size=64,
        weight_decay=1e-5,
        device=device,
    )

    print("\n=== Fine-tuning model B from A on +200 samples ===")
    model_B, hist_B = train_model(model_B, train_loader_B_extra, val_loader, cfg_B)
    loss_B = evaluate(model_B, val_loader, criterion, device=device)
    print(f"Final validation loss B: {loss_B:.6f}")

    # Дообучаем BB
    model_BB = BranchingModelA(d_in=d_in, d_pre=12, d_hidden=16, d_out=d_out)
    load_model_state(model_BB, state_A)

    cfg_BB = TrainConfig(
        lr=5e-4,
        epochs=40,
        batch_size=64,
        weight_decay=1e-5,
        device=device,
    )

    print("\n=== Fine-tuning model BB from A on +500 samples ===")
    model_BB, hist_BB = train_model(model_BB, train_loader_BB_extra, val_loader, cfg_BB)
    loss_BB = evaluate(model_BB, val_loader, criterion, device=device)
    print(f"Final validation loss BB: {loss_BB:.6f}")

    # Обучение C
    model_C = BranchingModelC(d_in=d_in, d_pre=12, d_hidden=16, d_out=d_out)
    cfg_C = TrainConfig(
        lr=1e-3,
        epochs=90,
        batch_size=64,
        weight_decay=1e-5,
        device=device,
    )

    print("\n=== Training model C (A + extra layer) on 1500 samples ===")
    model_C, hist_C = train_model(model_C, train_loader_A, val_loader, cfg_C)
    loss_C = evaluate(model_C, val_loader, criterion, device=device)
    print(f"Final validation loss C: {loss_C:.6f}")

    # Обучение D
    model_D = BranchingModelD(d_in=d_in, d_pre=12, d_hidden=16, d_out=d_out)
    cfg_D = TrainConfig(
        lr=1e-3,
        epochs=90,
        batch_size=64,
        weight_decay=1e-5,
        device=device,
    )

    print("\n=== Training model D (different topology) on 1500 samples ===")
    model_D, hist_D = train_model(model_D, train_loader_A, val_loader, cfg_D)
    loss_D = evaluate(model_D, val_loader, criterion, device=device)
    print(f"Final validation loss D: {loss_D:.6f}")


    torch.save(model_A.state_dict(), "./PyTorchModels/model_A.pt")
    torch.save(model_B.state_dict(), "./PyTorchModels/model_B.pt")
    torch.save(model_BB.state_dict(), "./PyTorchModels/model_BB.pt")
    torch.save(model_C.state_dict(), "./PyTorchModels/model_C.pt")
    torch.save(model_D.state_dict(), "./PyTorchModels/model_D.pt")

    print("\nSaved weights:")
    print(" - model_A.pt")
    print(" - model_B.pt")
    print(" - model_BB.pt")
    print(" - model_C.pt")
    print(" - model_D.pt")


    dummy_x = torch.randn(1, d_in, device=device)

    model_A.eval()
    torch.onnx.export(
        model_A,
        dummy_x,
        "./OnnxModels/model_A.onnx",
        input_names=["x"],
        output_names=["y", "g"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}, "g": {0: "batch"}},
        opset_version=17,
    )

    model_B.eval()
    torch.onnx.export(
        model_B,
        dummy_x,
        "./OnnxModels/model_B.onnx",
        input_names=["x"],
        output_names=["y", "g"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}, "g": {0: "batch"}},
        opset_version=17,
    )

    model_BB.eval()
    torch.onnx.export(
        model_BB,
        dummy_x,
        "./OnnxModels/model_BB.onnx",
        input_names=["x"],
        output_names=["y", "g"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}, "g": {0: "batch"}},
        opset_version=17,
    )

    model_C.eval()
    torch.onnx.export(
        model_C,
        dummy_x,
        "./OnnxModels/model_C.onnx",
        input_names=["x"],
        output_names=["y", "g"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}, "g": {0: "batch"}},
        opset_version=17,
    )

    model_D.eval()
    torch.onnx.export(
        model_D,
        dummy_x,
        "./OnnxModels/model_D.onnx",
        input_names=["x"],
        output_names=["y", "g"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}, "g": {0: "batch"}},
        opset_version=17,
    )

    print("\nExported ONNX:")
    print(" - model_A.onnx")
    print(" - model_B.onnx")
    print(" - model_BB.onnx")
    print(" - model_C.onnx")
    print(" - model_D.onnx")



main()