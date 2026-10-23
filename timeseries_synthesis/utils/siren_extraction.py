#!/usr/bin/env python3
"""
Train a tiny implicit neural representation (SIREN-style) per time series sample.
Input: timestamp t in [0,1]  -> Output: value y (C channels).
Saves a fixed-length parameter vector for each sample as a low-dimensional representation.

Usage:
  python implicit_repr_per_sample.py \
      --data_root /path/to/data/traffic \
      --epochs 1000 --hidden_dim 32 --num_hidden 3 --lr 1e-3 --omega0 30.0 \
      --save_states 0 --device cuda

Notes:
- Handles arrays shaped (N, T) or (N, C, T).
- Produces: implicit_train_parameters.npy, implicit_val_parameters.npy, implicit_test_parameters.npy
- Also writes a JSON metadata file with architecture details.
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import parameters_to_vector

# -----------------------
# Model: SIREN components
# -----------------------

class Sine(nn.Module):
    def __init__(self, omega0: float = 30.0):
        super().__init__()
        self.omega0 = omega0

    def forward(self, x):
        return torch.sin(self.omega0 * x)

def siren_init_(layer: nn.Linear, omega0: float, is_first: bool):
    with torch.no_grad():
        fan_in = layer.weight.size(1)
        if is_first:
            # First layer: uniform(-1/fan_in, 1/fan_in)
            bound = 1.0 / fan_in
        else:
            # Subsequent layers: uniform(-sqrt(6/fan_in)/omega0, sqrt(6/fan_in)/omega0)
            bound = np.sqrt(6.0 / fan_in) / omega0
        layer.weight.uniform_(-bound, bound)
        layer.bias.zero_()

class SIREN(nn.Module):
    """
    Simple SIREN: [Linear -> Sine] x num_hidden -> Linear(out_dim)
    """
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_hidden: int, omega0: float = 30.0):
        super().__init__()
        layers = []
        # First layer
        first = nn.Linear(in_dim, hidden_dim)
        siren_init_(first, omega0=omega0, is_first=True)
        layers += [first, Sine(omega0=omega0)]

        # Hidden layers
        for _ in range(num_hidden - 1):
            lin = nn.Linear(hidden_dim, hidden_dim)
            siren_init_(lin, omega0=omega0, is_first=False)
            layers += [lin, Sine(omega0=omega0)]

        # Final linear
        final = nn.Linear(hidden_dim, out_dim)
        # Small init for output layer to stabilize early training
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_dim) / omega0
            final.weight.uniform_(-bound, bound)
            final.bias.zero_()

        layers += [final]
        self.net = nn.Sequential(*layers)

    def forward(self, t):  # t: (B, in_dim)
        return self.net(t)

# -----------------------
# Utilities
# -----------------------

def ensure_3d(x: np.ndarray) -> np.ndarray:
    """
    Ensure shape is (N, C, T).
    Accepts (N, T) or (N, C, T).
    """
    if x.ndim == 2:
        N, T = x.shape
        x = x.reshape(N, 1, T)
    elif x.ndim == 3:
        pass
    else:
        raise ValueError(f"Expected (N,T) or (N,C,T); got {x.shape}")
    return x

def train_one_signal(
    y_np: np.ndarray,
    epochs: int = 1000,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    num_hidden: int = 3,
    omega0: float = 30.0,
    weight_decay: float = 0.0,
    device: str = "cuda",
    batch_size: int = None,
    use_lbfgs_final: bool = False,
):
    """
    Train a SIREN for a single (C,T) array.
    Returns trained model and MSE.
    """
    # y_np: (C, T)
    C, T = y_np.shape
    t = np.linspace(0.0, 1.0, T, dtype=np.float32)
    t_torch = torch.from_numpy(t).to(device).unsqueeze(-1)  # (T,1)
    y_torch = torch.from_numpy(y_np.T.astype(np.float32)).to(device)  # (T,C)

    model = SIREN(in_dim=1, out_dim=C, hidden_dim=hidden_dim, num_hidden=num_hidden, omega0=omega0).to(device)
    criterion = nn.MSELoss()

    if batch_size is None or batch_size >= T:
        # Full-batch Adam
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            pred = model(t_torch)  # (T,C)
            loss = criterion(pred, y_torch)
            loss.backward()
            optimizer.step()
    else:
        # Mini-batch Adam
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        idx_all = torch.arange(T, device=device)
        for _ in range(epochs):
            # simple random mini-batch
            batch_idx = idx_all[torch.randint(0, T, (batch_size,), device=device)]
            pred = model(t_torch[batch_idx])
            loss = criterion(pred, y_torch[batch_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    if use_lbfgs_final:
        # Optional final refinement with LBFGS on full data
        def closure():
            optimizer_lbfgs.zero_grad(set_to_none=True)
            pred_full = model(t_torch)
            loss_full = criterion(pred_full, y_torch)
            loss_full.backward()
            return loss_full

        optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=0.5, max_iter=200, tolerance_grad=1e-7, tolerance_change=1e-9)
        model.train()
        optimizer_lbfgs.step(closure)

    # Final MSE
    model.eval()
    with torch.no_grad():
        mse = criterion(model(t_torch), y_torch).item()

    return model, mse

def flatten_params(model: nn.Module) -> np.ndarray:
    with torch.no_grad():
        vec = parameters_to_vector([p.detach() for p in model.parameters()])
    return vec.cpu().numpy()

def process_split(
    split_name: str,
    arr: np.ndarray,
    args,
    out_dir: str,
):
    """
    Trains one model per sample and saves parameter matrix:
      implicit_{split}_parameters.npy  -> shape (N, P)
    Optionally saves state_dict per sample (if save_states=1).
    """
    arr = ensure_3d(arr)   # (N,C,T)
    N, C, T = arr.shape

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[{split_name}] Using device:", device)

    # Build a temp model to compute parameter dimensionality (same arch for all)
    tmp_model = SIREN(in_dim=1, out_dim=C, hidden_dim=args.hidden_dim, num_hidden=args.num_hidden, omega0=args.omega0).to(device)
    P = sum(p.numel() for p in tmp_model.parameters())
    del tmp_model

    params_mat = np.zeros((N, P), dtype=np.float32)
    mses = []

    os.makedirs(os.path.join(out_dir, "states", split_name), exist_ok=True)

    for i in tqdm(range(N), desc=f"Training {split_name}"):
        y_np = arr[i]  # (C,T)

        model, mse = train_one_signal(
            y_np=y_np,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            num_hidden=args.num_hidden,
            omega0=args.omega0,
            weight_decay=args.weight_decay,
            device=str(device),
            batch_size=args.batch_size if args.batch_size > 0 else None,
            use_lbfgs_final=bool(args.lbfgs_final),
        )
        
        import matplotlib.pyplot as plt
        plt.plot(y_np.T)
        t_torch = torch.linspace(0.0, 1.0, T, dtype=torch.float32).unsqueeze(-1).to(device)
        plt.plot(t_torch.detach().cpu().numpy().T[0] * 96, model(t_torch).detach().cpu().numpy().T[0])
        save_path = os.path.join(out_dir, "states", split_name, f"sample_{i:06d}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved plot to: {save_path}")
        from IPython import embed; embed()

        mses.append(mse)
        params_mat[i] = flatten_params(model)

        if args.save_states:
            torch.save(model.state_dict(), os.path.join(out_dir, "states", split_name, f"sample_{i:06d}.pt"))

        # Free GPU memory between samples if needed
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # Save parameter matrix + stats
    np.save(os.path.join(out_dir, f"implicit_{split_name}_parameters.npy"), params_mat)
    np.save(os.path.join(out_dir, f"implicit_{split_name}_mse.npy"), np.array(mses, dtype=np.float32))
    print(f"[{split_name}] Saved parameter matrix to: {os.path.join(out_dir, f'implicit_{split_name}_parameters.npy')}")
    print(f"[{split_name}] Mean MSE: {np.mean(mses):.6f} | Median MSE: {np.median(mses):.6f}")

    return P, float(np.mean(mses))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=8)
    parser.add_argument("--num_hidden", type=int, default=2, help="Number of hidden layers (each: Linear+Sine)")
    parser.add_argument("--omega0", type=float, default=30.0, help="SIREN frequency scaling")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=0, help="0 = full-batch; otherwise minibatch size")
    parser.add_argument("--lbfgs_final", type=int, default=0, help="1 to run a final LBFGS refinement")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save_states", type=int, default=0, help="1 to save per-sample state_dicts")
    parser.add_argument("--out_dir", type=str, default=None, help="Defaults to data_root")
    args = parser.parse_args()

    out_dir = args.out_dir or args.data_root
    os.makedirs(out_dir, exist_ok=True)

    # Load data (must exist like your SARIMA setup)
    train_path = os.path.join(args.data_root, "train_timeseries.npy")
    val_path   = os.path.join(args.data_root, "val_timeseries.npy")
    test_path  = os.path.join(args.data_root, "test_timeseries.npy")

    train_arr = np.load(train_path)
    val_arr   = np.load(val_path)
    test_arr  = np.load(test_path)

    # Process each split
    P_train, mse_train = process_split("train", train_arr, args, out_dir)
    P_val,   mse_val   = process_split("val",   val_arr,   args, out_dir)
    P_test,  mse_test  = process_split("test",  test_arr,  args, out_dir)

    # Sanity: parameter sizes must match across splits
    assert P_train == P_val == P_test, "Parameter dimension mismatch across splits (check channel counts)."

    # Save metadata
    meta = dict(
        arch="SIREN",
        in_dim=1,
        hidden_dim=args.hidden_dim,
        num_hidden=args.num_hidden,
        out_dim=int(ensure_3d(train_arr).shape[1]),
        omega0=args.omega0,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        lbfgs_final=bool(args.lbfgs_final),
        param_dim=P_train,
        mse_mean=dict(train=mse_train, val=mse_val, test=mse_test),
        notes="Each row in implicit_{split}_parameters.npy is the flattened parameter vector for one sample."
    )
    with open(os.path.join(out_dir, "implicit_repr_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Wrote metadata:", os.path.join(out_dir, "implicit_repr_meta.json"))

if __name__ == "__main__":
    main()
