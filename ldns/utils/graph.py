import torch
import numpy as np

__all__ = [
    "fully_connected_edge_index",
    "make_z_sym_gts",
    "edgevec_to_dense_mirror",
]

def fully_connected_edge_index(C: int, self_loops: bool = False, device=None):
    rows = torch.arange(C, device=device).repeat_interleave(C)
    cols = torch.arange(C, device=device).repeat(C)
    if not self_loops:
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]
    return torch.stack([rows, cols], dim=0)  # [2, E]

def make_z_sym_gts(z: torch.Tensor, total_neurons: int) -> torch.Tensor:
    z = z.reshape(-1, 1)
    z = z.reshape(total_neurons, total_neurons - 1)
    for i in range(total_neurons - 1):
        z[i + 1:total_neurons, i] = z[i, i:total_neurons - 1]
    z = z.reshape(-1, 1)
    return z

def edgevec_to_dense_mirror(z_edge: torch.Tensor, edge_index: torch.Tensor, C: int) -> np.ndarray:
    z = z_edge.view(-1)
    W = torch.zeros(C, C, dtype=z.dtype, device=z.device)
    src, dst = edge_index.long()
    W[src, dst] = z
    U = torch.triu(W, 1)
    Wm = U + U.T
    Wm.fill_diagonal_(0.0)
    return Wm.detach().cpu().numpy()
