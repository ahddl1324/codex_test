# main_newSTDP_recurrent.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from ldns.utils.dist import init_for_distributed, is_master, cleanup
from ldns.utils.graph import fully_connected_edge_index

from ldns.models.sl_GRU_edge_embedding_2conv_dilated_chunk import GTSStructureLearnerGRU
from ldns.models.sp_concat_mlp_nobn import sp_ver2
from ldns.train.fit_newSTDP_recurrent_fast import fit_gts_spver2_perbatchSL_clean

try:
    import wandb
except Exception:
    wandb = None
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--exp_name", type=str, default="recurrent_SL_test")
    ap.add_argument("--save_root", type=str, default="./newSTDP/exp")
    ap.add_argument("--use_wandb", action="store_true")

    ap.add_argument("--train_list", type=str, nargs="+", required=True)
    ap.add_argument("--val_list", type=str, nargs="+", required=True)
    ap.add_argument("--test_list", type=str, nargs="+", required=True)

    ap.add_argument("--totalsteps", type=int, default=600000)
    ap.add_argument("--skipsteps", type=int, default=0)

    # recurrent input length
    ap.add_argument("--seq_len", type=int, default=400)
    ap.add_argument("--seq_stride", type=int, default=1)

    # SP history
    ap.add_argument("--history", type=int, default=200)
    ap.add_argument("--pred_steps", type=int, default=1)

    ap.add_argument("--batch_size", type=int, default=64)

    # recurrent SL
    ap.add_argument("--conv_filter_size", type=int, default=80)
    ap.add_argument("--local_conv_stride", type=int, default=40)
    ap.add_argument("--conv2_time_kernel", type=int, default=4)
    ap.add_argument("--out_channel", type=int, default=32)
    ap.add_argument("--n_hid", type=int, default=32)
    ap.add_argument("--gru_hidden", type=int, default=64)
    ap.add_argument("--gru_layers", type=int, default=1)
    ap.add_argument("--gru_dropout", type=float, default=0.0)
    ap.add_argument("--conv_stride", type=int, default=1)
    ap.add_argument("--edge_chunk_num", type=int, default=20)

    ap.add_argument("--embed_mode", type=str, default="learnable",
                    choices=["none", "fixed", "learnable"])
    ap.add_argument("--posemb_dim", type=int, default=50)

    # SP
    ap.add_argument("--residual_flag", type=str2bool, default=False)

    # optim
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--minimum_lr", type=float, default=1e-5)
    ap.add_argument("--decay_step", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--max_epoch", type=int, default=200)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--beta_l2", type=float, default=0.0)
    ap.add_argument("--save_every", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--save_edge_snapshot", type=str2bool, default=True)
    ap.add_argument("--alpha_poisson", type=float, default=1.0)
    ap.add_argument("--alpha_mse", type=float, default=0.0)

    # dataloader
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--pin_memory", type=str2bool, default=True)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", type=str2bool, default=True)

    # ddp
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)

    return ap.parse_args()


def _load_spikeTN_as_counts_CT(path: str, totalsteps: int, skipsteps: int) -> torch.Tensor:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        x = np.load(path)

    elif ext in (".pt", ".pth"):
        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, torch.Tensor):
            x = obj
        elif isinstance(obj, (list, tuple)) and len(obj) >= 1:
            x = obj[0]
        elif isinstance(obj, dict):
            x = None
            for k in ["counts", "samples", "signal", "spikes", "spike", "X", "data", "x"]:
                if k in obj and isinstance(obj[k], torch.Tensor):
                    x = obj[k]
                    print(f"[data] using key '{k}' from {path}")
                    break
            if x is None:
                raise KeyError(f"{path}에서 spike tensor를 못 찾음. keys={list(obj.keys())}")
        else:
            raise TypeError(f"지원하지 않는 저장 형식: {type(obj)}")

        if isinstance(x, torch.Tensor) and x.dim() == 3:
            if x.size(0) != 1:
                print(f"[warn] counts shape={tuple(x.shape)}. 첫 배치만 사용합니다.")
            x = x[0]

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"loaded object is not Tensor: {type(x)}")

        x = x.cpu().numpy()

    else:
        raise ValueError(f"Unsupported file extension: {ext}")

    if x.ndim != 2:
        raise ValueError(f"Expected 2D array after load, got shape={x.shape}")

    # 입력이 [T,C]인지 [C,T]인지 자동 판별
    if x.shape[0] >= x.shape[1]:
        spikes_TC = x
    else:
        spikes_TC = x.T

    T, C = spikes_TC.shape

    t0 = skipsteps
    t1 = min(T, skipsteps + totalsteps)

    if t0 >= t1:
        raise ValueError(f"Invalid crop [{t0}, {t1}) for T={T} in {path}")

    spikes_TC = spikes_TC[t0:t1]
    counts_CT = torch.from_numpy(spikes_TC.T).float().contiguous()

    print(f"[data] loaded counts_CT={tuple(counts_CT.shape)} from {path}")
    return counts_CT


class LongRunRecurrentDataset(torch.utils.data.Dataset):
    """
    counts_CT: [C,T]

    return item:
      x: [Tlong,C]
      y: [S,C]
      a: dummy

    where:
      Tlong = seq_len
      S = seq_len - history + 1

    예:
      seq_len=400, history=200
      x = raw[t : t+400]
      y = raw[t+200 : t+401]
      S = 201
    """
    def __init__(self, counts_CT, seq_len, history, stride=1):
        super().__init__()
        self.counts_CT = counts_CT.float().contiguous()
        self.seq_len = int(seq_len)
        self.history = int(history)
        self.stride = int(stride)

        C, T = self.counts_CT.shape
        self.C = C
        self.T = T

        if self.seq_len < self.history:
            raise ValueError(f"seq_len={self.seq_len} must be >= history={self.history}")

        self.S = self.seq_len - self.history + 1

        # y 마지막은 raw[start + seq_len]까지 필요
        # x: start : start+seq_len
        # y: start+history : start+history+S
        #    마지막 index = start+history+S-1 = start+seq_len
        last_start = T - self.seq_len

        if last_start <= 0:
            self.starts = []
        else:
            self.starts = list(range(0, last_start, self.stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]

        x = self.counts_CT[:, s:s + self.seq_len].t().contiguous()
        # [seq_len,C]

        y_start = s + self.history
        y_end = y_start + self.S
        y = self.counts_CT[:, y_start:y_end].t().contiguous()
        # [S,C]

        angle_dummy = torch.tensor(0.0, dtype=torch.float32)

        return {
            "x": x,
            "y": y,
            "a": angle_dummy,
        }


def _collate_recurrent_to_fit(batch):
    """
    batch item:
      x: [Tlong,C]
      y: [S,C]

    return:
      x: [Tlong,B,C]
      y: [S,B,C]
      a: [B]
    """
    xs = torch.stack([b["x"] for b in batch], dim=1).contiguous()
    ys = torch.stack([b["y"] for b in batch], dim=1).contiguous()
    aa = torch.stack([b["a"] for b in batch], dim=0).contiguous()
    return xs, ys, aa


def _make_loader_from_dataset(dataset, batch_size, shuffle, args):
    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        collate_fn=_collate_recurrent_to_fit,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**loader_kwargs)


def _wrap_ddp_loader(dl, args, shuffle: bool):
    sampler = DistributedSampler(
        dl.dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=shuffle,
    )

    loader_kwargs = dict(
        dataset=dl.dataset,
        batch_size=dl.batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=getattr(dl, "drop_last", False),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.persistent_workers and args.num_workers > 0),
        collate_fn=getattr(dl, "collate_fn", None),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    return DataLoader(**loader_kwargs)


def _build_dataset_from_file_list(file_list, args, split_name, expected_C=None):
    datasets = []
    per_file_info = []

    for spike_path in file_list:
        counts_CT = _load_spikeTN_as_counts_CT(
            spike_path,
            totalsteps=args.totalsteps,
            skipsteps=args.skipsteps,
        )

        C_i, T_i = counts_CT.shape

        if expected_C is None:
            expected_C = C_i
        elif C_i != expected_C:
            raise ValueError(
                f"Neuron count mismatch: expected C={expected_C}, got C={C_i} from {spike_path}"
            )

        dataset_i = LongRunRecurrentDataset(
            counts_CT=counts_CT,
            seq_len=args.seq_len,
            history=args.history,
            stride=args.seq_stride,
        )

        datasets.append(dataset_i)

        info_i = {
            "path": spike_path,
            "split": split_name,
            "C": C_i,
            "T": T_i,
            "seq_len": args.seq_len,
            "history": args.history,
            "S": dataset_i.S,
            "n_samples": len(dataset_i),
        }
        per_file_info.append(info_i)

        print(
            f"[{split_name}] {spike_path}\n"
            f"         C={C_i}, T={T_i}, seq_len={args.seq_len}, "
            f"history={args.history}, S={dataset_i.S}, samples={len(dataset_i)}"
        )

    if len(datasets) == 0:
        raise ValueError(f"No valid dataset for split={split_name}")

    return ConcatDataset(datasets), expected_C, per_file_info


def build_multi_longrun_loaders(args):
    expected_C = None

    train_dataset, expected_C, train_info = _build_dataset_from_file_list(
        args.train_list, args, "train", expected_C
    )
    val_dataset, expected_C, val_info = _build_dataset_from_file_list(
        args.val_list, args, "val", expected_C
    )
    test_dataset, expected_C, test_info = _build_dataset_from_file_list(
        args.test_list, args, "test", expected_C
    )

    if is_master(args):
        print(
            f"[concat] train={len(train_dataset)}, "
            f"val={len(val_dataset)}, test={len(test_dataset)}"
        )

    train_loader = _make_loader_from_dataset(
        train_dataset, batch_size=args.batch_size, shuffle=True, args=args
    )
    val_loader = _make_loader_from_dataset(
        val_dataset, batch_size=args.batch_size, shuffle=False, args=args
    )
    test_loader = _make_loader_from_dataset(
        test_dataset, batch_size=args.batch_size, shuffle=False, args=args
    )

    per_file_info = {
        "train": train_info,
        "val": val_info,
        "test": test_info,
    }

    return train_loader, val_loader, test_loader, expected_C, per_file_info


def main():
    args = parse_args()
    set_seed(42)
    init_for_distributed(args)

    if is_master(args):
        print("[data] train_list:")
        for p in args.train_list:
            print("  ", p)
        print("[data] val_list:")
        for p in args.val_list:
            print("  ", p)
        print("[data] test_list:")
        for p in args.test_list:
            print("  ", p)

    if is_master(args) and args.use_wandb and (wandb is not None):
        wandb.init(project="connectivity", name=args.exp_name, config=vars(args))

    train_loader, val_loader, test_loader, C, per_file_info = build_multi_longrun_loaders(args)

    if args.distributed:
        train_loader = _wrap_ddp_loader(train_loader, args, shuffle=True)
        val_loader = _wrap_ddp_loader(val_loader, args, shuffle=False)
        test_loader = _wrap_ddp_loader(test_loader, args, shuffle=False)

    if is_master(args):
        xb, yb, ab = next(iter(train_loader))
        print(f"[debug] batch x shape = {tuple(xb.shape)}")
        print(f"[debug] batch y shape = {tuple(yb.shape)}")
        print(f"[debug] batch a shape = {tuple(ab.shape)}")
        print(f"[debug] expected S = {args.seq_len - args.history + 1}")

    if args.distributed and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    edge_index = fully_connected_edge_index(
        C,
        self_loops=False,
        device=device if isinstance(device, str) and device.startswith("cuda") else None,
    )

    sl = GTSStructureLearnerGRU(
        observed_neurons=C,
        total_neurons=C,
        hidden_neurons=0,

        # Conv1D kernel size
        # ex) 80
        history=args.conv_filter_size,

        n_hid=args.n_hid,
        HD_observed=None,
        HD_hidden=None,
        HD_total=None,
        edge_index=edge_index,
        out_channel=args.out_channel,

        # input sequence length
        # ex) 400
        total_steps=args.seq_len,

        # final effective receptive field
        # ex) 200
        receptive_field=args.history,

        st_x=args.conv_stride,
        use_logvar=False,
        embed_mode=args.embed_mode,
        emb_dim=args.posemb_dim,
        embed_ln=False,
        embed_dropout=0.0,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        edge_chunk_num=args.edge_chunk_num,
        # New dilated conv settings
        # Original per-window equivalent:
        #   Conv1D kernel=80, stride=40, out=32
        #   -> 4 tokens
        #
        # Full sequence equivalent:
        #   Conv1D kernel=80, stride=1
        #   -> Conv2D kernel=(32,4), dilation=(1,40)
        local_conv_kernel=args.conv_filter_size,
        local_conv_stride=args.local_conv_stride,
        conv2_time_kernel=args.conv2_time_kernel,
    )

    shared_conv = nn.Conv1d(
        1,
        args.n_hid,
        kernel_size=args.history,
        stride=1,
    )

    sp = sp_ver2(
        observed_neurons=C,
        total_neurons=C,
        hidden_neurons=0,
        history=args.history,
        n_hid=args.n_hid,
        HD_observed=None,
        HD_hidden=None,
        HD_total=None,
        edge_idx=edge_index,
        session="TH",
        trial=False,
        msg_hop=1,
        do_prob=0.0,
        shared_conv=shared_conv,
        residual_flag=args.residual_flag,
    )

    sl = sl.to(device)
    sp = sp.to(device)

    if args.distributed and torch.cuda.is_available():
        sl = DDP(
            sl,
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=False,
        )

        if any(p.requires_grad for p in sp.parameters()):
            sp = DDP(
                sp,
                device_ids=[args.gpu],
                output_device=args.gpu,
                find_unused_parameters=False,
            )
            sp.module.edge_index = edge_index
        else:
            sp.edge_index = edge_index

        sl.module.edge_index = edge_index

    else:
        sl.edge_index = edge_index
        sp.edge_index = edge_index

    res = fit_gts_spver2_perbatchSL_clean(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        sl_module=sl,
        sp_module=sp,
        device=device,
        lr=args.lr,
        minimum_lr=args.minimum_lr,
        max_epoch=args.max_epoch,
        patience=args.patience,
        pred_steps=args.pred_steps,
        save_root=args.save_root,
        exp_name=args.exp_name,
        beta_l2=args.beta_l2,
        alpha_poisson=args.alpha_poisson,
        alpha_mse=args.alpha_mse,
        decay_step=args.decay_step,
        gamma=args.gamma,
        save_every=args.save_every,
        max_grad_norm=args.max_grad_norm,
        use_wandb=(is_master(args) and args.use_wandb and (wandb is not None)),
        save_edge_snapshot=args.save_edge_snapshot,
    )

    if is_master(args):
        be = res["best_state"]["epoch"] if res["best_state"] else None
        print("Best epoch:", be)
        print("Best val:", res["best_val"], " Test loss:", res["test_loss"])

    cleanup()


if __name__ == "__main__":
    main()