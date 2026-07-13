# ldns/train/fit_stdp_perbatchSL_clean.py
import os, time, gc, random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from ldns.utils.graph import make_z_sym_gts

__all__ = ["fit_gts_spver2_perbatchSL_clean"]

LOG_MIN = -20.0


def _is_rank0():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _reduce_preds_to_BC(preds: torch.Tensor) -> torch.Tensor:
    if preds.dim() == 4:
        preds = preds[..., -1, 0]  # [C,B]
        return preds.permute(1, 0).contiguous()
    elif preds.dim() == 3 and preds.size(-1) == 1:
        preds = preds[..., 0]      # [C,B]
        return preds.permute(1, 0).contiguous()
    elif preds.dim() == 2:
        return preds.permute(1, 0).contiguous()
    raise ValueError(f"Unexpected preds shape: {tuple(preds.shape)}")


def _ensure_BC_targets(y_spk: torch.Tensor, B: int, C: int) -> torch.Tensor:
    if y_spk.dim() != 2:
        raise ValueError(f"y_spk must be 2D, got {tuple(y_spk.shape)}")

    if y_spk.shape == (B, C):
        return y_spk
    if y_spk.shape == (C, B):
        return y_spk.t().contiguous()

    raise ValueError(
        f"y_spk shape mismatch: got {tuple(y_spk.shape)}, "
        f"expected (B,C)=({B},{C}) or (C,B)=({C},{B})"
    )


def _repeat_ang_for_sp_chunk(ang, Sc: int, B: int):
    """
    기존 ang: [B] 또는 [B,...]
    chunked ang: [Sc*B] 또는 [Sc*B,...]

    ang가 batch dimension을 가지면 S chunk만큼 반복.
    batch dimension이 없으면 그대로 반환.
    """
    if not torch.is_tensor(ang):
        return ang

    if ang.dim() >= 1 and ang.size(0) == B:
        return (
            ang.unsqueeze(0)              # [1,B,...]
            .expand(Sc, *ang.shape)       # [Sc,B,...]
            .reshape(Sc * B, *ang.shape[1:])
            .contiguous()
        )

    return ang


def _sl_edges_sequence_det(x_spk_TBC: torch.Tensor, sl_module, C: int) -> torch.Tensor:
    """
    recurrent SL:
      x_spk_TBC: [Tlong, B, C]
      return:
        mu_seq: [B, S, E, 1]
    """
    Tlong, B, Cobs = x_spk_TBC.shape
    if Cobs != C:
        raise ValueError(f"SL expects C={C}, got {Cobs}")

    x_bct = x_spk_TBC.permute(1, 2, 0).contiguous()  # [B,C,Tlong]

    out = sl_module(x_bct)
    mu_seq = out[0] if isinstance(out, (tuple, list)) else out

    if not (mu_seq.dim() == 4 and mu_seq.size(0) == B and mu_seq.size(-1) == 1):
        raise ValueError(f"Recurrent SL must return [B,S,E,1], got {tuple(mu_seq.shape)}")

    return mu_seq


def _ensure_y_seq_SBC(y_spk: torch.Tensor, S: int, B: int, C: int) -> torch.Tensor:
    """
    y_spk를 [S,B,C]로 강제.
    허용:
      [S,B,C]
      [B,S,C]
    """
    if y_spk.dim() != 3:
        raise ValueError(f"y_spk must be 3D sequence target, got {tuple(y_spk.shape)}")

    if y_spk.shape == (S, B, C):
        return y_spk
    if y_spk.shape == (B, S, C):
        return y_spk.permute(1, 0, 2).contiguous()

    raise ValueError(
        f"y_spk shape mismatch: got {tuple(y_spk.shape)}, "
        f"expected [S,B,C]=[{S},{B},{C}] or [B,S,C]=[{B},{S},{C}]"
    )


@torch.no_grad()
def _rand_val_edge_from_one_sample(sl_module, loader, device, C: int) -> torch.Tensor:
    """
    snapshot용:
      recurrent SL output [B,S,E,1] 중 랜덤 batch/sample과 마지막 recurrent step 사용.
    """
    it = iter(loader)
    batch = None

    for _ in range(random.randint(0, 2) + 1):
        try:
            batch = next(it)
        except StopIteration:
            break

    if batch is None:
        batch = next(iter(loader))

    x_spk, _, _ = batch
    x_spk = x_spk.to(device, non_blocking=True)  # [Tlong,B,C]

    mu_seq = _sl_edges_sequence_det(x_spk, sl_module, C)  # [B,S,E,1]

    b = random.randint(0, mu_seq.size(0) - 1)
    s = mu_seq.size(1) - 1

    mu_one = mu_seq[b, s].contiguous()  # [E,1]
    return make_z_sym_gts(mu_one, C).cpu()


def _sequence_loss_one_batch(
    x_spk: torch.Tensor,
    y_spk: torch.Tensor,
    ang: torch.Tensor,
    sl_module,
    sp_module,
    criterion,
    C: int,
    pred_steps: int,
    alpha_poisson: float,
    alpha_mse: float,
    beta_l2: float,
    sp_chunk_s: int = 128,
):
    """
    x_spk: [Tlong, B, C]
    y_spk: [S, B, C] 또는 [B, S, C]

    recurrent SL:
      x_spk -> mu_seq [B,S,E,1]

    chunked SP:
      기존:
        for s:
          x_win = x_spk[s:s+H]    [H,B,C]
          z_t   = mu_seq[:,s]     [B,E,1]
          y_t   = y_seq[s]        [B,C]

      변경:
        S축을 chunk 단위로 batch 축에 합침.
          x_flat = [H, Sc*B, C]
          z_flat = [Sc*B, E, 1]
          y_flat = [Sc*B, C]
    """
    if x_spk.dim() != 3:
        raise ValueError(f"x_spk must be [Tlong,B,C], got {tuple(x_spk.shape)}")

    Tlong, B, Cobs = x_spk.shape
    if Cobs != C:
        raise ValueError(f"Input C mismatch: got {Cobs}, expected {C}")

    sp_inner = getattr(sp_module, "module", sp_module)
    sl_inner = getattr(sl_module, "module", sl_module)

    H = int(getattr(sp_inner, "history", None) or getattr(sl_inner, "history", None))
    if Tlong < H:
        raise ValueError(f"Tlong={Tlong} must be >= history={H}")

    # ============================================================
    # 1. SL module: 한 batch당 1회 호출
    # ============================================================
    mu_seq = _sl_edges_sequence_det(x_spk, sl_module, C)  # [B,S_sl,E,1]
    S_sl = mu_seq.size(1)
    E = mu_seq.size(2)

    y_seq = _ensure_y_seq_SBC(y_spk, S_sl, B, C)          # [S_sl,B,C]
    S = y_seq.size(0)

    if S != S_sl:
        raise ValueError(f"y_seq S={S} must match SL output S={S_sl}")

    if Tlong - H + 1 != S:
        raise ValueError(
            f"x/y mismatch: Tlong={Tlong}, H={H} gives S={Tlong-H+1}, "
            f"but y_seq has S={S}"
        )

    if sp_chunk_s is None or sp_chunk_s <= 0:
        sp_chunk_s = S
    sp_chunk_s = min(int(sp_chunk_s), S)

    # ============================================================
    # 2. sliding windows 생성
    #    x_spk: [Tlong,B,C]
    #    unfold -> [S,B,C,H]
    #    permute -> [S,H,B,C]
    # ============================================================
    x_windows = x_spk.unfold(dimension=0, size=H, step=1)  # [S,B,C,H]
    x_windows = x_windows.permute(0, 3, 1, 2).contiguous() # [S,H,B,C]

    # ============================================================
    # 3. Chunked SP prediction
    # ============================================================
    loss_rec_sum = 0.0

    for s0 in range(0, S, sp_chunk_s):
        s1 = min(s0 + sp_chunk_s, S)
        Sc = s1 - s0

        # --------------------------------------------------------
        # x_chunk:
        # [Sc,H,B,C] -> [H,Sc,B,C] -> [H,Sc*B,C]
        # --------------------------------------------------------
        x_chunk = x_windows[s0:s1]

        x_flat = (
            x_chunk
            .permute(1, 0, 2, 3)        # [H,Sc,B,C]
            .reshape(H, Sc * B, C)      # [H,Sc*B,C]
            .contiguous()
        )

        # --------------------------------------------------------
        # z_chunk:
        # mu_seq [B,S,E,1]
        # -> [B,Sc,E,1]
        # -> [Sc,B,E,1]
        # -> [Sc*B,E,1]
        # --------------------------------------------------------
        z_flat = (
            mu_seq[:, s0:s1, :, :]      # [B,Sc,E,1]
            .permute(1, 0, 2, 3)        # [Sc,B,E,1]
            .reshape(Sc * B, E, 1)      # [Sc*B,E,1]
            .contiguous()
        )

        # --------------------------------------------------------
        # y_chunk:
        # y_seq [S,B,C]
        # -> [Sc,B,C]
        # -> [Sc*B,C]
        # --------------------------------------------------------
        y_flat = (
            y_seq[s0:s1]                # [Sc,B,C]
            .reshape(Sc * B, C)         # [Sc*B,C]
            .contiguous()
        )

        # --------------------------------------------------------
        # ang:
        # [B] or [B,...] -> [Sc*B] or [Sc*B,...]
        # --------------------------------------------------------
        ang_flat = _repeat_ang_for_sp_chunk(ang, Sc, B)

        preds = sp_module(
            x_flat,
            ang_flat,
            z_flat,
            pred_steps=pred_steps,
        )

        preds = torch.clamp(preds, min=LOG_MIN, max=40)
        preds_BC = _reduce_preds_to_BC(preds)  # [Sc*B,C]

        if preds_BC.shape != y_flat.shape:
            raise ValueError(
                f"SP pred/target mismatch: preds_BC={tuple(preds_BC.shape)}, "
                f"y_flat={tuple(y_flat.shape)}"
            )

        loss_poisson = criterion(preds_BC, y_flat)
        loss_mse = nn.functional.mse_loss(torch.exp(preds_BC), y_flat)

        loss_rec_chunk = alpha_poisson * loss_poisson + alpha_mse * loss_mse

        # criterion이 mean reduction이므로, 기존 s별 평균 loss 합산 방식에 맞춰 Sc만큼 가중.
        loss_rec_sum = loss_rec_sum + loss_rec_chunk * Sc

    loss_rec = loss_rec_sum / S

    # 기존:
    #   loss_l2_sum += 0.5 * z_t.pow(2).mean()
    #   loss_l2 = loss_l2_sum / S
    #
    # 전체 mu_seq mean과 동일.
    loss_l2 = 0.5 * mu_seq.pow(2).mean()

    loss = loss_rec + beta_l2 * loss_l2

    return loss, loss_rec, loss_l2, B


def train_epoch_perbatchSL(
    train_loader,
    sl_module,
    sp_module,
    optimizer,
    criterion,
    device,
    C: int,
    pred_steps: int = 1,
    beta_l2: float = 1.0,
    max_grad_norm: float = 1.0,
    alpha_poisson: float = 1.0,
    alpha_mse: float = 0.0,
    sp_chunk_s: int = 128,
):
    sl_module.train()
    sp_module.train()

    total_sum = 0.0
    rec_sum = 0.0
    l2_sum = 0.0
    steps = 0

    for x_spk, y_spk, ang in train_loader:
        x_spk = x_spk.to(device, non_blocking=True)
        y_spk = y_spk.to(device, non_blocking=True)
        ang = ang.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        loss, loss_rec, loss_l2, _ = _sequence_loss_one_batch(
            x_spk=x_spk,
            y_spk=y_spk,
            ang=ang,
            sl_module=sl_module,
            sp_module=sp_module,
            criterion=criterion,
            C=C,
            pred_steps=pred_steps,
            alpha_poisson=alpha_poisson,
            alpha_mse=alpha_mse,
            beta_l2=beta_l2,
            sp_chunk_s=sp_chunk_s,
        )

        loss.backward()

        if max_grad_norm is not None and max_grad_norm > 0:
            sl_params = [p for p in sl_module.parameters() if p.requires_grad]
            sp_params = [p for p in sp_module.parameters() if p.requires_grad]

            if len(sl_params) > 0:
                nn.utils.clip_grad_norm_(sl_params, max_grad_norm)
            if len(sp_params) > 0:
                nn.utils.clip_grad_norm_(sp_params, max_grad_norm)

        optimizer.step()

        total_sum += float(loss.detach().item())
        rec_sum += float(loss_rec.detach().item())
        l2_sum += float(loss_l2.detach().item())
        steps += 1

    denom = max(steps, 1)
    return {
        "train_total": total_sum / denom,
        "train_rec": rec_sum / denom,
        "train_l2": l2_sum / denom,
    }


@torch.no_grad()
def evaluate_perbatchSL(
    loader,
    sl_module,
    sp_module,
    criterion,
    device,
    C: int,
    pred_steps: int = 1,
    alpha_poisson: float = 1.0,
    alpha_mse: float = 0.0,
    beta_l2: float = 0.0,
    sp_chunk_s: int = 128,
):
    sl_module.eval()
    sp_module.eval()

    loss_sum = 0.0
    n = 0

    for x_spk, y_spk, ang in loader:
        x_spk = x_spk.to(device, non_blocking=True)
        y_spk = y_spk.to(device, non_blocking=True)
        ang = ang.to(device, non_blocking=True)

        loss, _, _, B = _sequence_loss_one_batch(
            x_spk=x_spk,
            y_spk=y_spk,
            ang=ang,
            sl_module=sl_module,
            sp_module=sp_module,
            criterion=criterion,
            C=C,
            pred_steps=pred_steps,
            alpha_poisson=alpha_poisson,
            alpha_mse=alpha_mse,
            beta_l2=beta_l2,
            sp_chunk_s=sp_chunk_s,
        )

        loss_sum += float(loss.item()) * B
        n += B

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        t = torch.tensor([loss_sum, n], dtype=torch.float64, device=device)
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        loss_sum, n = t[0].item(), int(t[1].item())

    return loss_sum / max(n, 1)


def fit_gts_spver2_perbatchSL_clean(
    train_loader,
    val_loader,
    test_loader,
    sl_module,
    sp_module,
    device="cuda",
    lr=5e-4,
    minimum_lr=1e-5,
    max_epoch=200,
    patience=20,
    pred_steps=1,
    save_root="./storage/exp",
    exp_name="run1",
    beta_l2: float = 1.0,
    alpha_poisson: float = 1.0,
    alpha_mse: float = 0.0,
    decay_step=50,
    gamma=0.5,
    save_every=1,
    max_grad_norm=1.0,
    use_wandb=False,
    save_edge_snapshot: bool = True,
    sp_chunk_s: int = 128,
):
    os.makedirs(save_root, exist_ok=True)

    ckpt_dir = os.path.join(save_root, "model_ckpt", exp_name)
    w_dir = os.path.join(save_root, "weight_ckpt", exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(w_dir, exist_ok=True)

    run_start = time.strftime("%Y%m%d_%H%M%S")
    snap_dir = os.path.join(w_dir, f"connectivity_{run_start}")
    os.makedirs(snap_dir, exist_ok=True)

    best_start_epoch = 5

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    sl_module = sl_module.to(device)
    sp_module = sp_module.to(device)

    try:
        x0, y0, a0 = next(iter(train_loader))
        C = int(x0.shape[2])
    except Exception:
        C = int(getattr(sl_module, "observed_neurons", None) or getattr(sp_module, "observed_neurons", None))
        if not C:
            raise ValueError(
                "Cannot infer C. Provide non-empty train_loader or ensure modules have observed_neurons."
            )

    criterion = nn.PoissonNLLLoss(log_input=True).to(device)

    opt_params = list(sl_module.parameters()) + list(sp_module.parameters())
    if len(opt_params) == 0:
        raise ValueError("No trainable parameters found in sl_module + sp_module.")

    optimizer = torch.optim.Adam(opt_params, lr=lr)

    init_lr = lr
    min_lr = minimum_lr

    if init_lr <= min_lr:
        min_factor = 1.0
    else:
        min_factor = min_lr / init_lr

    def lr_lambda(epoch):
        factor = gamma ** (epoch // decay_step)
        return max(factor, min_factor)

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_val = float("inf")
    best_state = None
    no_improve = 0
    last_edge_snapshot = None

    if _is_rank0():
        print(f"[SP chunked prediction] sp_chunk_s={sp_chunk_s}")

    for epoch in range(max_epoch):
        try:
            from torch.utils.data.distributed import DistributedSampler
            if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
        except Exception:
            pass

        t_train_start = time.time()

        train_stats = train_epoch_perbatchSL(
            train_loader=train_loader,
            sl_module=sl_module,
            sp_module=sp_module,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            C=C,
            pred_steps=pred_steps,
            beta_l2=beta_l2,
            max_grad_norm=max_grad_norm,
            alpha_poisson=alpha_poisson,
            alpha_mse=alpha_mse,
            sp_chunk_s=sp_chunk_s,
        )

        train_time = time.time() - t_train_start

        t_val_start = time.time()

        val_loss = evaluate_perbatchSL(
            loader=val_loader,
            sl_module=sl_module,
            sp_module=sp_module,
            criterion=criterion,
            device=device,
            C=C,
            pred_steps=pred_steps,
            alpha_poisson=alpha_poisson,
            alpha_mse=alpha_mse,
            beta_l2=0.0,
            sp_chunk_s=sp_chunk_s,
        )

        val_time = time.time() - t_val_start

        curr_lr = optimizer.param_groups[0]["lr"]

        if _is_rank0():
            print(
                f"[Epoch {epoch:03d}] lr={curr_lr:.6g}  "
                f"train_total={train_stats['train_total']:.6f}  "
                f"train_rec={train_stats['train_rec']:.6f}  "
                f"train_l2={train_stats['train_l2']:.6f}  "
                f"val={val_loss:.6f}  "
                f"train_time={train_time:.2f}s  "
                f"val_time={val_time:.2f}s  "
                f"total={train_time + val_time:.2f}s"
            )

        if use_wandb and _is_rank0():
            import wandb
            wandb.log({
                "epoch": epoch,
                "lr": curr_lr,
                **train_stats,
                "val_loss": val_loss,
            })

        if save_edge_snapshot and _is_rank0() and (epoch > 0) and (epoch % save_every == 0):
            try:
                edge_vec = _rand_val_edge_from_one_sample(sl_module, val_loader, device, C)
                last_edge_snapshot = edge_vec
                torch.save(edge_vec, os.path.join(snap_dir, f"edge_epoch{epoch:04d}.pt"))
            except Exception as e:
                print("[snapshot] skipped:", e)

        if epoch >= best_start_epoch:
            if val_loss < best_val:
                best_val = val_loss
                no_improve = 0

                if _is_rank0():
                    best_state = {
                        "sl": (getattr(sl_module, "module", sl_module)).state_dict(),
                        "sp": (getattr(sp_module, "module", sp_module)).state_dict(),
                        "epoch": epoch,
                        "val_loss": best_val,
                    }
                    torch.save(best_state, os.path.join(ckpt_dir, "best_model.pth"))
            else:
                no_improve += 1
                if no_improve > patience:
                    if _is_rank0():
                        print("Early stopping.")
                    break

        scheduler.step()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ======================================================
    # Load best checkpoint on all ranks
    # ======================================================
    best_path = os.path.join(ckpt_dir, "best_model.pth")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    if os.path.exists(best_path):
        best_state = torch.load(best_path, map_location=device)

        sl_inner = getattr(sl_module, "module", sl_module)
        sp_inner = getattr(sp_module, "module", sp_module)

        sl_inner.load_state_dict(best_state["sl"])
        sp_inner.load_state_dict(best_state["sp"])
    else:
        if _is_rank0():
            print("[warning] best_model.pth not found. Using last epoch model.")

    test_loss = evaluate_perbatchSL(
        loader=test_loader,
        sl_module=sl_module,
        sp_module=sp_module,
        criterion=criterion,
        device=device,
        C=C,
        pred_steps=pred_steps,
        alpha_poisson=alpha_poisson,
        alpha_mse=alpha_mse,
        beta_l2=0.0,
        sp_chunk_s=sp_chunk_s,
    )

    if _is_rank0():
        print(f"[TEST] loss={test_loss:.6f}")

    return {
        "best_state": best_state,
        "best_val": best_val,
        "test_loss": test_loss,
        "last_edge_snapshot": last_edge_snapshot,
    }