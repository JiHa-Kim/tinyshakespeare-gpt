import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

from scionc.compile_env import ensure_compile_env
from scionc.optim.parametrization import (
    resolve_schedule,
    retention_from_half_life,
    schedule_at_step,
)
from scionc.optim.setup import (
    DEFAULT_HYPERBALL_UPDATE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_STATE_HALF_LIFE,
    GROUP_NAMES,
    OPTIMIZER_NAME,
    apply_scheduled_lr,
    build_optimizer,
    count_increment,
    format_optimizer_schedule,
    optimizer_io_label,
    optimizer_rms_state,
    resolve_hyperball_update,
    rms_state_text,
)
from scionc.models.gpt import (
    Derf,
    EquivariantLowRankKV,
    GPT,
    BatchSource,
    CharDataset,
    GPTConfig,
    RMSNorm,
    maybe_download_tiny_shakespeare,
)
from scionc.probes.convergence import ConvergenceProbe
from scionc.probes.line import (
    apply_line_scale,
    capture_params,
    capture_rng,
    finish_line_snapshot,
    line_curve_stats,
    line_curve_text,
    line_probe_stats,
    line_probe_text,
    parse_line_scales,
    restore_rng,
)
from scionc.probes.optimizer_stats import (
    accumulate_step_stats,
    capture_step_stats,
    consume_step_stats,
)


def sync_now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def cuda_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    return {
        "alloc_gb": torch.cuda.memory_allocated(device) / 1e9,
        "reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9,
        "total_gb": torch.cuda.get_device_properties(device).total_memory / 1e9,
    }


def cuda_memory_text(device: torch.device) -> str:
    stats = cuda_memory_stats(device)
    if not stats:
        return ""
    return (
        f" | cuda_alloc {stats['alloc_gb']:.2f}G"
        f" | cuda_reserved {stats['reserved_gb']:.2f}G"
        f" | cuda_max_reserved {stats['max_reserved_gb']:.2f}G"
    )


def jsonable(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return str(value)


class MetricsLogger:
    def __init__(self, path: str, run_name: str = "") -> None:
        self.run_name = run_name
        self.file = None
        if path:
            metrics_path = Path(path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.file = metrics_path.open("a", encoding="utf-8", buffering=1)

    def write(self, event: str, **values) -> None:
        if self.file is None:
            return
        record = {"event": event, "run_name": self.run_name, **values}
        self.file.write(json.dumps(jsonable(record), separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


class NormalizedSGD:
    """Scion-style normalized descent for small unconstrained parameter groups."""

    def __init__(self, params, lr: float, beta: float, eps: float = 1e-12):
        self.params = [p for p in params if p.requires_grad]
        self.lr_peak = float(lr)
        self.lr = float(lr)
        self.beta = float(beta)
        self.eps = eps
        self.state: dict[torch.Tensor, torch.Tensor] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            if set_to_none:
                p.grad = None
            else:
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        total = 0
        sq = None
        updates = []
        for p in self.params:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("NormalizedSGD does not support sparse gradients")
            m = self.state.get(p)
            if m is None:
                m = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p] = m
            m.lerp_(p.grad, 1.0 - self.beta)
            m32 = m.float()
            value = m32.square().sum()
            sq = value if sq is None else sq + value
            total += p.numel()
            updates.append((p, m))

        if not updates or sq is None or total <= 0:
            return
        rms = (sq / total).sqrt()
        if float(rms) <= self.eps:
            return
        for p, m in updates:
            p.add_(m / rms.to(dtype=m.dtype), alpha=-self.lr)


def step_stats_text(stats: dict[str, dict]) -> str:
    if not stats:
        return ""
    parts = []
    for name, values in stats.items():
        parts.append(
            f"{name}:cos={values['cos']:.3f},"
            f"u/p={values['update_param_rms']:.2e},"
            f"u/g={values['update_grad_rms']:.2e},"
            f"g/p={values['grad_param_rms']:.2e},"
            f"xg={values['param_grad_cos']:.3f},"
            f"xu={values['param_update_cos']:.3f},"
            f"ga/r={values['grad_abs_rms']:.3f},"
            f"ua/r={values['update_abs_rms']:.3f},"
            f"gk={values['grad_kurtosis']:.2e},"
            f"uk={values['update_kurtosis']:.2e}"
        )
    return " | step_stats " + "; ".join(parts)


def amp_ctx(amp_dtype: torch.dtype | None):
    return (
        torch.autocast(device_type="cuda", dtype=amp_dtype)
        if amp_dtype is not None
        else nullcontext()
    )


@torch.inference_mode()
def estimate_loss(
    model: GPT,
    source: BatchSource,
    eval_iters: int,
    amp_dtype: torch.dtype | None,
    splits=("train", "val"),
    fixed_batches: dict[str, tuple[tuple[torch.Tensor, torch.Tensor], ...]]
    | None = None,
):
    was_training = model.training
    model.eval()
    out = {}
    with amp_ctx(amp_dtype):
        for split in splits:
            total = 0.0
            batches = fixed_batches.get(split) if fixed_batches else None
            if batches is None:
                batches = tuple(source.get(split) for _ in range(eval_iters))
            for xb, yb in batches:
                _, loss = model(xb, yb)
                total += float(loss)
            out[split] = total / len(batches)
    model.train(was_training)
    return out


def update_logit_stats(
    acc: dict[str, float], logits: torch.Tensor, targets: torch.Tensor
) -> None:
    values = logits.detach().float()
    token_count = values.numel() // values.size(-1)
    centered = values - values.mean(dim=-1, keepdim=True)
    log_probs = torch.log_softmax(values, dim=-1)
    probs = log_probs.exp()
    top2 = torch.topk(values, k=min(2, values.size(-1)), dim=-1).values
    top_margin = (
        top2[..., 0] - top2[..., 1]
        if top2.size(-1) > 1
        else torch.zeros_like(top2[..., 0])
    )
    target_probs = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    acc["tokens"] += float(token_count)
    acc["logit_var"] += float(centered.square().mean(dim=-1).sum())
    acc["logit_margin"] += float(top_margin.sum())
    acc["softmax_entropy"] += float((-(probs * log_probs).sum(dim=-1)).sum())
    acc["softmax_max_prob"] += float(probs.max(dim=-1).values.sum())
    acc["target_prob"] += float(target_probs.sum())


def finalize_logit_stats(acc: dict[str, float]) -> dict[str, float]:
    tokens = max(acc.get("tokens", 0.0), 1.0)
    return {
        "logit_std": math.sqrt(max(0.0, acc["logit_var"] / tokens)),
        "logit_margin": acc["logit_margin"] / tokens,
        "softmax_entropy": acc["softmax_entropy"] / tokens,
        "softmax_max_prob": acc["softmax_max_prob"] / tokens,
        "target_prob": acc["target_prob"] / tokens,
    }


@torch.inference_mode()
def estimate_val_metrics(
    model: GPT,
    source: BatchSource,
    eval_iters: int,
    amp_dtype: torch.dtype | None,
    collect_logit_stats: bool = False,
    fixed_batches: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
) -> tuple[float, dict[str, float]]:
    was_training = model.training
    model.eval()
    total = 0.0
    logit_acc = {
        "tokens": 0.0,
        "logit_var": 0.0,
        "logit_margin": 0.0,
        "softmax_entropy": 0.0,
        "softmax_max_prob": 0.0,
        "target_prob": 0.0,
    }
    batches = fixed_batches
    if batches is None:
        batches = tuple(source.get("val") for _ in range(eval_iters))
    with amp_ctx(amp_dtype):
        for xb, yb in batches:
            logits, loss = model(xb, yb)
            total += float(loss)
            if collect_logit_stats:
                update_logit_stats(logit_acc, logits, yb)
    model.train(was_training)
    stats = finalize_logit_stats(logit_acc) if collect_logit_stats else {}
    return total / len(batches), stats


def save_checkpoint(path: Path, model: GPT, dataset: CharDataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "model_cfg": asdict(model.cfg),
            "chars": dataset.chars,
        },
        path,
    )


def save_eval_checkpoint(
    path: Path,
    step: int,
    val_loss: float,
    model: GPT,
    dataset: CharDataset,
    args,
):
    if args.save_interval <= 0 or step % args.save_interval != 0:
        return
    eval_path = path.with_name(
        f"{path.stem}_step{step:05d}_val{val_loss:.4f}{path.suffix}"
    )
    save_checkpoint(eval_path, model, dataset)


def load_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    chars = ckpt["chars"]
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    model = GPT(GPTConfig(**ckpt["model_cfg"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, stoi, itos


def maybe_compile(
    model: GPT,
    source: BatchSource,
    args,
    amp_dtype: torch.dtype | None,
    device: torch.device,
):
    if not (args.compile and hasattr(torch, "compile")):
        return model, 0.0
    ensure_compile_env()
    model = torch.compile(model)
    xb, yb = source.seeded_batch("train", resolve_compile_seed(args))
    t0 = sync_now(device)
    model.zero_grad(set_to_none=True)
    with amp_ctx(amp_dtype):
        _, loss = model(xb, yb)
    loss.backward()
    model.zero_grad(set_to_none=True)
    was_training = model.training
    model.eval()
    with torch.no_grad(), amp_ctx(amp_dtype):
        model(xb, yb)
    model.train(was_training)
    return model, sync_now(device) - t0


def resolve_data_seed(args) -> int:
    return args.data_seed if args.data_seed is not None else args.seed


def resolve_eval_seed(args) -> int:
    return args.eval_seed if args.eval_seed is not None else args.seed + 1


def resolve_compile_seed(args) -> int:
    return args.compile_seed if args.compile_seed is not None else args.seed + 2


def split_eval_seed(args, split: str) -> int:
    offset = 0 if split == "val" else 1_000_003
    return resolve_eval_seed(args) + offset


def fixed_eval_batches(args, source: BatchSource):
    if not args.fixed_eval_batches:
        return None
    return {
        split: source.fixed_batches(split, args.eval_iters, split_eval_seed(args, split))
        for split in ("train", "val")
    }


def configure_runtime(args) -> tuple[torch.device, torch.dtype | None]:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(args.deterministic, warn_only=True)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = not args.deterministic
        torch.backends.cudnn.allow_tf32 = not args.deterministic
        torch.backends.cudnn.benchmark = not args.deterministic

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else None
    )
    return device, amp_dtype


def load_dataset(args) -> CharDataset:
    data_path = Path(args.data_path)
    maybe_download_tiny_shakespeare(data_path)
    return CharDataset(data_path)


def build_model(args, dataset: CharDataset, device: torch.device) -> GPT:
    cfg = GPTConfig(
        vocab_size=len(dataset.chars),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        rope_base=args.rope_base,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
        norm_type=args.norm_type,
        derf_alpha=args.derf_alpha,
        derf_shift=args.derf_shift,
        attn_type=args.attn_type,
        kv_cache=args.kv_cache,
        kv_key_rank=args.kv_key_rank,
        kv_value_rank=args.kv_value_rank,
        resid_scale=args.resid_scale,
        block_type=args.block_type,
        deepnorm_alpha=args.deepnorm_alpha,
        deepnorm_branch_scale=args.deepnorm_branch_scale,
        lns=args.lns,
    )
    return GPT(cfg).to(device)


def _rms_float(x: torch.Tensor) -> float:
    return float(x.detach().float().square().mean().sqrt())


@torch.no_grad()
def calibrate_deepnorm_branches(
    model: GPT, idx: torch.Tensor, eps: float = 1e-12
) -> dict[str, float]:
    if model.cfg.block_type != "deepnorm":
        return {}

    target_delta = 1.0 / math.sqrt(2.0 * model.cfg.n_layer)
    was_training = model.training
    model.eval()
    attn_scales = []
    mlp_scales = []
    attn_ratios_before = []
    mlp_ratios_before = []
    attn_ratios_after = []
    mlp_ratios_after = []
    try:
        x = model.tok_emb(idx)
        for block in model.blocks:
            if block.block_type != "deepnorm":
                x = block(x)
                continue
            if block.post_attn_norm is None or block.post_mlp_norm is None:
                raise RuntimeError("DeepNorm post-norms are missing")

            alpha = float(block.deepnorm_alpha)
            attn_out = block.attn(x)
            attn_den = max(alpha * _rms_float(x), eps)
            attn_raw = _rms_float(attn_out)
            attn_ratio = attn_raw / attn_den
            attn_scale = target_delta / max(attn_ratio, eps)
            block.deepnorm_attn_scale.fill_(attn_scale)
            x = block.post_attn_norm(alpha * x + attn_scale * attn_out)

            mlp_out = block.mlp(x)
            mlp_den = max(alpha * _rms_float(x), eps)
            mlp_raw = _rms_float(mlp_out)
            mlp_ratio = mlp_raw / mlp_den
            mlp_scale = target_delta / max(mlp_ratio, eps)
            block.deepnorm_mlp_scale.fill_(mlp_scale)
            x = block.post_mlp_norm(alpha * x + mlp_scale * mlp_out)

            attn_scales.append(attn_scale)
            mlp_scales.append(mlp_scale)
            attn_ratios_before.append(attn_ratio)
            mlp_ratios_before.append(mlp_ratio)
            attn_ratios_after.append(attn_ratio * attn_scale)
            mlp_ratios_after.append(mlp_ratio * mlp_scale)
    finally:
        model.train(was_training)

    def stats(values: list[float], prefix: str) -> dict[str, float]:
        if not values:
            return {}
        return {
            f"{prefix}_mean": sum(values) / len(values),
            f"{prefix}_min": min(values),
            f"{prefix}_max": max(values),
        }

    return {
        "target_delta": target_delta,
        **stats(attn_scales, "attn_scale"),
        **stats(mlp_scales, "mlp_scale"),
        **stats(attn_ratios_before, "attn_ratio_before"),
        **stats(mlp_ratios_before, "mlp_ratio_before"),
        **stats(attn_ratios_after, "attn_ratio_after"),
        **stats(mlp_ratios_after, "mlp_ratio_after"),
    }


def deepnorm_calibration_text(stats: dict[str, float]) -> str:
    if not stats:
        return ""
    return (
        "deepnorm_calibration "
        f"target_delta={stats['target_delta']:.6g} "
        f"attn_scale={stats['attn_scale_mean']:.4g}"
        f"[{stats['attn_scale_min']:.4g},{stats['attn_scale_max']:.4g}] "
        f"mlp_scale={stats['mlp_scale_mean']:.4g}"
        f"[{stats['mlp_scale_min']:.4g},{stats['mlp_scale_max']:.4g}] "
        f"attn_ratio_before={stats['attn_ratio_before_mean']:.4g} "
        f"mlp_ratio_before={stats['mlp_ratio_before_mean']:.4g}"
    )


def configure_derf_training(model: GPT, args) -> None:
    if args.train_derf_shape:
        return
    for module in model.modules():
        if isinstance(module, Derf):
            module.alpha.requires_grad_(False)
            module.shift.requires_grad_(False)


def derf_parameter_groups(model: GPT) -> dict[str, list[torch.Tensor]]:
    groups = {"shape": [], "affine": []}
    for module in model.modules():
        if isinstance(module, Derf):
            groups["shape"].extend(
                p for p in (module.alpha, module.shift) if p.requires_grad
            )
            groups["affine"].extend(
                p for p in (module.gamma, module.beta) if p.requires_grad
            )
        elif isinstance(module, RMSNorm) and module.gamma is not None:
            groups["affine"].append(module.gamma)
    return groups


def build_derf_optimizers(model: GPT, args) -> dict[str, NormalizedSGD]:
    groups = {
        name: params for name, params in derf_parameter_groups(model).items() if params
    }
    if not groups:
        return {}
    beta = retention_from_half_life(
        count_increment(args), args.derf_state_half_life, "derf_state_half_life"
    )
    return {
        name: NormalizedSGD(params, args.derf_lr, beta)
        for name, params in groups.items()
    }


def zero_derf_optimizers(
    opts: dict[str, NormalizedSGD], set_to_none: bool = True
) -> None:
    for opt in opts.values():
        opt.zero_grad(set_to_none=set_to_none)


def step_derf_optimizers(opts: dict[str, NormalizedSGD]) -> None:
    for opt in opts.values():
        opt.step()


def kv_decoder_parameters(model: GPT) -> list[torch.Tensor]:
    params = []
    for module in model.modules():
        if isinstance(module, EquivariantLowRankKV):
            params.extend(p for p in module.decoder_parameters() if p.requires_grad)
    return params


def build_kv_decoder_optimizer(model: GPT, args) -> NormalizedSGD | None:
    params = kv_decoder_parameters(model)
    if not params:
        return None
    beta = retention_from_half_life(
        count_increment(args), args.state_half_life, "kv_decoder_state_half_life"
    )
    return NormalizedSGD(params, args.kv_decoder_lr, beta)


def kv_cache_summary(model: GPT) -> dict[str, float | int | str]:
    modules = [m for m in model.modules() if isinstance(m, EquivariantLowRankKV)]
    if not modules:
        attn = model.blocks[0].attn
        original = 2 * attn.n_head * attn.head_dim
        return {
            "type": "full",
            "cache_dim": original,
            "original_cache_dim": original,
            "cache_ratio": 1.0,
            "key_rank": attn.n_head,
            "value_rank": attn.n_head * attn.head_dim,
        }
    module = modules[0]
    return {
        "type": "equivariant-lowrank",
        "cache_dim": module.cache_dim,
        "original_cache_dim": module.original_cache_dim,
        "cache_ratio": module.cache_dim / module.original_cache_dim,
        "key_rank": module.key_rank,
        "value_rank": module.value_rank,
    }


def parameter_summary(model: GPT) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


@torch.no_grad()
def kv_decoder_state(model: GPT) -> dict[str, float]:
    key_values = []
    value_values = []
    for module in model.modules():
        if isinstance(module, EquivariantLowRankKV):
            key_values.append(module.key_decoder.detach().float().flatten())
            value_values.append(module.value_decoder.detach().float().flatten())
    if not key_values:
        return {}
    key = torch.cat(key_values)
    value = torch.cat(value_values)
    return {
        "key_decoder_rms": float(key.square().mean().sqrt()),
        "key_decoder_min": float(key.min()),
        "key_decoder_max": float(key.max()),
        "value_decoder_rms": float(value.square().mean().sqrt()),
        "value_decoder_min": float(value.min()),
        "value_decoder_max": float(value.max()),
    }


def kv_decoder_state_text(state: dict[str, float]) -> str:
    if not state:
        return ""
    return (
        " | kv_decoder "
        f"k_rms={state['key_decoder_rms']:.3g},"
        f"k=[{state['key_decoder_min']:.3g},{state['key_decoder_max']:.3g}],"
        f"v_rms={state['value_decoder_rms']:.3g},"
        f"v=[{state['value_decoder_min']:.3g},{state['value_decoder_max']:.3g}]"
    )


@torch.no_grad()
def derf_state(model: GPT) -> dict[str, float]:
    alphas = []
    shifts = []
    gammas = []
    betas = []
    for module in model.modules():
        if isinstance(module, Derf):
            alphas.append(float(module.alpha))
            shifts.append(float(module.shift))
            gammas.extend(float(x) for x in module.gamma.detach().flatten())
            betas.extend(float(x) for x in module.beta.detach().flatten())
    if not alphas:
        return {}
    return {
        "alpha_mean": sum(alphas) / len(alphas),
        "alpha_min": min(alphas),
        "alpha_max": max(alphas),
        "shift_mean": sum(shifts) / len(shifts),
        "shift_min": min(shifts),
        "shift_max": max(shifts),
        "gamma_mean": sum(gammas) / len(gammas),
        "gamma_min": min(gammas),
        "gamma_max": max(gammas),
        "beta_mean": sum(betas) / len(betas),
        "beta_min": min(betas),
        "beta_max": max(betas),
    }


def derf_state_text(state: dict[str, float]) -> str:
    if not state:
        return ""
    return (
        " | derf "
        f"a={state['alpha_mean']:.3g}"
        f"[{state['alpha_min']:.3g},{state['alpha_max']:.3g}],"
        f"s={state['shift_mean']:.3g}"
        f"[{state['shift_min']:.3g},{state['shift_max']:.3g}],"
        f"g={state['gamma_mean']:.3g}"
        f"[{state['gamma_min']:.3g},{state['gamma_max']:.3g}],"
        f"b={state['beta_mean']:.3g}"
        f"[{state['beta_min']:.3g},{state['beta_max']:.3g}]"
    )


@torch.no_grad()
def rmsnorm_affine_state(model: GPT) -> dict[str, float]:
    values = []
    final_values = []
    block_values = []
    for name, module in model.named_modules():
        if not isinstance(module, RMSNorm) or module.gamma is None:
            continue
        gamma = module.gamma.detach().float().flatten()
        values.append(gamma)
        if name == "norm_f":
            final_values.append(gamma)
        else:
            block_values.append(gamma)
    if not values:
        return {}

    def tensor_stats(prefix: str, parts: list[torch.Tensor]) -> dict[str, float]:
        if not parts:
            return {}
        x = torch.cat(parts)
        return {
            f"{prefix}_mean": float(x.mean()),
            f"{prefix}_rms": float(x.square().mean().sqrt()),
            f"{prefix}_min": float(x.min()),
            f"{prefix}_max": float(x.max()),
        }

    return {
        **tensor_stats("gamma", values),
        **tensor_stats("block_gamma", block_values),
        **tensor_stats("final_gamma", final_values),
    }


def rmsnorm_affine_state_text(state: dict[str, float]) -> str:
    if not state:
        return ""
    final = ""
    if "final_gamma_rms" in state:
        final = f",final_rms={state['final_gamma_rms']:.3g}"
    return (
        " | norm_affine "
        f"g_mean={state['gamma_mean']:.3g},g_rms={state['gamma_rms']:.3g},"
        f"g=[{state['gamma_min']:.3g},{state['gamma_max']:.3g}]"
        f"{final}"
    )


def resolve_training_schedule(args) -> tuple[int, int, int]:
    warmup_steps = (
        args.warmup_iters
        if args.warmup_iters >= 0
        else round(args.warmup_frac * args.max_iters)
    )
    decay_steps = (
        args.decay_iters
        if args.decay_iters >= 0
        else round(args.decay_frac * args.max_iters)
    )
    return resolve_schedule(args.max_iters, warmup_steps, decay_steps)


def line_probe_active(args, step: int) -> bool:
    return (
        args.track_line_probe
        and args.grad_accum == 1
        and args.line_probe_interval > 0
        and step % args.line_probe_interval == 0
    )


def run_line_probe(
    model,
    step: int,
    batch,
    rng_before,
    loss_before: float | None,
    params_before,
    curve_scales: list[float],
    line_stats: dict[str, dict],
    amp_dtype: torch.dtype | None,
    device: torch.device,
) -> dict:
    if batch is None or rng_before is None or loss_before is None:
        return {}

    rng_after = capture_rng(device)
    curve_losses = []
    curve_values = {}
    if params_before is None:
        restore_rng(rng_before, device)
        with torch.no_grad(), amp_ctx(amp_dtype):
            _, loss_after = model(*batch)
        restore_rng(rng_after, device)
        loss_after_value = float(loss_after.detach())
    else:
        snapshot = finish_line_snapshot(params_before)
        for scale in curve_scales:
            apply_line_scale(snapshot, scale)
            restore_rng(rng_before, device)
            with torch.no_grad(), amp_ctx(amp_dtype):
                _, curve_loss = model(*batch)
            curve_losses.append((scale, float(curve_loss.detach())))
        apply_line_scale(snapshot, 1.0)
        restore_rng(rng_after, device)
        loss_after_value = min(curve_losses, key=lambda item: abs(item[0] - 1.0))[1]
        curve_values = line_curve_stats(curve_losses)
        curve_text = line_curve_text(step, curve_losses)
        if curve_text:
            print(curve_text)

    probe_values = line_probe_stats(loss_before, loss_after_value, line_stats)
    line_text = line_probe_text(step, loss_before, loss_after_value, line_stats)
    if line_text:
        print(line_text)
    return {
        "loss_before": loss_before,
        "loss_after": loss_after_value,
        "probe": probe_values,
        "curve": curve_values,
        "curve_losses": [
            {"scale": scale, "loss": loss} for scale, loss in curve_losses
        ],
        "step_stats": line_stats,
    }


def train(args):
    args.hyperball_update = resolve_hyperball_update(args)
    device, amp_dtype = configure_runtime(args)
    metrics = MetricsLogger(args.metrics_jsonl, args.run_name)
    line_curve_scales = parse_line_scales(args.line_curve_scales)
    if line_curve_scales:
        args.track_line_probe = True

    dataset = load_dataset(args)
    raw_model = build_model(args, dataset, device)
    configure_derf_training(raw_model, args)
    source = BatchSource(
        dataset.train,
        dataset.val,
        args.block_size,
        args.batch_size,
        device,
        train_seed=resolve_data_seed(args),
        val_seed=resolve_eval_seed(args),
    )
    eval_batches = fixed_eval_batches(args, source)
    opt = build_optimizer(raw_model, args, device)
    deepnorm_calibration = {}
    if args.deepnorm_calibrate_branches:
        idx, _targets = source.seeded_batch("train", resolve_data_seed(args))
        deepnorm_calibration = calibrate_deepnorm_branches(raw_model, idx)
        text = deepnorm_calibration_text(deepnorm_calibration)
        if text:
            print(text)
    derf_opts = build_derf_optimizers(raw_model, args)
    kv_decoder_opt = build_kv_decoder_optimizer(raw_model, args)
    conv_probe = (
        ConvergenceProbe(raw_model, opt, args) if args.track_convergence_stats else None
    )
    if conv_probe is not None:
        conv_probe.register_hooks(raw_model)
        if args.compile:
            print("compile_disabled_for_convergence_stats")
            args.compile = False
    model, compile_seconds = maybe_compile(raw_model, source, args, amp_dtype, device)
    if compile_seconds:
        print(f"compile_seconds {compile_seconds:.3f}")

    warmup_steps, stable_steps, decay_steps = resolve_training_schedule(args)
    effective_tokens = count_increment(args)
    first_group = opt.param_groups[0]
    lr_peak = float(first_group.get("lr_peak", first_group["lr"]))
    beta = float(first_group.get("beta", math.nan))
    derf_beta = next(iter(derf_opts.values())).beta if derf_opts else math.nan
    kv_decoder_beta = kv_decoder_opt.beta if kv_decoder_opt is not None else math.nan
    state_half_life = first_group.get("state_half_life", math.nan)
    io_weights = optimizer_io_label(raw_model)
    params_info = parameter_summary(raw_model)
    kv_info = kv_cache_summary(raw_model)
    deepnorm_alpha = raw_model.cfg.resolved_deepnorm_alpha

    group_text = format_optimizer_schedule(opt)
    print(
        "schedule "
        f"warmup_steps={warmup_steps} stable_steps={stable_steps} decay_steps={decay_steps} "
        f"count_increment={effective_tokens} "
        f"lr_peak={lr_peak:.6f} "
        f"state_half_life={state_half_life:.3g} beta={beta:.6f} "
        f"optimizer={OPTIMIZER_NAME} update={args.hyperball_update} "
        f"norm={args.norm_type} derf_lr={args.derf_lr:.6f} "
        f"derf_beta={derf_beta:.6f} derf_shape={args.train_derf_shape} "
        f"kv_cache={kv_info['type']} kv_dim={kv_info['cache_dim']}/"
        f"{kv_info['original_cache_dim']} kv_ratio={kv_info['cache_ratio']:.3f} "
        f"kv_key_rank={kv_info['key_rank']} kv_value_rank={kv_info['value_rank']} "
        f"kv_decoder_lr={args.kv_decoder_lr:.6f} "
        f"kv_decoder_beta={kv_decoder_beta:.6f} "
        f"params={params_info['total']} trainable_params={params_info['trainable']} "
        f"dropout={args.dropout:.3f} "
        f"resid_scale={args.resid_scale:.6f} "
        f"block_type={args.block_type} "
        f"deepnorm_alpha={deepnorm_alpha:.6g} "
        f"deepnorm_branch_scale={args.deepnorm_branch_scale:.6g} "
        f"deepnorm_calibrate={args.deepnorm_calibrate_branches} "
        f"lns={args.lns} "
        f"attn={args.attn_type} "
        f"seed={args.seed} data_seed={resolve_data_seed(args)} "
        f"eval_seed={resolve_eval_seed(args)} fixed_eval={args.fixed_eval_batches} "
        f"hidden_ulmo={args.hidden_ulmo} "
        f"io_weights={io_weights} embed_ulmo={args.embed_ulmo} out_ulmo={args.out_ulmo} "
        f"qkv=fused spi_iteration={args.spi_iteration} "
        f"groups={group_text}"
    )
    metrics.write(
        "config",
        args=vars(args),
        schedule={
            "warmup_steps": warmup_steps,
            "stable_steps": stable_steps,
            "decay_steps": decay_steps,
            "count_increment": effective_tokens,
            "seed": args.seed,
            "data_seed": resolve_data_seed(args),
            "eval_seed": resolve_eval_seed(args),
            "compile_seed": resolve_compile_seed(args),
            "fixed_eval_batches": args.fixed_eval_batches,
        },
        optimizer={
            "name": OPTIMIZER_NAME,
            "lr_peak": lr_peak,
            "state_half_life": state_half_life,
            "beta": beta,
            "derf_lr_peak": args.derf_lr,
            "derf_state_half_life": args.derf_state_half_life,
            "derf_beta": derf_beta,
            "derf_groups": list(derf_opts),
            "kv_decoder_lr_peak": args.kv_decoder_lr,
            "kv_decoder_beta": kv_decoder_beta,
            "update_rule": args.hyperball_update,
            "groups": group_text,
        },
        model={
            "params": params_info,
            "dropout": args.dropout,
            "resid_scale": args.resid_scale,
            "block_type": args.block_type,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_branch_scale": args.deepnorm_branch_scale,
            "deepnorm_calibrate_branches": args.deepnorm_calibrate_branches,
            "deepnorm_calibration": deepnorm_calibration,
            "lns": args.lns,
            "norm_type": args.norm_type,
            "derf_alpha": args.derf_alpha,
            "derf_shift": args.derf_shift,
            "train_derf_shape": args.train_derf_shape,
            "attn_type": args.attn_type,
            "kv_cache": kv_info,
            "hidden_ulmo": args.hidden_ulmo,
            "io_weights": io_weights,
            "embed_ulmo": args.embed_ulmo,
            "out_ulmo": args.out_ulmo,
            "qkv": "fused",
            "spi_iteration": args.spi_iteration,
        },
    )
    if args.track_line_probe and args.grad_accum != 1:
        print("line_probe_disabled_requires_grad_accum_1")
    if line_curve_scales:
        print("line_curve_scales " + ",".join(f"{x:g}" for x in line_curve_scales))

    total_opt_steps = 0
    best_val = float("inf")
    max_val = float("-inf")
    last_train_loss = float("nan")
    last_val_loss = float("nan")
    initial_val = None
    diverged = False
    diverge_reason = ""
    train_start = sync_now(device)
    eval_seconds = 0.0
    step_stat_accum = {}

    for step in range(args.max_iters):
        current_lrs = apply_scheduled_lr(
            opt,
            step,
            args.max_iters,
            warmup_steps,
            decay_steps,
            args.schedule_floor,
        )
        lr = current_lrs.get("hidden", next(iter(current_lrs.values())))
        derf_lr = 0.0
        if derf_opts:
            derf_lr = schedule_at_step(
                step,
                args.max_iters,
                args.derf_lr,
                args.schedule_floor * args.derf_lr,
                warmup_steps,
                decay_steps,
            )
            for derf_group_opt in derf_opts.values():
                derf_group_opt.lr = derf_lr
        kv_decoder_lr = 0.0
        if kv_decoder_opt is not None:
            kv_decoder_lr = schedule_at_step(
                step,
                args.max_iters,
                kv_decoder_opt.lr_peak,
                args.schedule_floor * kv_decoder_opt.lr_peak,
                warmup_steps,
                decay_steps,
            )
            kv_decoder_opt.lr = kv_decoder_lr

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            eval_start = sync_now(device)
            train_loss = float(last_train_loss)
            val_loss, logit_stats = estimate_val_metrics(
                model,
                source,
                args.eval_iters,
                amp_dtype,
                args.track_logit_stats,
                eval_batches["val"] if eval_batches is not None else None,
            )
            last_val_loss = val_loss
            opt_stats = (
                consume_step_stats(step_stat_accum) if args.track_step_stats else {}
            )
            weight_rms = optimizer_rms_state(opt)
            derf_stats = derf_state(raw_model)
            norm_affine_stats = rmsnorm_affine_state(raw_model)
            kv_decoder_stats = kv_decoder_state(raw_model)

            if not math.isfinite(val_loss):
                diverged, diverge_reason = True, "nonfinite_eval_loss"
            else:
                prev_best = best_val
                if initial_val is None:
                    initial_val = val_loss
                best_val = min(best_val, val_loss)
                max_val = max(max_val, val_loss)
                if step > 0 and val_loss > initial_val * args.diverge_mult:
                    diverged = True
                    diverge_reason = (
                        f"val_loss_exceeded_{args.diverge_mult:.2f}x_initial"
                    )
                if not args.no_save and val_loss < prev_best:
                    save_checkpoint(Path(args.out_path), raw_model, dataset)
                if not args.no_save:
                    save_eval_checkpoint(
                        Path(args.out_path), step, val_loss, raw_model, dataset, args
                    )
                    if step == args.max_iters - 1:
                        path = Path(args.out_path)
                        save_checkpoint(
                            path.with_name(f"{path.stem}_final{path.suffix}"),
                            raw_model,
                            dataset,
                        )

            now = sync_now(device)
            eval_seconds += now - eval_start
            elapsed = max(now - train_start, 1e-9)
            train_elapsed = max(elapsed - eval_seconds, 1e-9)
            total_tokens = total_opt_steps * effective_tokens
            mem_text = cuda_memory_text(device)
            opt_text = step_stats_text(opt_stats)
            rms_text = rms_state_text(weight_rms)
            derf_text = derf_state_text(derf_stats)
            norm_affine_text = rmsnorm_affine_state_text(norm_affine_stats)
            kv_decoder_text = kv_decoder_state_text(kv_decoder_stats)
            logit_text = (
                " | logits "
                f"std={logit_stats['logit_std']:.3f},"
                f"H={logit_stats['softmax_entropy']:.3f},"
                f"pmax={logit_stats['softmax_max_prob']:.3f}"
                if logit_stats
                else ""
            )
            print(
                f"step {step:5d} | lr {lr:.6f} beta {beta:.6f} "
                f"derf_lr {derf_lr:.6f} kv_decoder_lr {kv_decoder_lr:.6f} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"best_val {best_val:.4f} | train_seconds {elapsed:.3f} | "
                f"tok/s {total_tokens / elapsed:.0f} "
                f"train_tok/s {total_tokens / train_elapsed:.0f}"
                f"{mem_text}{logit_text}{rms_text}{derf_text}{norm_affine_text}"
                f"{kv_decoder_text}{opt_text}"
            )
            memory = cuda_memory_stats(device)
            metrics.write(
                "eval",
                step=step,
                total_opt_steps=total_opt_steps,
                lr=lr,
                derf_lr=derf_lr,
                kv_decoder_lr=kv_decoder_lr,
                beta=beta,
                lrs=current_lrs,
                train_loss=train_loss,
                val_loss=val_loss,
                best_val=best_val,
                max_val=max_val,
                train_seconds=elapsed,
                train_compute_seconds=train_elapsed,
                eval_seconds=eval_seconds,
                tokens_per_second=total_tokens / elapsed,
                train_tokens_per_second=total_tokens / train_elapsed,
                cuda_memory=memory,
                logit_stats=logit_stats,
                weight_rms=weight_rms,
                derf=derf_stats,
                norm_affine=norm_affine_stats,
                kv_decoder=kv_decoder_stats,
                step_stats=opt_stats,
            )
            if (
                args.max_cuda_reserved_gb > 0
                and memory
                and memory["reserved_gb"] > args.max_cuda_reserved_gb
            ):
                diverged = True
                diverge_reason = (
                    f"cuda_reserved_{memory['reserved_gb']:.2f}G_exceeded_"
                    f"{args.max_cuda_reserved_gb:.2f}G"
                )
        if diverged:
            print(f"diverged {diverge_reason}")
            break

        line_active = line_probe_active(args, step)
        line_batch = None
        line_rng_before = None
        line_loss_before = None
        opt.zero_grad(set_to_none=True)
        if derf_opts:
            zero_derf_optimizers(derf_opts, set_to_none=True)
        if kv_decoder_opt is not None:
            kv_decoder_opt.zero_grad(set_to_none=True)
        if conv_probe is not None:
            conv_probe.start_step(step)
        train_loss = None
        for micro_step in range(args.grad_accum):
            batch = source.get("train")
            if line_active and micro_step == 0:
                line_batch = batch
                line_rng_before = capture_rng(device)
            with amp_ctx(amp_dtype):
                _, loss = model(*batch)
                loss = loss / args.grad_accum
            loss_value = loss.detach()
            if line_active and micro_step == 0:
                line_loss_before = float(loss_value)
            train_loss = loss_value if train_loss is None else train_loss + loss_value
            loss.backward()
        if diverged:
            print(f"diverged {diverge_reason}")
            break
        last_train_loss = train_loss if train_loss is not None else float("nan")

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        if conv_probe is not None:
            conv_text = conv_probe.capture(step, current_lrs)
            if conv_text:
                print(conv_text)
                metrics.write(
                    "convergence",
                    step=step,
                    total_opt_steps=total_opt_steps,
                    lrs=current_lrs,
                    groups=conv_probe.summary,
                )
        line_params_before = (
            capture_params(raw_model.parameters())
            if line_active and line_curve_scales
            else None
        )
        stat_snapshot = (
            capture_step_stats(opt) if args.track_step_stats or line_active else None
        )
        opt.step()
        if derf_opts:
            step_derf_optimizers(derf_opts)
        if kv_decoder_opt is not None:
            kv_decoder_opt.step()
        line_stats = {}
        if stat_snapshot is not None:
            if args.track_step_stats:
                accumulate_step_stats(step_stat_accum, stat_snapshot)
            if line_active:
                line_stat_accum = {}
                accumulate_step_stats(line_stat_accum, stat_snapshot)
                line_stats = consume_step_stats(line_stat_accum)
        total_opt_steps = step + 1
        if line_active:
            line_record = run_line_probe(
                model,
                step,
                line_batch,
                line_rng_before,
                line_loss_before,
                line_params_before,
                line_curve_scales,
                line_stats,
                amp_dtype,
                device,
            )
            if line_record:
                metrics.write(
                    "line_probe",
                    step=step,
                    total_opt_steps=total_opt_steps,
                    **line_record,
                )

    if not (args.skip_sample or diverged):
        prompt = args.prompt or "\n"
        x = torch.tensor([dataset.encode(prompt)], dtype=torch.long, device=device)
        texts = generate_texts(
            raw_model,
            x,
            dataset.decode,
            args.sample_count,
            args.sample_tokens,
            args.temperature,
            args.top_k,
        )
        if not write_sample_report(args, texts):
            print_samples(texts)

    for group in opt.param_groups:
        stats = getattr(group.get("ulmo"), "stats", None)
        if stats:
            print(
                f"{group.get('name', 'group')}_ulmo_stats "
                + " ".join(f"{k}={v}" for k, v in stats.items())
            )

    last_train_loss = float(last_train_loss)
    result = {
        "best_val": best_val,
        "final_train": last_train_loss,
        "final_val": last_val_loss,
        "compile_seconds": compile_seconds,
        "initial_val": float("nan") if initial_val is None else initial_val,
        "max_val": max_val,
        "diverged": diverged,
        "diverge_reason": diverge_reason,
        "warmup_steps": warmup_steps,
        "stable_steps": stable_steps,
        "decay_steps": decay_steps,
    }
    metrics.write("final", **result)
    metrics.close()
    return result


@torch.inference_mode()
def sample(args):
    device, _ = configure_runtime(args)
    model, stoi, itos = load_checkpoint(Path(args.out_path), device)
    prompt = args.prompt or "\n"
    bad = [c for c in prompt if c not in stoi]
    if bad:
        raise ValueError(f"prompt contains unseen chars: {bad}")
    x = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    texts = generate_texts(
        model,
        x,
        lambda ids: "".join(itos[int(i)] for i in ids),
        args.sample_count,
        args.sample_tokens,
        args.temperature,
        args.top_k,
    )

    if write_sample_report(args, texts):
        return

    print_samples(texts)


def generate_texts(
    model,
    x: torch.Tensor,
    decode,
    sample_count: int,
    sample_tokens: int,
    temperature: float,
    top_k: int,
) -> list[str]:
    texts = []
    for _ in range(sample_count):
        y = model.generate(
            x,
            max_new_tokens=sample_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        texts.append(decode(y[0].tolist()))
    return texts


def write_sample_report(args, texts: list[str]) -> bool:
    if not args.sample_out:
        return False
    path = Path(args.sample_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sample_report(args, texts), encoding="utf-8")
    print(f"wrote_samples {path}")
    return True


def print_samples(texts: list[str]) -> None:
    for i, text in enumerate(texts, start=1):
        if len(texts) > 1:
            print(f"\n--- sample {i} ---\n")
        elif i == 1:
            print("\n--- sample ---\n")
        print(text)


def sample_report(args, texts: list[str]) -> str:
    prompt = args.prompt or "\\n"
    lines = [
        "# Sample Report",
        "",
        f"- checkpoint: `{args.out_path}`",
        f"- seed: `{args.seed}`",
        f"- prompt: `{prompt}`",
        f"- sample_tokens: `{args.sample_tokens}`",
        f"- temperature: `{args.temperature}`",
        f"- top_k: `{args.top_k}`",
        f"- sample_count: `{len(texts)}`",
        "",
    ]
    for i, text in enumerate(texts, start=1):
        lines.extend([f"## Sample {i}", "", "```text", text, "```", ""])
    return "\n".join(lines)


@torch.inference_mode()
def evaluate(args):
    device, amp_dtype = configure_runtime(args)
    data_path = Path(args.data_path)
    maybe_download_tiny_shakespeare(data_path)
    dataset = CharDataset(data_path)
    model, _, _ = load_checkpoint(Path(args.out_path), device)
    source = BatchSource(
        dataset.train,
        dataset.val,
        model.cfg.block_size,
        args.batch_size,
        device,
        train_seed=resolve_data_seed(args),
        val_seed=resolve_eval_seed(args),
    )
    eval_batches = fixed_eval_batches(args, source)
    losses = estimate_loss(model, source, args.eval_iters, amp_dtype, fixed_batches=eval_batches)
    print(
        f"eval_iters {args.eval_iters} | batch_size {args.batch_size} | "
        f"fixed_eval {args.fixed_eval_batches} | eval_seed {resolve_eval_seed(args)} | "
        f"train {losses['train']:.4f} | val {losses['val']:.4f}"
        f"{cuda_memory_text(device)}"
    )


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "sample", "eval"], default="train")
    p.add_argument("--data-path", default="data/tiny_shakespeare.txt")
    p.add_argument("--out-path", default="out/hyperball_shakespeare.pt")
    p.add_argument("--device", default="")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="seed for the independent training-batch RNG; defaults to --seed",
    )
    p.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="seed for fixed evaluation batches; defaults to --seed + 1",
    )
    p.add_argument(
        "--compile-seed",
        type=int,
        default=None,
        help="seed for the compile warmup batch; defaults to --seed + 2",
    )
    p.add_argument(
        "--fixed-eval-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse the same sampled train/val eval batches at every evaluation",
    )
    p.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="request deterministic kernels and disable TF32/benchmark autotuning",
    )
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64, help="microbatch size")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--rope-base", type=float, default=10000.0)
    p.add_argument(
        "--resid-scale",
        type=float,
        default=1.0,
        help="Pre-LN residual branch multiplier; DeepNorm uses its own residual alpha",
    )
    p.add_argument(
        "--block-type",
        choices=["preln", "deepnorm"],
        default="preln",
        help="Transformer block topology",
    )
    p.add_argument(
        "--deepnorm-alpha",
        type=float,
        default=0.0,
        help="DeepNorm residual multiplier; <=0 uses decoder-only default (2*n_layer)^(1/4)",
    )
    p.add_argument(
        "--deepnorm-branch-scale",
        type=float,
        default=1.0,
        help="fixed scalar multiplier on DeepNorm attention/MLP residual branches",
    )
    p.add_argument(
        "--deepnorm-calibrate-branches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "set fixed DeepNorm branch scales at init so branch/(alpha*x) "
            "matches 1/sqrt(2*n_layer)"
        ),
    )
    p.add_argument(
        "--lns",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply LayerNorm Scaling in Pre-LN blocks by scaling norm outputs by 1/sqrt(layer)",
    )
    p.add_argument(
        "--kv-cache",
        choices=["full", "equivariant-lowrank"],
        default="full",
        help="KV-cache architecture used by attention",
    )
    p.add_argument(
        "--kv-key-rank",
        type=int,
        default=3,
        help="complex head-mixing rank per RoPE frequency for low-rank KV",
    )
    p.add_argument(
        "--kv-value-rank",
        type=int,
        default=192,
        help="shared real value-cache rank for low-rank KV",
    )
    p.add_argument(
        "--kv-decoder-lr",
        type=float,
        default=0.001,
        help="peak normalized-SGD learning rate for low-rank KV decoder tensors",
    )
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument(
        "--attn-type",
        choices=["softmax", "linear", "erf"],
        default="softmax",
        help=(
            "attention kernel; linear is a normalized ELU+1 feature-kernel "
            "reference, erf uses normalized 1+erf(score) weights"
        ),
    )
    p.add_argument(
        "--norm-type",
        choices=["rmsnorm", "rmsnorm-affine", "derf"],
        default="rmsnorm",
        help="activation transform used at pre-attn, pre-MLP, and final norm sites",
    )
    p.add_argument(
        "--derf-alpha",
        type=float,
        default=0.5,
        help="Derf input scale alpha initialization",
    )
    p.add_argument(
        "--derf-shift",
        type=float,
        default=0.0,
        help="Derf horizontal shift initialization",
    )
    p.add_argument(
        "--derf-lr",
        type=float,
        default=0.001,
        help="peak normalized-SGD learning rate for Derf shape and small norm affine groups",
    )
    p.add_argument(
        "--derf-state-half-life",
        type=float,
        default=DEFAULT_STATE_HALF_LIFE,
        help="momentum half-life for Derf normalized-SGD updates",
    )
    p.add_argument(
        "--train-derf-shape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train Derf alpha/shift; gamma/beta remain trainable when Derf is active",
    )
    p.add_argument(
        "--tie-weights",
        action="store_true",
        help="share input embedding and output head weights",
    )

    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--diverge-mult", type=float, default=2.0)

    p.add_argument(
        "--warmup-iters", type=int, default=100, help="if >=0, overrides warmup-frac"
    )
    p.add_argument("--warmup-frac", type=float, default=0.0)
    p.add_argument(
        "--decay-iters", type=int, default=-1, help="if >=0, overrides decay-frac"
    )
    p.add_argument("--decay-frac", type=float, default=0.15)

    p.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=(
            "optimizer step size for all groups; retract uses it before "
            "retraction, slerp uses it as an angular step"
        ),
    )
    for group in GROUP_NAMES:
        p.add_argument(f"--lr-{group}", type=float, default=None)
    p.add_argument(
        "--state-half-life",
        type=float,
        default=DEFAULT_STATE_HALF_LIFE,
        help="momentum-state retention half-life in processed tokens",
    )
    for group in GROUP_NAMES:
        p.add_argument(f"--state-half-life-{group}", type=float, default=None)
    p.add_argument(
        "--schedule-floor",
        type=float,
        default=0.0,
        help="WSD schedule floor for the learning-rate ratio (0 = no movement at decay end)",
    )
    p.add_argument(
        "--hyperball-update",
        choices=["slerp", "retract"],
        default=DEFAULT_HYPERBALL_UPDATE,
        help="fixed-radius update rule; slerp is the tangent geodesic variant",
    )
    p.add_argument(
        "--target-rms",
        dest="target_rms",
        type=float,
        default=None,
        help=(
            "initialization RMS target for all optimizer groups; "
            "defaults are embed=0.70, hidden=0.051, out=0.022"
        ),
    )
    for group in GROUP_NAMES:
        p.add_argument(f"--target-rms-{group}", type=float, default=None)
    p.add_argument(
        "--out-rms-rule",
        choices=["fixed", "fan-in"],
        default="fixed",
        help=(
            "default output-head radius when --target-rms-out is omitted; "
            "fan-in uses 1/sqrt(d_model)"
        ),
    )
    p.add_argument(
        "--hidden-ulmo",
        choices=["streaming-svd", "gram-ns"],
        default="gram-ns",
        help="hidden-matrix ULMO",
    )
    p.add_argument(
        "--embed-ulmo",
        choices=["colnorm", "sign", "rownorm"],
        default="colnorm",
        help="embedding-table ULMO; tied weights force Sign",
    )
    p.add_argument(
        "--out-ulmo",
        choices=["sign", "colnorm", "rownorm"],
        default="sign",
        help="output-head ULMO",
    )
    p.add_argument("--pe-steps", type=int, default=5, help="Gram-NS coefficient steps")
    p.add_argument("--spi-steps", type=int, default=1)
    p.add_argument("--spi-ridge", type=float, default=1e-3)
    p.add_argument(
        "--spi-iteration",
        choices=["scqr2", "norm-power"],
        default="norm-power",
        help="streaming-SVD subspace iteration path",
    )
    p.add_argument("--spi-refresh-interval", type=int, default=100)
    p.add_argument("--spi-refresh-threshold", type=float, default=0.10)

    p.add_argument("--prompt", default="To be, or not to be")
    p.add_argument("--sample-tokens", type=int, default=400)
    p.add_argument("--sample-count", type=int, default=1)
    p.add_argument("--sample-out", default="")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--skip-sample", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument(
        "--save-interval",
        type=int,
        default=400,
        help="save eval checkpoints every N steps in addition to best/final",
    )
    p.add_argument(
        "--max-cuda-reserved-gb",
        type=float,
        default=0.0,
        help="abort if process CUDA reserved memory exceeds this limit",
    )
    p.add_argument(
        "--track-step-stats",
        action="store_true",
        help="accumulate optimizer group stats and print them on eval lines",
    )
    p.add_argument(
        "--track-logit-stats",
        action="store_true",
        help="log cheap validation-batch softmax/logit statistics on eval lines",
    )
    p.add_argument(
        "--metrics-jsonl",
        default="",
        help="append structured config/eval/convergence/final records to this JSONL path",
    )
    p.add_argument(
        "--run-name",
        default="",
        help="optional run label included in structured metrics records",
    )
    p.add_argument(
        "--track-convergence-stats",
        action="store_true",
        help="probe smoothness and spectral-ratio stats during training",
    )
    p.add_argument(
        "--track-line-probe",
        action="store_true",
        help="estimate same-batch learning-rate aggressiveness with one extra forward",
    )
    p.add_argument(
        "--line-probe-interval",
        type=int,
        default=100,
        help="optimizer-step interval for same-batch line probes",
    )
    p.add_argument(
        "--line-curve-scales",
        default="",
        help="comma-separated update multipliers for expensive same-batch line curves",
    )
    p.add_argument(
        "--convergence-interval",
        type=int,
        default=50,
        help="optimizer-step interval for convergence probes",
    )
    p.add_argument(
        "--convergence-action-scale",
        dest="convergence_action_scale",
        type=float,
        default=0.5,
        help="target normalized action scale for L1-derived learning-rate reports",
    )
    p.add_argument(
        "--convergence-probe",
        choices=["representative", "all"],
        default="representative",
        help="which parameters to include in convergence probes",
    )
    p.add_argument(
        "--convergence-support-steps",
        type=int,
        default=7,
        help="Gram-NS polar-support steps for spectral dual stats",
    )

    return p


def main():
    args = make_parser().parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "sample":
        sample(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
