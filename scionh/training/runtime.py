from pathlib import Path

import torch

from scionh.models.gpt import (
    BatchSource,
    CharDataset,
    GPT,
    GPTConfig,
    maybe_download_tiny_shakespeare,
)


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
