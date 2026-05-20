import math
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (d_model,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, None, self.eps)


def rotary_cache(seq_len: int, head_dim: int, base: float = 10000.0):
    if head_dim % 2:
        raise ValueError("head_dim must be even for RoPE")
    half = head_dim // 2
    freq = 1.0 / (base ** (torch.arange(half, dtype=torch.float32) / half))
    freqs = torch.outer(torch.arange(seq_len, dtype=torch.float32), freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(dtype=x.dtype)
    sin = sin.to(dtype=x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        block_size: int,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_head:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        cos, sin = rotary_cache(block_size, self.head_dim, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q, k, v = self.qkv(x).split(d_model, dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[:seqlen].view(1, 1, seqlen, self.head_dim // 2)
        sin = self.rope_sin[:seqlen].view(1, 1, seqlen, self.head_dim // 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        return self.dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.up_gate = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.up_gate(x).chunk(2, dim=-1)
        hidden = F.silu(gate) * up
        return self.dropout(self.down(hidden))


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        hidden_dim: int,
        block_size: int,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        resid_scale: float = 1.0,
        block_type: str = "deepnorm",
        deepnorm_alpha: float = 1.0,
        deepnorm_branch_scale: float = 1.0,
    ):
        super().__init__()
        if resid_scale < 0:
            raise ValueError("resid_scale must be non-negative")
        if block_type not in {"preln", "deepnorm"}:
            raise ValueError(f"unsupported block type: {block_type}")
        if block_type == "deepnorm" and deepnorm_alpha <= 0:
            raise ValueError("deepnorm_alpha must be positive")
        if deepnorm_branch_scale <= 0:
            raise ValueError("deepnorm_branch_scale must be positive")
        self.resid_scale = resid_scale
        self.block_type = block_type
        self.deepnorm_alpha = deepnorm_alpha
        self.register_buffer(
            "deepnorm_attn_scale",
            torch.tensor(float(deepnorm_branch_scale)),
        )
        self.register_buffer(
            "deepnorm_mlp_scale",
            torch.tensor(float(deepnorm_branch_scale)),
        )
        self.norm1 = None if block_type == "deepnorm" else RMSNorm(d_model)
        self.norm2 = None if block_type == "deepnorm" else RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, block_size, rope_base, dropout)
        self.mlp = MLP(d_model, hidden_dim, dropout)
        self.post_attn_norm = RMSNorm(d_model) if block_type == "deepnorm" else None
        self.post_mlp_norm = RMSNorm(d_model) if block_type == "deepnorm" else None

    @torch.no_grad()
    def set_deepnorm_branch_scales(
        self, attn_scale: float, mlp_scale: float
    ) -> None:
        if attn_scale <= 0.0 or mlp_scale <= 0.0:
            raise ValueError("DeepNorm branch scales must be positive")
        self.deepnorm_attn_scale.fill_(float(attn_scale))
        self.deepnorm_mlp_scale.fill_(float(mlp_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_type == "preln":
            if self.norm1 is None or self.norm2 is None:
                raise RuntimeError("Pre-LN norms are missing")
            x = x + self.resid_scale * self.attn(self.norm1(x))
            return x + self.resid_scale * self.mlp(self.norm2(x))

        if self.post_attn_norm is None or self.post_mlp_norm is None:
            raise RuntimeError("DeepNorm post-norms are missing")
        x = self.post_attn_norm(
            self.deepnorm_alpha * x + self.deepnorm_attn_scale * self.attn(x)
        )
        return self.post_mlp_norm(
            self.deepnorm_alpha * x + self.deepnorm_mlp_scale * self.mlp(x)
        )


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_weights: bool = False
    resid_scale: float = 1.0
    block_type: str = "deepnorm"
    deepnorm_alpha: float = 0.0
    deepnorm_branch_scale: float = 1.0

    @property
    def hidden_dim(self) -> int:
        return 64 * math.ceil(((8 * self.d_model) // 3) / 64)

    @property
    def resolved_deepnorm_alpha(self) -> float:
        if self.deepnorm_alpha > 0.0:
            return self.deepnorm_alpha
        return (2.0 * self.n_layer) ** 0.25


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        deepnorm_alpha = cfg.resolved_deepnorm_alpha
        self.blocks = nn.ModuleList(
            [
                Block(
                    cfg.d_model,
                    cfg.n_head,
                    cfg.hidden_dim,
                    cfg.block_size,
                    cfg.rope_base,
                    cfg.dropout,
                    cfg.resid_scale,
                    cfg.block_type,
                    deepnorm_alpha,
                    cfg.deepnorm_branch_scale,
                )
                for _ in range(cfg.n_layer)
            ]
        )
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm_f(x))
        loss = (
            None
            if targets is None
            else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 40,
    ):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size :])
            logits = logits[:, -1]
            if temperature != 1.0:
                logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
            idx = torch.cat(
                (idx, torch.multinomial(F.softmax(logits, dim=-1), 1)), dim=1
            )
        self.train(was_training)
        return idx


class CharDataset:
    def __init__(self, path: Path):
        text = path.read_text(encoding="utf-8")
        self.chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        self.train = data[:n].contiguous()
        self.val = data[n:].contiguous()

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class BatchSource:
    def __init__(
        self,
        train: torch.Tensor,
        val: torch.Tensor,
        block_size: int,
        batch_size: int,
        device: torch.device,
        train_seed: int = 0,
        val_seed: int = 1,
    ):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.train = train.to(device, non_blocking=True)
        self.val = val.to(device, non_blocking=True)
        self.offsets = torch.arange(block_size + 1, device=device)
        self.train_generator = self.make_generator(device, train_seed)
        self.val_generator = self.make_generator(device, val_seed)

    @staticmethod
    def make_generator(device: torch.device, seed: int) -> torch.Generator:
        generator = (
            torch.Generator(device=device)
            if device.type == "cuda"
            else torch.Generator()
        )
        generator.manual_seed(int(seed))
        return generator

    def get(self, split: str, generator: torch.Generator | None = None):
        data = self.train if split == "train" else self.val
        if generator is None:
            generator = self.train_generator if split == "train" else self.val_generator
        ix = torch.randint(
            0,
            data.numel() - self.block_size,
            (self.batch_size, 1),
            device=self.device,
            generator=generator,
        )
        chunk = data[ix + self.offsets]
        return chunk[:, :-1], chunk[:, 1:]

    def seeded_batch(self, split: str, seed: int):
        generator = self.make_generator(self.device, seed)
        return self.get(split, generator)

    def fixed_batches(self, split: str, batches: int, seed: int):
        generator = self.make_generator(self.device, seed)
        return tuple(self.get(split, generator) for _ in range(batches))


def maybe_download_tiny_shakespeare(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
