import math
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.eps)


class Derf(nn.Module):
    def __init__(self, d_model: int, alpha: float = 0.5, shift: float = 0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.shift = nn.Parameter(torch.tensor(float(shift)))
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * torch.erf(self.alpha * x + self.shift) + self.beta


def make_norm(
    kind: str, d_model: int, derf_alpha: float = 0.5, derf_shift: float = 0.0
) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm()
    if kind == "derf":
        return Derf(d_model, derf_alpha, derf_shift)
    raise ValueError(f"unsupported norm type: {kind}")


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
        attn_type: str = "softmax",
    ):
        super().__init__()
        if d_model % n_head:
            raise ValueError("d_model must be divisible by n_head")
        if attn_type not in {"softmax", "linear", "erf"}:
            raise ValueError(f"unsupported attention type: {attn_type}")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.attn_type = attn_type
        self.kernel_attn_eps = 1e-6
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        cos, sin = rotary_cache(block_size, self.head_dim, rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        mask = torch.ones(block_size, block_size, dtype=torch.bool).tril()
        self.register_buffer("causal_mask", mask, persistent=False)

    def _normalized_kernel_attention(
        self, weights: torch.Tensor, v: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        seqlen = weights.size(-1)
        mask = self.causal_mask[:seqlen, :seqlen].view(1, 1, seqlen, seqlen)
        weights = weights.masked_fill(~mask, 0.0)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(self.kernel_attn_eps)
        return ((weights @ v.float()) / denom).to(dtype)

    def _linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        dtype = v.dtype
        scale = self.head_dim**-0.25
        q_feat = F.elu((q * scale).float()) + 1.0
        k_feat = F.elu((k * scale).float()) + 1.0
        weights = q_feat @ k_feat.transpose(-2, -1)
        return self._normalized_kernel_attention(weights, v, dtype)

    def _erf_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        dtype = v.dtype
        scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = torch.erf(scores) + 1.0
        return self._normalized_kernel_attention(weights, v, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q = self.q(x).view(bsz, seqlen, self.n_head, self.head_dim)
        k = self.k(x).view(bsz, seqlen, self.n_head, self.head_dim)
        v = self.v(x).view(bsz, seqlen, self.n_head, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos = self.rope_cos[:seqlen].view(1, 1, seqlen, self.head_dim // 2)
        sin = self.rope_sin[:seqlen].view(1, 1, seqlen, self.head_dim // 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.attn_type == "softmax":
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.attn_type == "linear":
            y = self._linear_attention(q, k, v)
        else:
            y = self._erf_attention(q, k, v)
        y = self.proj(y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model))
        return self.dropout(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden_dim, bias=False)
        self.up = nn.Linear(d_model, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        hidden_dim: int,
        block_size: int,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        derf_alpha: float = 0.5,
        derf_shift: float = 0.0,
        attn_type: str = "softmax",
    ):
        super().__init__()
        self.norm1 = make_norm(norm_type, d_model, derf_alpha, derf_shift)
        self.attn = CausalSelfAttention(
            d_model, n_head, block_size, rope_base, dropout, attn_type
        )
        self.norm2 = make_norm(norm_type, d_model, derf_alpha, derf_shift)
        self.mlp = MLP(d_model, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


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
    norm_type: str = "rmsnorm"
    derf_alpha: float = 0.5
    derf_shift: float = 0.0
    attn_type: str = "softmax"

    @property
    def hidden_dim(self) -> int:
        return 64 * math.ceil(((8 * self.d_model) // 3) / 64)


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(
            [
                Block(
                    cfg.d_model,
                    cfg.n_head,
                    cfg.hidden_dim,
                    cfg.block_size,
                    cfg.rope_base,
                    cfg.dropout,
                    cfg.norm_type,
                    cfg.derf_alpha,
                    cfg.derf_shift,
                    cfg.attn_type,
                )
                for _ in range(cfg.n_layer)
            ]
        )
        self.norm_f = make_norm(
            cfg.norm_type, cfg.d_model, cfg.derf_alpha, cfg.derf_shift
        )
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
    ):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device
        self.train = train.to(device, non_blocking=True)
        self.val = val.to(device, non_blocking=True)
        self.offsets = torch.arange(block_size + 1, device=device)

    def get(self, split: str):
        data = self.train if split == "train" else self.val
        ix = torch.randint(
            0, data.numel() - self.block_size, (self.batch_size, 1), device=self.device
        )
        chunk = data[ix + self.offsets]
        return chunk[:, :-1], chunk[:, 1:]


def maybe_download_tiny_shakespeare(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
