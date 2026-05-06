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
    def __init__(
        self,
        d_model: int | None = None,
        eps: float = 1e-6,
        affine: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (d_model,) if d_model is not None else None
        if affine:
            if d_model is None:
                raise ValueError("affine RMSNorm requires d_model")
            self.gamma = nn.Parameter(torch.ones(d_model))
        else:
            self.register_parameter("gamma", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalized_shape is not None:
            return F.rms_norm(x, self.normalized_shape, self.gamma, self.eps)
        y = rms_norm(x, self.eps)
        if self.gamma is None:
            return y
        return self.gamma * y


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
    kind: str,
    d_model: int,
    derf_alpha: float = 0.5,
    derf_shift: float = 0.0,
) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(d_model)
    if kind == "rmsnorm-affine":
        return RMSNorm(d_model, affine=True)
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


class EquivariantLowRankKV(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        head_dim: int,
        key_rank: int,
        value_rank: int,
    ):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even for RoPE-equivariant KV")
        if key_rank <= 0:
            raise ValueError(f"invalid KV key rank: {key_rank}")
        if key_rank > n_head:
            raise ValueError(
                f"KV key rank {key_rank} cannot exceed n_head={n_head}"
            )
        if value_rank <= 0:
            raise ValueError(f"invalid KV value rank: {value_rank}")
        self.n_head = n_head
        self.head_dim = head_dim
        self.freq_count = head_dim // 2
        self.key_rank = key_rank
        self.value_rank = value_rank
        self.k = nn.Linear(d_model, 2 * self.freq_count * key_rank, bias=False)
        self.v = nn.Linear(d_model, value_rank, bias=False)
        self.key_decoder = nn.Parameter(
            torch.empty(self.freq_count, n_head, key_rank, 2)
        )
        self.value_decoder = nn.Parameter(torch.empty(n_head, value_rank, head_dim))
        self.reset_parameters()

    @property
    def key_latent_dim(self) -> int:
        return 2 * self.freq_count * self.key_rank

    @property
    def cache_dim(self) -> int:
        return self.key_latent_dim + self.value_rank

    @property
    def original_cache_dim(self) -> int:
        return 2 * self.n_head * self.head_dim

    def decoder_parameters(self) -> tuple[nn.Parameter, nn.Parameter]:
        return self.key_decoder, self.value_decoder

    def reset_parameters(self) -> None:
        nn.init.normal_(self.key_decoder, std=1.0 / math.sqrt(2 * self.key_rank))
        nn.init.normal_(self.value_decoder, std=1.0 / math.sqrt(self.value_rank))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = x.shape
        z_key = self.k(x).view(bsz, seqlen, self.freq_count, self.key_rank, 2)
        z_value = self.v(x)
        return z_key, z_value

    def decode_keys(self, z_key: torch.Tensor) -> torch.Tensor:
        basis = self.key_decoder.to(dtype=z_key.dtype)
        bsz, seqlen = z_key.shape[:2]
        z = z_key.permute(2, 0, 1, 3, 4).reshape(
            self.freq_count, bsz * seqlen, 2 * self.key_rank
        )
        basis_real = basis[..., 0].permute(0, 2, 1)
        basis_imag = basis[..., 1].permute(0, 2, 1)
        row_real = torch.stack((basis_real, basis_imag), dim=-1)
        row_imag = torch.stack((-basis_imag, basis_real), dim=-1)
        decoder = torch.stack((row_real, row_imag), dim=2).reshape(
            self.freq_count, 2 * self.key_rank, 2 * self.n_head
        )
        out = torch.bmm(z, decoder).reshape(
            self.freq_count, bsz, seqlen, self.n_head, 2
        )
        return (
            out.permute(1, 2, 3, 0, 4)
            .contiguous()
            .flatten(-2)
        )

    def decode_values(self, context: torch.Tensor) -> torch.Tensor:
        decoder = self.value_decoder.to(dtype=context.dtype)
        return torch.einsum("bhtr,hrd->bhtd", context, decoder)

    def decode_value_tokens(self, z_value: torch.Tensor) -> torch.Tensor:
        decoder = self.value_decoder.permute(0, 2, 1).reshape(
            self.n_head * self.head_dim, self.value_rank
        )
        values = F.linear(z_value, decoder.to(dtype=z_value.dtype))
        return values.view(z_value.size(0), z_value.size(1), self.n_head, self.head_dim)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        block_size: int,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        attn_type: str = "softmax",
        kv_cache: str = "full",
        kv_key_rank: int = 3,
        kv_value_rank: int = 192,
    ):
        super().__init__()
        if d_model % n_head:
            raise ValueError("d_model must be divisible by n_head")
        if attn_type not in {"softmax", "linear", "erf"}:
            raise ValueError(f"unsupported attention type: {attn_type}")
        if kv_cache not in {"full", "equivariant-lowrank"}:
            raise ValueError(f"unsupported KV cache type: {kv_cache}")
        if kv_cache != "full" and attn_type != "softmax":
            raise ValueError(
                "equivariant low-rank KV cache currently expects softmax attention"
            )
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.attn_type = attn_type
        self.kv_cache = kv_cache
        self.kernel_attn_eps = 1e-6
        if kv_cache == "full":
            self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
            self.q = None
            self.k = None
            self.v = None
            self.lowrank_kv = None
        else:
            self.qkv = None
            self.q = nn.Linear(d_model, d_model, bias=False)
            self.k = None
            self.v = None
            self.lowrank_kv = EquivariantLowRankKV(
                d_model, n_head, self.head_dim, kv_key_rank, kv_value_rank
            )
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

    def _forward_lowrank_kv(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if self.lowrank_kv is None:
            raise RuntimeError("low-rank KV path called without a low-rank module")
        bsz, seqlen, d_model = x.shape
        z_key, z_value = self.lowrank_kv.encode(x)
        k = self.lowrank_kv.decode_keys(z_key)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.transpose(1, 2)
        cos = self.rope_cos[:seqlen].view(1, 1, seqlen, self.head_dim // 2)
        sin = self.rope_sin[:seqlen].view(1, 1, seqlen, self.head_dim // 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # Equivalent to decoding after the weighted sum, but keeps SDPA's V
        # dimension at head_dim for the dense training path.
        v = self.lowrank_kv.decode_value_tokens(z_value).transpose(1, 2)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        return self.proj(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        if self.qkv is not None:
            q, k, v = self.qkv(x).split(d_model, dim=-1)
        else:
            if self.q is None:
                raise RuntimeError("low-rank KV path called without q projection")
            q = self.q(x)
        if self.lowrank_kv is not None:
            return self.dropout(self._forward_lowrank_kv(x, q))
        if self.qkv is None:
            raise RuntimeError("full KV path called without fused qkv projection")
        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim)
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
        norm_type: str = "rmsnorm",
        derf_alpha: float = 0.5,
        derf_shift: float = 0.0,
        attn_type: str = "softmax",
        kv_cache: str = "full",
        kv_key_rank: int = 3,
        kv_value_rank: int = 192,
        resid_scale: float = 1.0,
        block_type: str = "preln",
        deepnorm_alpha: float = 1.0,
        deepnorm_branch_scale: float = 1.0,
        lns_scale: float = 1.0,
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
        if lns_scale <= 0:
            raise ValueError("lns_scale must be positive")
        self.resid_scale = resid_scale
        self.block_type = block_type
        self.deepnorm_alpha = deepnorm_alpha
        self.lns_scale = lns_scale
        self.register_buffer(
            "deepnorm_attn_scale",
            torch.tensor(float(deepnorm_branch_scale)),
        )
        self.register_buffer(
            "deepnorm_mlp_scale",
            torch.tensor(float(deepnorm_branch_scale)),
        )
        self.norm1 = (
            None
            if block_type == "deepnorm"
            else make_norm(norm_type, d_model, derf_alpha, derf_shift)
        )
        self.attn = CausalSelfAttention(
            d_model,
            n_head,
            block_size,
            rope_base,
            dropout,
            attn_type,
            kv_cache,
            kv_key_rank,
            kv_value_rank,
        )
        self.norm2 = (
            None
            if block_type == "deepnorm"
            else make_norm(norm_type, d_model, derf_alpha, derf_shift)
        )
        self.mlp = MLP(d_model, hidden_dim, dropout)
        self.post_attn_norm = (
            make_norm(norm_type, d_model, derf_alpha, derf_shift)
            if block_type == "deepnorm"
            else None
        )
        self.post_mlp_norm = (
            make_norm(norm_type, d_model, derf_alpha, derf_shift)
            if block_type == "deepnorm"
            else None
        )

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
            x = x + self.resid_scale * self.attn(self.lns_scale * self.norm1(x))
            return x + self.resid_scale * self.mlp(self.lns_scale * self.norm2(x))

        if self.block_type == "deepnorm":
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
    norm_type: str = "rmsnorm"
    derf_alpha: float = 0.5
    derf_shift: float = 0.0
    attn_type: str = "softmax"
    kv_cache: str = "full"
    kv_key_rank: int = 3
    kv_value_rank: int = 192
    resid_scale: float = 1.0
    block_type: str = "preln"
    deepnorm_alpha: float = 0.0
    deepnorm_branch_scale: float = 1.0
    lns: bool = False

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
                    cfg.norm_type,
                    cfg.derf_alpha,
                    cfg.derf_shift,
                    cfg.attn_type,
                    cfg.kv_cache,
                    cfg.kv_key_rank,
                    cfg.kv_value_rank,
                    cfg.resid_scale,
                    cfg.block_type,
                    deepnorm_alpha,
                    cfg.deepnorm_branch_scale,
                    1.0 / math.sqrt(i + 1) if cfg.lns else 1.0,
                )
                for i in range(cfg.n_layer)
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
