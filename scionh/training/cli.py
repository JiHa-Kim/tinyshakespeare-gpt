import argparse

from scionh.optim.setup import (
    DEFAULT_HYPERBALL_UPDATE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_STATE_HALF_LIFE,
    GROUP_NAMES,
)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_runtime_args(parser)
    _add_model_args(parser)
    _add_training_args(parser)
    _add_optimizer_args(parser)
    _add_sampling_args(parser)
    _add_logging_args(parser)
    _add_probe_args(parser)
    return parser


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=["train", "sample", "eval"], default="train")
    parser.add_argument("--data-path", default="data/tiny_shakespeare.txt")
    parser.add_argument("--out-path", default="out/hyperball_shakespeare.pt")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="seed for the independent training-batch RNG; defaults to --seed",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="seed for fixed evaluation batches; defaults to --seed + 1",
    )
    parser.add_argument(
        "--compile-seed",
        type=int,
        default=None,
        help="seed for the compile warmup batch; defaults to --seed + 2",
    )
    parser.add_argument(
        "--fixed-eval-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reuse the same sampled train/val eval batches at every evaluation",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="request deterministic kernels and disable TF32/benchmark autotuning",
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64, help="microbatch size")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument(
        "--resid-scale",
        type=float,
        default=1.0,
        help="Pre-LN residual branch multiplier; DeepNorm uses its own residual alpha",
    )
    parser.add_argument(
        "--block-type",
        choices=["preln", "deepnorm"],
        default="preln",
        help="Transformer block topology",
    )
    parser.add_argument(
        "--deepnorm-alpha",
        type=float,
        default=0.0,
        help="DeepNorm residual multiplier; <=0 uses decoder-only default (2*n_layer)^(1/4)",
    )
    parser.add_argument(
        "--deepnorm-branch-scale",
        type=float,
        default=1.0,
        help="fixed scalar multiplier on DeepNorm attention/MLP residual branches",
    )
    parser.add_argument(
        "--deepnorm-calibrate-branches",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "set fixed DeepNorm branch scales at init so branch/(alpha*x) "
            "matches 1/sqrt(2*n_layer)"
        ),
    )
    parser.add_argument(
        "--lns",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply LayerNorm Scaling in Pre-LN blocks by scaling norm outputs by 1/sqrt(layer)",
    )
    parser.add_argument(
        "--kv-cache",
        choices=["full", "equivariant-lowrank"],
        default="full",
        help="KV-cache architecture used by attention",
    )
    parser.add_argument(
        "--kv-key-rank",
        type=int,
        default=3,
        help="complex head-mixing rank per RoPE frequency for low-rank KV",
    )
    parser.add_argument(
        "--kv-value-rank",
        type=int,
        default=192,
        help="shared real value-cache rank for low-rank KV",
    )
    parser.add_argument(
        "--kv-decoder-lr",
        type=float,
        default=0.001,
        help="peak normalized-SGD learning rate for low-rank KV decoder tensors",
    )
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument(
        "--attn-type",
        choices=["softmax", "linear", "erf"],
        default="softmax",
        help=(
            "attention kernel; linear is a normalized ELU+1 feature-kernel "
            "reference, erf uses normalized 1+erf(score) weights"
        ),
    )
    parser.add_argument(
        "--norm-type",
        choices=["rmsnorm", "rmsnorm-affine", "derf"],
        default="rmsnorm",
        help="activation transform used at pre-attn, pre-MLP, and final norm sites",
    )
    parser.add_argument(
        "--derf-alpha",
        type=float,
        default=0.5,
        help="Derf input scale alpha initialization",
    )
    parser.add_argument(
        "--derf-shift",
        type=float,
        default=0.0,
        help="Derf horizontal shift initialization",
    )
    parser.add_argument(
        "--derf-lr",
        type=float,
        default=0.001,
        help="peak normalized-SGD learning rate for Derf shape and small norm affine groups",
    )
    parser.add_argument(
        "--derf-state-half-life",
        type=float,
        default=DEFAULT_STATE_HALF_LIFE,
        help="momentum half-life for Derf normalized-SGD updates",
    )
    parser.add_argument(
        "--train-derf-shape",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="train Derf alpha/shift; gamma/beta remain trainable when Derf is active",
    )
    parser.add_argument(
        "--tie-weights",
        action="store_true",
        help="share input embedding and output head weights",
    )


def _add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--diverge-mult", type=float, default=2.0)
    parser.add_argument(
        "--warmup-iters", type=int, default=100, help="if >=0, overrides warmup-frac"
    )
    parser.add_argument("--warmup-frac", type=float, default=0.0)
    parser.add_argument(
        "--decay-iters", type=int, default=-1, help="if >=0, overrides decay-frac"
    )
    parser.add_argument("--decay-frac", type=float, default=0.15)


def _add_optimizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help=(
            "optimizer step size for all groups; retract uses it before "
            "retraction, slerp uses it as an angular step"
        ),
    )
    for group in GROUP_NAMES:
        parser.add_argument(f"--lr-{group}", type=float, default=None)
    parser.add_argument(
        "--state-half-life",
        type=float,
        default=DEFAULT_STATE_HALF_LIFE,
        help="momentum-state retention half-life in processed tokens",
    )
    for group in GROUP_NAMES:
        parser.add_argument(f"--state-half-life-{group}", type=float, default=None)
    parser.add_argument(
        "--schedule-floor",
        type=float,
        default=0.0,
        help="WSD schedule floor for the learning-rate ratio (0 = no movement at decay end)",
    )
    parser.add_argument(
        "--hyperball-update",
        choices=["slerp", "retract"],
        default=DEFAULT_HYPERBALL_UPDATE,
        help="fixed-radius update rule; slerp is the tangent geodesic variant",
    )
    parser.add_argument(
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
        parser.add_argument(f"--target-rms-{group}", type=float, default=None)
    parser.add_argument(
        "--out-rms-rule",
        choices=["fixed", "fan-in"],
        default="fixed",
        help=(
            "default output-head radius when --target-rms-out is omitted; "
            "fan-in uses 1/sqrt(d_model)"
        ),
    )
    parser.add_argument(
        "--hidden-ulmo",
        choices=["streaming-svd", "gram-ns"],
        default="gram-ns",
        help="hidden-matrix ULMO",
    )
    parser.add_argument(
        "--embed-ulmo",
        choices=["colnorm", "sign", "rownorm"],
        default="colnorm",
        help="embedding-table ULMO; tied weights force Sign",
    )
    parser.add_argument(
        "--out-ulmo",
        choices=["sign", "colnorm", "rownorm"],
        default="sign",
        help="output-head ULMO",
    )
    parser.add_argument("--pe-steps", type=int, default=5, help="Gram-NS coefficient steps")
    parser.add_argument("--spi-steps", type=int, default=1)
    parser.add_argument("--spi-ridge", type=float, default=1e-3)
    parser.add_argument(
        "--spi-iteration",
        choices=["scqr2", "norm-power"],
        default="norm-power",
        help="streaming-SVD subspace iteration path",
    )
    parser.add_argument("--spi-refresh-interval", type=int, default=100)
    parser.add_argument("--spi-refresh-threshold", type=float, default=0.10)


def _add_sampling_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", default="To be, or not to be")
    parser.add_argument("--sample-tokens", type=int, default=400)
    parser.add_argument("--sample-count", type=int, default=1)
    parser.add_argument("--sample-out", default="")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--save-interval",
        type=int,
        default=400,
        help="save eval checkpoints every N steps in addition to best/final",
    )


def _add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-cuda-reserved-gb",
        type=float,
        default=0.0,
        help="abort if process CUDA reserved memory exceeds this limit",
    )
    parser.add_argument(
        "--track-step-stats",
        action="store_true",
        help="accumulate optimizer group stats and print them on eval lines",
    )
    parser.add_argument(
        "--track-logit-stats",
        action="store_true",
        help="log cheap validation-batch softmax/logit statistics on eval lines",
    )
    parser.add_argument(
        "--metrics-jsonl",
        default="",
        help="append structured config/eval/convergence/final records to this JSONL path",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="optional run label included in structured metrics records",
    )


def _add_probe_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--track-convergence-stats",
        action="store_true",
        help="probe smoothness and spectral-ratio stats during training",
    )
    parser.add_argument(
        "--track-line-probe",
        action="store_true",
        help="estimate same-batch learning-rate aggressiveness with one extra forward",
    )
    parser.add_argument(
        "--line-probe-interval",
        type=int,
        default=100,
        help="optimizer-step interval for same-batch line probes",
    )
    parser.add_argument(
        "--line-curve-scales",
        default="",
        help="comma-separated update multipliers for expensive same-batch line curves",
    )
    parser.add_argument(
        "--convergence-interval",
        type=int,
        default=50,
        help="optimizer-step interval for convergence probes",
    )
    parser.add_argument(
        "--convergence-action-scale",
        dest="convergence_action_scale",
        type=float,
        default=0.5,
        help="target normalized action scale for L1-derived learning-rate reports",
    )
    parser.add_argument(
        "--convergence-probe",
        choices=["representative", "all"],
        default="representative",
        help="which parameters to include in convergence probes",
    )
    parser.add_argument(
        "--convergence-support-steps",
        type=int,
        default=7,
        help="Gram-NS polar-support steps for spectral dual stats",
    )
