import torch

from scionh.models.gpt import Derf, EquivariantLowRankKV, GPT, RMSNorm
from scionh.optim.normalized_sgd import NormalizedSGD
from scionh.optim.parametrization import retention_from_half_life
from scionh.optim.setup import count_increment


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
