from scionh.models.gpt import GPT


def parameter_summary(model: GPT) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
