import argparse
import importlib.util
from collections import OrderedDict
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT_PATH = SCRIPT_DIR / "train_gpt.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location("train_gpt_local", TRAIN_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_config_spec(spec: str) -> tuple[int, int]:
    parts = spec.split(":")
    if len(parts) == 2:
        model_dim, lora_rank = map(int, parts)
        return model_dim, lora_rank
    raise ValueError(
        f"Invalid config spec {spec!r}. Use MODEL_DIM:LORA_RANK."
    )


def numel_of(state_dict: dict[str, torch.Tensor], *, weights_only: bool = False) -> int:
    if weights_only:
        return sum(tensor.numel() for tensor in state_dict.values() if tensor.ndim >= 2)
    return sum(tensor.numel() for tensor in state_dict.values())


def ordered_subset(
    state_dict: dict[str, torch.Tensor],
    predicate,
) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((key, value) for key, value in state_dict.items() if predicate(key, value))


def make_hparams(
    module,
    *,
    model_dim: int,
    lora_rank: int,
    lora_freeze_a: bool,
):
    h = module.Hyperparameters()
    h.model_dim = model_dim
    h.lora_rank = lora_rank
    h.lora_freeze_a = lora_freeze_a
    h.distributed = False
    h.rank = 0
    h.world_size = 1
    h.local_rank = 0
    h.is_main_process = True
    return h


def build_state_dicts(
    module,
    device: torch.device,
    *,
    model_dim: int,
    lora_rank: int,
    lora_freeze_a: bool,
):
    full_h = make_hparams(
        module,
        model_dim=model_dim,
        lora_rank=0,
        lora_freeze_a=False,
    )
    full_model = module.build_model(full_h, device, lora_rank=0)
    full_state = OrderedDict((key, value.detach().cpu()) for key, value in full_model.state_dict().items())

    result = {
        "full_state": full_state,
        "full_hparams": full_h,
    }

    if lora_rank > 0:
        lora_h = make_hparams(
            module,
            model_dim=model_dim,
            lora_rank=lora_rank,
            lora_freeze_a=lora_freeze_a,
        )
        lora_model = module.build_model(
            lora_h,
            device,
            lora_rank=lora_rank,
            lora_freeze_a=lora_freeze_a,
        )
        lora_raw_state = OrderedDict((key, value.detach().cpu()) for key, value in lora_model.state_dict().items())
        lora_efficient_state = OrderedDict(
            (key, value.detach().cpu()) for key, value in module.build_serializable_state_dict(lora_model).items()
        )
        reconstructible_names = module.get_reconstructible_tensor_names(lora_model)
        reconstructible_state = ordered_subset(
            lora_raw_state,
            lambda key, _: key in reconstructible_names,
        )
        adapters_only_state = ordered_subset(
            lora_raw_state,
            lambda key, _: module.is_train_lora_tensor_name(key),
        )
        other_kept_state = ordered_subset(
            lora_efficient_state,
            lambda key, _: key not in adapters_only_state,
        )
        result.update(
            {
                "lora_raw_state": lora_raw_state,
                "lora_efficient_state": lora_efficient_state,
                "lora_reconstructible_state": reconstructible_state,
                "lora_adapters_state": adapters_only_state,
                "lora_other_kept_state": other_kept_state,
                "lora_hparams": lora_h,
            }
        )

    del full_model
    if "lora_model" in locals():
        del lora_model
    return result


def print_metric_block(
    *,
    title: str,
    full_params: int,
    lora_raw_params: int,
    lora_efficient_params: int,
    lora_adapters_params: int,
    lora_other_kept_params: int,
    lora_reconstructible_params: int,
) -> None:
    print(title)
    print(f"Full model params:               {full_params:,}")
    print(f"LoRA checkpoint params (raw):    {lora_raw_params:,}")
    print(f"LoRA efficient params:           {lora_efficient_params:,}")
    print(f"  - LoRA adapters only:          {lora_adapters_params:,}")
    print(f"  - Other kept params:           {lora_other_kept_params:,}")
    print(f"  - Reconstructible tensors:     {lora_reconstructible_params:,}")
    print()
    print(f"LoRA efficient / full:           {lora_efficient_params / full_params:.2%}")
    print(f"Compression factor vs full:      {full_params / lora_efficient_params:.2f}x")


def summarize_config(
    module,
    device: torch.device,
    *,
    model_dim: int,
    lora_rank: int,
    lora_freeze_a: bool,
    include_weights_only: bool,
    verbose: bool,
) -> dict[str, int | str]:
    states = build_state_dicts(
        module,
        device,
        model_dim=model_dim,
        lora_rank=lora_rank,
        lora_freeze_a=lora_freeze_a,
    )
    full_params = numel_of(states["full_state"])
    full_weight_params = numel_of(states["full_state"], weights_only=True)

    summary = {
        "model_dim": model_dim,
        "lora_rank": lora_rank,
        "lora_freeze_a": lora_freeze_a,
        "full_params": full_params,
        "full_weight_params": full_weight_params,
    }

    if verbose:
        print("=" * 88)
        print(
            f"Config: model_dim={model_dim} num_layers={states['full_hparams'].num_layers} "
            f"lora_rank={lora_rank} "
            f"lora_freeze_a={lora_freeze_a}"
        )

    if lora_rank <= 0:
        if verbose:
            print(f"Full model params:               {full_params:,}")
            print(f"Full model weight params:        {full_weight_params:,}")
            print("LoRA disabled for this config.")
        summary.update(
            {
                "lora_raw_params": full_params,
                "lora_efficient_params": full_params,
                "lora_adapters_params": 0,
                "lora_other_kept_params": full_params,
                "lora_reconstructible_params": 0,
            }
        )
        return summary

    lora_raw_params = numel_of(states["lora_raw_state"])
    lora_efficient_params = numel_of(states["lora_efficient_state"])
    lora_adapters_params = numel_of(states["lora_adapters_state"])
    lora_other_kept_params = numel_of(states["lora_other_kept_state"])
    lora_reconstructible_params = numel_of(states["lora_reconstructible_state"])

    if verbose:
        print_metric_block(
            title="All state tensors:",
            full_params=full_params,
            lora_raw_params=lora_raw_params,
            lora_efficient_params=lora_efficient_params,
            lora_adapters_params=lora_adapters_params,
            lora_other_kept_params=lora_other_kept_params,
            lora_reconstructible_params=lora_reconstructible_params,
        )

    if verbose and include_weights_only:
        print()
        print_metric_block(
            title="Weight tensors only:",
            full_params=full_weight_params,
            lora_raw_params=numel_of(states["lora_raw_state"], weights_only=True),
            lora_efficient_params=numel_of(states["lora_efficient_state"], weights_only=True),
            lora_adapters_params=numel_of(states["lora_adapters_state"], weights_only=True),
            lora_other_kept_params=numel_of(states["lora_other_kept_state"], weights_only=True),
            lora_reconstructible_params=numel_of(states["lora_reconstructible_state"], weights_only=True),
        )

    summary.update(
        {
            "lora_raw_params": lora_raw_params,
            "lora_efficient_params": lora_efficient_params,
            "lora_adapters_params": lora_adapters_params,
            "lora_other_kept_params": lora_other_kept_params,
            "lora_reconstructible_params": lora_reconstructible_params,
        }
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate parameter savings for full vs LoRA checkpoints by instantiating the local "
            "train_gpt.py model with one or more MODEL_DIM / LORA_RANK configs."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        help=(
            "Configuration spec. Use MODEL_DIM:LORA_RANK. "
            "May be provided multiple times."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to build models on. Default: cpu",
    )
    parser.add_argument(
        "--include-weights-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also print the same breakdown for tensors whose key contains 'weight'.",
    )
    parser.add_argument(
        "--summary-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the verbose per-config block and only print the compact summary table.",
    )
    parser.add_argument(
        "--lora-freeze-a",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze LoRA A and treat it as reconstructible from init seed in compact checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = load_train_module()
    configs = args.configs
    if not configs:
        base_h = module.Hyperparameters()
        configs = [f"{base_h.model_dim}:{base_h.lora_rank}"]

    device = torch.device(args.device)
    summaries = []
    for spec in configs:
        model_dim, lora_rank = parse_config_spec(spec)
        summary = summarize_config(
            module,
            device,
            model_dim=model_dim,
            lora_rank=lora_rank,
            lora_freeze_a=args.lora_freeze_a,
            include_weights_only=args.include_weights_only and not args.summary_only,
            verbose=not args.summary_only,
        )
        summaries.append(summary)
        if not args.summary_only:
            print()

    print("Summary")
    print(
        "model_dim lora_rank lora_freeze_a full_params lora_raw_params "
        "lora_efficient_params efficient/full"
    )
    for summary in summaries:
        efficient_ratio = summary["lora_efficient_params"] / summary["full_params"]
        print(
            f"{summary['model_dim']:>8} {summary['lora_rank']:>9} "
            f"{str(summary['lora_freeze_a']):>13} "
            f"{summary['full_params']:>11,} {summary['lora_raw_params']:>15,} "
            f"{summary['lora_efficient_params']:>20,} {efficient_ratio:>14.2%}"
        )


if __name__ == "__main__":
    main()
