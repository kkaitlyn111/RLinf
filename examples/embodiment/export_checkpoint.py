#!/usr/bin/env python3
"""Export an RLinf FSDP/DCP training checkpoint to a loadable HuggingFace model.

Combines the two-stage conversion pipeline (DCP -> .pt -> HuggingFace safetensors)
into a single script. Supports LoRA merging.

Usage:
    # From a training config (extracts all model settings automatically):
    python export_checkpoint.py \
        --config config/ppo_openvlaoft_pickcube.yaml \
        --ckpt_dir results/ppo_openvlaoft_pickcube/checkpoints/global_step_40/actor \
        --output_dir ./exported_openvlaoft_pickcube

    # With --cpu for running on login nodes without GPU:
    python export_checkpoint.py \
        --config config/ppo_openvlaoft_pickcube.yaml \
        --ckpt_dir .../actor \
        --output_dir ./exported_model \
        --cpu

    # Without config (manual args, defaults match ppo_openvlaoft_pickcube.yaml):
    python export_checkpoint.py \
        --ckpt_dir .../actor \
        --output_dir ./exported_model \
        --model_type openvla_oft \
        --base_model Haozhan72/Openvla-oft-SFT-libero10-trajall \
        --lora_rank 32 \
        --lora_path RLinf/RLinf-OpenVLAOFT-ManiSkill-Base-Lora

    # From an already-converted .pt file (skip DCP stage):
    python export_checkpoint.py \
        --config config/ppo_openvlaoft_pickcube.yaml \
        --pt_path /tmp/model.pt \
        --output_dir ./exported_model

    # Save only the LoRA adapter (don't merge into base model):
    python export_checkpoint.py \
        --config config/ppo_openvlaoft_pickcube.yaml \
        --ckpt_dir .../actor \
        --output_dir ./exported_model \
        --no_merge_lora
"""

import argparse
import os
import sys
import tempfile

import torch

try:
    import yaml
except ImportError:
    yaml = None


class PlainConfig:
    """Dot-accessible config wrapper that keeps all values as plain Python types.

    Unlike OmegaConf DictConfig, lists stay as list (not ListConfig) and dicts
    stay as dict (not DictConfig), so they are JSON-serializable when HuggingFace
    code calls json.dumps on the model config.
    """

    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, PlainConfig(v))
            else:
                setattr(self, k, v)
        self._keys = list(d.keys())

    def get(self, key, default=None):
        return getattr(self, key, default)

    def items(self):
        for k in self._keys:
            yield k, getattr(self, k)

    def __contains__(self, key):
        return key in self._keys

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self._keys)


def _find_model_defaults_yaml(config_path: str, model_type: str) -> str | None:
    """Find the model defaults YAML referenced by a training config."""
    # The training config lives in e.g. examples/embodiment/config/
    # Model defaults are in examples/embodiment/config/model/
    config_dir = os.path.dirname(os.path.abspath(config_path))

    # Map model_type to default yaml name
    model_yaml_map = {
        "openvla_oft": "openvla_oft.yaml",
        "openpi": "pi0_5.yaml",
        "openvla": "openvla.yaml",
    }
    yaml_name = model_yaml_map.get(model_type)
    if yaml_name is None:
        return None

    candidate = os.path.join(config_dir, "model", yaml_name)
    if os.path.isfile(candidate):
        return candidate
    return None


def load_model_config_from_yaml(config_path: str) -> dict:
    """Build a complete model config from a training YAML.

    Loads the model defaults YAML as a base, then overlays the training
    config's actor.model overrides, so all required fields are present.
    """
    if yaml is None:
        raise ImportError("PyYAML is required for --config. Install with: pip install pyyaml")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    actor = raw.get("actor", {})
    train_model_overrides = actor.get("model", {})
    model_type = train_model_overrides.get("model_type", "openvla_oft")

    # Load the full model defaults YAML
    defaults_path = _find_model_defaults_yaml(config_path, model_type)
    if defaults_path is not None:
        with open(defaults_path) as f:
            model_cfg = yaml.safe_load(f)
    else:
        model_cfg = {}

    # Overlay training config overrides
    model_cfg.update(train_model_overrides)

    return model_cfg


def convert_dcp_to_pt(dcp_path: str, output_pt_path: str) -> None:
    """Stage 1: Convert DCP checkpoint to a PyTorch state dict .pt file."""
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    print(f"[Stage 1] Converting DCP checkpoint: {dcp_path}")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = os.path.join(temp_dir, "temp_torch_save.pt")
        dcp_to_torch_save(dcp_path, temp_file)
        temp_pt = torch.load(temp_file, weights_only=False)
        model_state_dict = temp_pt["fsdp_checkpoint"]["model"]
        torch.save(model_state_dict, output_pt_path)
    print(f"[Stage 1] Saved state dict to: {output_pt_path}")


def convert_pt_to_hf(
    pt_path: str,
    output_dir: str,
    model_config: dict,
    merge_lora: bool = True,
    cpu_only: bool = False,
) -> None:
    """Stage 2: Load .pt state dict into model and save as HuggingFace format.

    Args:
        pt_path: Path to the .pt state dict file.
        output_dir: Directory to save the exported model.
        model_config: Complete model config dict (all fields get_model expects).
        merge_lora: Whether to merge LoRA into the base model.
        cpu_only: Run on CPU only.
    """
    if cpu_only:
        # Hide CUDA from torch so get_model() keeps everything on CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        torch.cuda.is_available = lambda: False
        print("[Stage 2] Running in CPU-only mode")

    from rlinf.models import get_model

    model_type = model_config.get("model_type", "openvla_oft")
    base_model_path = model_config.get("model_path", "")
    is_lora = model_config.get("is_lora", False)

    print(f"[Stage 2] Loading model: {model_type} from {base_model_path}")

    # Ensure add_value_head is False for export (no value head needed)
    model_config["add_value_head"] = False

    # Use a plain DictConfig but immediately resolve back to native Python types.
    # get_model() iterates the config and calls setattr() on the HF PretrainedConfig,
    # and HF's json.dumps chokes on OmegaConf ListConfig/DictConfig objects.
    # PlainConfig provides dot-access while keeping values as plain Python types.
    model_cfg = PlainConfig(model_config)
    model = get_model(model_cfg)

    print(f"[Stage 2] Loading checkpoint: {pt_path}")
    map_location = "cpu" if cpu_only else None
    state_dict = torch.load(pt_path, weights_only=False, map_location=map_location)
    # Filter out value head weights if present
    state_dict = {
        k: v for k, v in state_dict.items() if not k.startswith("value_head.")
    }
    model.load_state_dict(state_dict, strict=False)

    os.makedirs(output_dir, exist_ok=True)

    # Copy config/code files from base model
    _copy_model_config_and_code(base_model_path, output_dir)

    # Model-specific save helpers (e.g., vision backbone for OFT)
    model_save_helper = _get_model_save_helper(model_type)

    if is_lora:
        if merge_lora:
            print(f"[Stage 2] Merging LoRA weights into base model...")
            model = model.merge_and_unload()
            model.save_pretrained(output_dir, safe_serialization=True)
            if model_save_helper is not None:
                model_save_helper(model.state_dict(), model_cfg, output_dir)
            print(f"[Stage 2] Saved merged model to: {output_dir}")
        else:
            lora_dir = os.path.join(output_dir, "lora_adapter")
            model.save_pretrained(lora_dir, safe_serialization=True)
            print(f"[Stage 2] Saved LoRA adapter to: {lora_dir}")
    else:
        _save_state_dict_sharded(model.state_dict(), output_dir)
        if model_save_helper is not None:
            model_save_helper(model.state_dict(), model_cfg, output_dir)
        print(f"[Stage 2] Saved full model to: {output_dir}")


def _copy_model_config_and_code(model_path: str, save_path: str) -> None:
    """Copy .py, .json, .md files from base model to output dir."""
    import shutil

    if not os.path.exists(model_path):
        print(f"  Warning: base model path {model_path} not found locally, skipping config copy")
        return
    os.makedirs(save_path, exist_ok=True)
    suffixes = (".py", ".json", ".md")
    for root, _, files in os.walk(model_path):
        for file in files:
            if file.endswith(suffixes):
                src = os.path.join(root, file)
                rel = os.path.relpath(src, model_path)
                dst = os.path.join(save_path, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)


def _get_model_save_helper(model_type: str):
    """Return model-specific post-save helper, or None."""
    if model_type == "openvla_oft":
        return _openvla_oft_save_helper
    return None


def _openvla_oft_save_helper(model_state_dict, model_config, save_path, **kwargs):
    """Save vision backbone and proprio projector if present."""
    global_step = kwargs.get("global_step", 0)
    use_film = model_config.get("use_film", False) if hasattr(model_config, "get") else getattr(model_config, "use_film", False)
    use_proprio = model_config.get("use_proprio", False) if hasattr(model_config, "get") else getattr(model_config, "use_proprio", False)

    if use_film:
        vision_sd = {
            k.replace("vision_backbone.", "", 1): v
            for k, v in model_state_dict.items()
            if k.startswith("vision_backbone.")
        }
        torch.save(vision_sd, os.path.join(save_path, f"vision_backbone--{global_step}_checkpoint.pt"))
    if use_proprio:
        proprio_sd = {
            k.replace("proprio_projector.", "", 1): v
            for k, v in model_state_dict.items()
            if k.startswith("proprio_projector.")
        }
        torch.save(proprio_sd, os.path.join(save_path, f"proprio_projector--{global_step}_checkpoint.pt"))


def _save_state_dict_sharded(state_dict: dict, out_dir: str) -> None:
    """Save as sharded safetensors (for non-LoRA models)."""
    try:
        from rlinf.utils.ckpt_convertor.fsdp_convertor.utils import save_state_dict_sharded_safetensors
        save_state_dict_sharded_safetensors(state_dict=state_dict, out_dir=out_dir)
    except ImportError:
        # Fallback: save as single .pt file
        torch.save(state_dict, os.path.join(out_dir, "model.pt"))
        print("  (safetensors not available, saved as model.pt)")


def main():
    parser = argparse.ArgumentParser(
        description="Export RLinf training checkpoint to HuggingFace model format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input (one of these required)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--ckpt_dir", type=str,
        help="Path to actor checkpoint dir (containing dcp_checkpoint/ subdir)",
    )
    input_group.add_argument(
        "--pt_path", type=str,
        help="Path to an already-converted .pt state dict (skip DCP stage)",
    )

    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the exported model")

    # Training config (extracts model settings automatically)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to training YAML config (e.g., ppo_openvlaoft_pickcube.yaml). "
                             "Extracts model_type, base_model, lora settings, etc. "
                             "CLI args below override config values.")

    # Model config (used as overrides or when --config is not provided)
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["openvla_oft", "openpi"],
                        help="Model type")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model path or HuggingFace ID")
    parser.add_argument("--policy_setup", type=str, default=None,
                        help="Policy setup")
    parser.add_argument("--unnorm_key", type=str, default=None,
                        help="Unnormalization key")
    parser.add_argument("--max_prompt_length", type=int, default=None,
                        help="Max prompt length")

    # LoRA config
    parser.add_argument("--no_lora", action="store_true",
                        help="Model does not use LoRA")
    parser.add_argument("--lora_rank", type=int, default=None,
                        help="LoRA rank")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to base LoRA adapter")
    parser.add_argument("--merge_lora", action="store_true", default=True,
                        help="Merge LoRA into base model (default: True)")
    parser.add_argument("--no_merge_lora", action="store_true",
                        help="Save LoRA adapter separately without merging")

    # Device
    parser.add_argument("--cpu", action="store_true",
                        help="Run entirely on CPU (no GPU required)")

    args = parser.parse_args()

    # Build model config: start from training YAML (includes model defaults),
    # then override with any explicit CLI args.
    if args.config:
        model_config = load_model_config_from_yaml(args.config)
        print(f"Loaded model config from: {args.config}")
    else:
        model_config = {}

    # CLI overrides (only apply if explicitly provided)
    if args.model_type is not None:
        model_config["model_type"] = args.model_type
    if args.base_model is not None:
        model_config["model_path"] = args.base_model
    if args.policy_setup is not None:
        model_config["policy_setup"] = args.policy_setup
    if args.unnorm_key is not None:
        model_config["unnorm_key"] = args.unnorm_key
    if args.max_prompt_length is not None:
        model_config["max_prompt_length"] = args.max_prompt_length
    if args.no_lora:
        model_config["is_lora"] = False
    if args.lora_rank is not None:
        model_config["lora_rank"] = args.lora_rank
    if args.lora_path is not None:
        model_config["lora_path"] = args.lora_path

    merge_lora = args.merge_lora and not args.no_merge_lora

    print(f"Export settings: model_type={model_config.get('model_type', '?')}, "
          f"model_path={model_config.get('model_path', '?')}, "
          f"is_lora={model_config.get('is_lora', False)}, "
          f"lora_rank={model_config.get('lora_rank', '?')}")

    # Stage 1: DCP -> .pt (if needed)
    if args.ckpt_dir:
        # Auto-detect dcp_checkpoint subdir
        dcp_path = args.ckpt_dir
        dcp_subdir = os.path.join(args.ckpt_dir, "dcp_checkpoint")
        if os.path.isdir(dcp_subdir):
            dcp_path = dcp_subdir

        pt_path = os.path.join(args.output_dir, "_intermediate_state_dict.pt")
        os.makedirs(args.output_dir, exist_ok=True)
        convert_dcp_to_pt(dcp_path, pt_path)
    else:
        pt_path = args.pt_path

    # Stage 2: .pt -> HuggingFace
    convert_pt_to_hf(
        pt_path=pt_path,
        output_dir=args.output_dir,
        model_config=model_config,
        merge_lora=merge_lora,
        cpu_only=args.cpu,
    )

    # Clean up intermediate file
    intermediate = os.path.join(args.output_dir, "_intermediate_state_dict.pt")
    if os.path.exists(intermediate):
        os.remove(intermediate)
        print(f"Cleaned up intermediate file: {intermediate}")

    print(f"\nExport complete: {args.output_dir}")


if __name__ == "__main__":
    main()
