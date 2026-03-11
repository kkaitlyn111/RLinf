#!/usr/bin/env python3
"""Standalone GPU VRAM estimator for RLinf PPO training of VLA models.

Models each component (env, rollout, actor) as a SEPARATE OS PROCESS with its
own CUDA context. On shared GPUs, process memories are summed (they can't share
or offload each other's memory). With enable_offload, actor and rollout alternate
but the non-active process still holds CUDA context + residual memory.

Usage:
    # From CLI args:
    python estimate_vram.py --model openvla_oft --num_gpus 4 --micro_batch_size 8 \\
        --global_batch_size 128 --sharding shard_grad_op --gradient_checkpointing \\
        --lora --lora_rank 32 --num_envs 64 --enable_offload

    # From YAML config:
    python estimate_vram.py --config ppo_openvlaoft_pickcube.yaml

    # Suggest max batch size:
    python estimate_vram.py --config ppo_openvlaoft_pickcube.yaml --suggest --gpu_mem 48
"""

import argparse
import math
import os
from dataclasses import dataclass, field

try:
    import yaml
except ImportError:
    yaml = None

GB = 1024**3
MB = 1024**2

# Per-process CUDA context overhead (driver, allocator metadata, cuBLAS handles)
CUDA_CONTEXT_OVERHEAD = 700 * MB
# Residual GPU memory when a process has offloaded its model to CPU
# (CUDA context stays, plus small PyTorch internal buffers)
OFFLOADED_RESIDUAL = 200 * MB


# ---------------------------------------------------------------------------
# Architecture specs
# ---------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    total_params: int
    num_layers: int
    hidden_size: int
    num_heads: int
    head_dim: int
    vocab_size: int
    vision_patches: int
    default_prompt_len: int
    action_tokens: int  # for autoregressive models (OFT)
    bytes_per_param: float  # bf16=2, mixed≈2.5
    # LoRA: approximate params per unit rank (sum over all target modules)
    lora_params_per_rank: int
    # Value head
    value_head_params: int
    # OpenPI-specific
    has_expert: bool = False
    expert_params: int = 0
    expert_layers: int = 0
    expert_hidden: int = 0
    num_denoising_steps: int = 1
    prefix_tokens: int = 0
    action_horizon: int = 0
    action_dim: int = 7


OPENVLA_OFT = ModelSpec(
    name="OpenVLA-OFT",
    total_params=7_000_000_000,
    num_layers=32,
    hidden_size=4096,
    num_heads=32,
    head_dim=128,
    vocab_size=32000,
    vision_patches=256,
    default_prompt_len=30,
    action_tokens=56,  # 8 chunks * 7 action_dim
    bytes_per_param=2.0,  # bf16
    lora_params_per_rank=3_300_000,
    value_head_params=2_200_000,  # 4096→512→128→1
)

OPENPI_PI05 = ModelSpec(
    name="OpenPI (Pi0.5)",
    total_params=4_000_000_000,
    num_layers=18,  # VLM layers
    hidden_size=2048,  # VLM hidden
    num_heads=16,
    head_dim=128,
    vocab_size=257152,  # Gemma
    vision_patches=256,
    default_prompt_len=200,
    action_tokens=0,  # not autoregressive
    bytes_per_param=2.5,  # mixed bf16/fp32
    lora_params_per_rank=800_000,
    value_head_params=2_900_000,  # 2048→1024→512→256→1
    has_expert=True,
    expert_params=1_300_000_000,
    expert_layers=18,
    expert_hidden=1024,
    num_denoising_steps=4,
    prefix_tokens=968,
    action_horizon=8,
    action_dim=7,
)

MODEL_SPECS = {
    "openvla_oft": OPENVLA_OFT,
    "openpi": OPENPI_PI05,
}


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    model_type: str = "openvla_oft"
    num_gpus: int = 4
    micro_batch_size: int = 8
    global_batch_size: int = 128
    sharding: str = "shard_grad_op"  # full_shard, shard_grad_op, no_shard
    gradient_checkpointing: bool = True
    is_lora: bool = True
    lora_rank: int = 32
    add_value_head: bool = True
    num_envs: int = 64
    max_steps_per_epoch: int = 64
    enable_offload: bool = True  # actor/rollout CPU offload
    env_gpus: str = "0-1"
    rollout_gpus: str = "2-3"
    actor_gpus: str = "0-3"
    # OpenPI overrides
    num_denoising_steps: int = 4
    num_action_chunks: int = 8
    action_dim: int = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_gpu_range(s: str) -> set[int]:
    """Parse '0-3' into {0,1,2,3} or '2' into {2}."""
    s = s.strip()
    if "-" in s:
        parts = s.split("-")
        return set(range(int(parts[0]), int(parts[1]) + 1))
    return {int(s)}


def fmt_bytes(b: float) -> str:
    """Format bytes as human-readable GB."""
    return f"{b / GB:.2f} GB"


def fmt_bytes_short(b: float) -> str:
    if b >= GB:
        return f"{b / GB:.1f}G"
    return f"{b / MB:.0f}M"


# ---------------------------------------------------------------------------
# Memory estimators (component-level)
# ---------------------------------------------------------------------------

def estimate_lora_params(spec: ModelSpec, rank: int) -> int:
    return spec.lora_params_per_rank * rank


def get_trainable_params(spec: ModelSpec, cfg: TrainingConfig) -> int:
    """Total trainable parameters."""
    params = 0
    if cfg.is_lora:
        params += estimate_lora_params(spec, cfg.lora_rank)
    else:
        params += spec.total_params
    if cfg.add_value_head:
        params += spec.value_head_params
    return params


def estimate_model_weights(spec: ModelSpec, cfg: TrainingConfig) -> float:
    """Model weights memory per actor GPU (bytes)."""
    if cfg.sharding == "full_shard":
        shard = cfg.num_gpus
    else:  # shard_grad_op, no_shard: params replicated
        shard = 1
    return spec.total_params * spec.bytes_per_param / shard


def estimate_lora_weights(spec: ModelSpec, cfg: TrainingConfig) -> float:
    """LoRA adapter weights memory (bytes). Not sharded by default."""
    if not cfg.is_lora:
        return 0
    return estimate_lora_params(spec, cfg.lora_rank) * spec.bytes_per_param


def estimate_optimizer_memory(spec: ModelSpec, cfg: TrainingConfig) -> float:
    """AdamW optimizer states per GPU (bytes). m + v in float32."""
    trainable = get_trainable_params(spec, cfg)
    total = trainable * 2 * 4  # 2 states * 4 bytes (float32)
    if cfg.sharding in ("full_shard", "shard_grad_op"):
        return total / cfg.num_gpus
    return total


def estimate_gradient_memory(spec: ModelSpec, cfg: TrainingConfig) -> float:
    """Gradient memory per GPU (bytes)."""
    trainable = get_trainable_params(spec, cfg)
    total = trainable * spec.bytes_per_param
    if cfg.sharding in ("full_shard", "shard_grad_op"):
        return total / cfg.num_gpus
    return total


def _seq_len(spec: ModelSpec, cfg: TrainingConfig) -> int:
    """Effective sequence length during training."""
    if spec.has_expert:
        return spec.prefix_tokens
    return spec.vision_patches + spec.default_prompt_len + spec.action_tokens + 1


def estimate_activation_memory(spec: ModelSpec, cfg: TrainingConfig) -> float:
    """Activation memory per GPU during training forward/backward (bytes)."""
    seq_len = _seq_len(spec, cfg)
    bpe = spec.bytes_per_param

    if cfg.gradient_checkpointing:
        factor = math.sqrt(spec.num_layers) * 4
    else:
        factor = spec.num_layers * 10

    vlm_activation = cfg.micro_batch_size * seq_len * spec.hidden_size * bpe * factor

    if spec.has_expert:
        expert_seq = spec.action_horizon
        if cfg.gradient_checkpointing:
            expert_factor = math.sqrt(spec.expert_layers) * 4
        else:
            expert_factor = spec.expert_layers * 10
        expert_activation = (
            cfg.micro_batch_size
            * expert_seq
            * spec.expert_hidden
            * bpe
            * expert_factor
            * cfg.num_denoising_steps
        )
        return vlm_activation + expert_activation

    return vlm_activation


def estimate_kv_cache(spec: ModelSpec, cfg: TrainingConfig, batch_size: int) -> float:
    """KV cache memory for inference/rollout (bytes)."""
    seq_len = _seq_len(spec, cfg)
    return (
        2
        * spec.num_layers
        * batch_size
        * seq_len
        * spec.head_dim
        * spec.num_heads
        * spec.bytes_per_param
    )


def estimate_sim_memory(num_envs: int) -> float:
    """ManiSkill GPU simulation memory (bytes).

    Empirical: ~450 MB/env for simple tabletop tasks (PickCube, StackCube),
    ~830 MB/env for complex scene tasks (PutOnPlateInScene).
    Using 500 MB/env as a conservative middle estimate.
    """
    return num_envs * 500 * MB


def estimate_rollout_buffers(num_envs: int, max_steps: int) -> float:
    """Rollout data buffers on GPU (bytes). Most data is on CPU; this is the GPU portion."""
    per_step_per_env = 1024  # ~1KB of float tensors per step per env
    return num_envs * max_steps * per_step_per_env


# ---------------------------------------------------------------------------
# Per-process memory model
# ---------------------------------------------------------------------------

@dataclass
class ProcessMemory:
    """Memory for a single worker process on a GPU.

    Each RLinf component (env, rollout, actor) runs as a separate OS process
    with its own CUDA context. They cannot share or free each other's memory.
    """
    component: str  # "env", "rollout", "actor"
    gpu_id: int

    # Active-phase memory (excluding CUDA context)
    active: float = 0
    # Weight sync spike memory (excluding CUDA context)
    sync_spike: float = 0
    # Offloaded-phase residual (excluding CUDA context; model on CPU)
    offloaded: float = OFFLOADED_RESIDUAL

    # Breakdown for reporting
    details: dict = field(default_factory=dict)

    @property
    def cuda_context(self) -> float:
        return CUDA_CONTEXT_OVERHEAD

    def mem_active(self) -> float:
        """Total process memory when active."""
        return self.cuda_context + self.active

    def mem_sync(self) -> float:
        """Total process memory during weight sync."""
        if self.sync_spike > 0:
            return self.cuda_context + self.sync_spike
        return self.mem_active()

    def mem_offloaded(self) -> float:
        """Total process memory when offloaded to CPU."""
        return self.cuda_context + self.offloaded


@dataclass
class GPUPeakResult:
    """Per-GPU peak memory analysis."""
    gpu_id: int
    processes: list[ProcessMemory]
    peak: float
    peak_phase: str
    phase_breakdown: dict  # phase_name -> total memory


def compute_processes(spec: ModelSpec, cfg: TrainingConfig) -> list[ProcessMemory]:
    """Compute per-process memory for all components across all GPUs."""
    env_gpus = parse_gpu_range(cfg.env_gpus)
    rollout_gpus = parse_gpu_range(cfg.rollout_gpus)
    actor_gpus = parse_gpu_range(cfg.actor_gpus)

    num_env_gpus = len(env_gpus)
    num_rollout_gpus = len(rollout_gpus)
    envs_per_env_gpu = cfg.num_envs / max(num_env_gpus, 1)
    rollout_batch = cfg.num_envs / max(num_rollout_gpus, 1)

    # Precompute actor memory components
    model_w = estimate_model_weights(spec, cfg)
    lora_w = estimate_lora_weights(spec, cfg)
    opt_mem = estimate_optimizer_memory(spec, cfg)
    grad_mem = estimate_gradient_memory(spec, cfg)
    act_mem = estimate_activation_memory(spec, cfg)

    actor_active = model_w + lora_w + opt_mem + grad_mem + act_mem

    # Actor sync spike: FSDP all-gather to produce full state dict for rollout
    if cfg.sharding == "full_shard":
        allgather_extra = spec.total_params * spec.bytes_per_param * (cfg.num_gpus - 1) / cfg.num_gpus
    else:
        allgather_extra = 0
    actor_sync = model_w + lora_w + opt_mem + allgather_extra

    actor_details = {
        "Model weights": model_w,
        "LoRA weights": lora_w,
        "Optimizer states (AdamW)": opt_mem,
        "Gradients": grad_mem,
        "Activations": act_mem,
    }
    if not cfg.is_lora:
        del actor_details["LoRA weights"]

    # Precompute rollout memory components
    rollout_model_mem = spec.total_params * spec.bytes_per_param  # full copy, not sharded
    kv_mem = estimate_kv_cache(spec, cfg, int(rollout_batch))

    rollout_active = rollout_model_mem + kv_mem
    rollout_sync = rollout_model_mem * 2  # old model + incoming weights

    rollout_details = {
        "Rollout model (full copy)": rollout_model_mem,
        "KV cache": kv_mem,
    }

    # Precompute env memory
    sim_mem = estimate_sim_memory(int(envs_per_env_gpu))
    buf_mem = estimate_rollout_buffers(int(envs_per_env_gpu), cfg.max_steps_per_epoch)
    env_active = sim_mem + buf_mem
    env_details = {
        "ManiSkill sim": sim_mem,
        "Rollout buffers": buf_mem,
    }

    processes = []
    for gpu_id in range(cfg.num_gpus):
        if gpu_id in env_gpus:
            proc = ProcessMemory(
                component="env",
                gpu_id=gpu_id,
                active=env_active,
                sync_spike=0,
                offloaded=env_active,  # env never offloads; always resident
                details=dict(env_details),
            )
            processes.append(proc)

        if gpu_id in rollout_gpus:
            proc = ProcessMemory(
                component="rollout",
                gpu_id=gpu_id,
                active=rollout_active,
                sync_spike=rollout_sync,
                details=dict(rollout_details),
            )
            processes.append(proc)

        if gpu_id in actor_gpus:
            proc = ProcessMemory(
                component="actor",
                gpu_id=gpu_id,
                active=actor_active,
                sync_spike=actor_sync,
                details=dict(actor_details),
            )
            if allgather_extra > 0:
                proc.details["Weight sync all-gather (temp)"] = allgather_extra
            processes.append(proc)

    return processes


def compute_gpu_peaks(
    processes: list[ProcessMemory], cfg: TrainingConfig
) -> list[GPUPeakResult]:
    """Compute peak memory per GPU by summing concurrent process memories across phases."""
    # Group processes by GPU
    by_gpu: dict[int, list[ProcessMemory]] = {}
    for p in processes:
        by_gpu.setdefault(p.gpu_id, []).append(p)

    results = []
    for gpu_id in sorted(by_gpu):
        procs = by_gpu[gpu_id]
        comp_map = {p.component: p for p in procs}

        # Compute memory for each phase by summing all processes on this GPU.
        # Env is always active (separate process, can't be offloaded by others).
        # With offload: actor and rollout alternate active/offloaded phases.
        # Without offload: actor and rollout both active simultaneously.

        phases = {}

        def _phase_total(actor_state: str, rollout_state: str) -> float:
            total = 0
            if "env" in comp_map:
                total += comp_map["env"].mem_active()  # always resident
            if "actor" in comp_map:
                if actor_state == "active":
                    total += comp_map["actor"].mem_active()
                elif actor_state == "sync":
                    total += comp_map["actor"].mem_sync()
                elif actor_state == "offloaded":
                    total += comp_map["actor"].mem_offloaded()
            if "rollout" in comp_map:
                if rollout_state == "active":
                    total += comp_map["rollout"].mem_active()
                elif rollout_state == "sync":
                    total += comp_map["rollout"].mem_sync()
                elif rollout_state == "offloaded":
                    total += comp_map["rollout"].mem_offloaded()
            return total

        if cfg.enable_offload:
            # Training phase: actor active, rollout offloaded
            phases["training"] = _phase_total("active", "offloaded")
            # Rollout phase: rollout active, actor offloaded
            phases["rollout"] = _phase_total("offloaded", "active")
            # Weight sync (rollout recv): rollout gets 2x model, actor offloaded
            if "rollout" in comp_map:
                phases["weight_sync (rollout)"] = _phase_total("offloaded", "sync")
            # Weight sync (actor all-gather): actor gathers full params, rollout offloaded
            if "actor" in comp_map and comp_map["actor"].sync_spike > 0:
                phases["weight_sync (actor)"] = _phase_total("sync", "offloaded")
        else:
            # No offload: actor and rollout both active
            phases["training+rollout"] = _phase_total("active", "active")
            # Sync spikes still possible
            if "rollout" in comp_map:
                phases["weight_sync (rollout)"] = _phase_total("active", "sync")
            if "actor" in comp_map and comp_map["actor"].sync_spike > 0:
                phases["weight_sync (actor)"] = _phase_total("sync", "active")

        peak_phase = max(phases, key=phases.get)
        peak_mem = phases[peak_phase]
        # Add 10% for fragmentation
        peak_mem_with_overhead = peak_mem * 1.10

        results.append(GPUPeakResult(
            gpu_id=gpu_id,
            processes=procs,
            peak=peak_mem_with_overhead,
            peak_phase=peak_phase,
            phase_breakdown=phases,
        ))

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    spec: ModelSpec,
    cfg: TrainingConfig,
    processes: list[ProcessMemory],
    gpu_peaks: list[GPUPeakResult],
):
    trainable = get_trainable_params(spec, cfg)
    grad_accum = cfg.global_batch_size / (cfg.micro_batch_size * cfg.num_gpus)

    print("=" * 70)
    print(f"  GPU VRAM Estimation: {spec.name} (multi-process model)")
    print("=" * 70)
    print()
    print("Model Architecture:")
    print(f"  Total params:        {spec.total_params / 1e9:.2f}B ({fmt_bytes(spec.total_params * spec.bytes_per_param)})")
    if cfg.is_lora:
        lp = estimate_lora_params(spec, cfg.lora_rank)
        print(f"  LoRA params:         {lp / 1e6:.1f}M (rank={cfg.lora_rank})")
    print(f"  Trainable params:    {trainable / 1e6:.1f}M")
    if cfg.add_value_head:
        print(f"  Value head params:   {spec.value_head_params / 1e6:.1f}M")
    print()
    print("Training Configuration:")
    print(f"  micro_batch_size:    {cfg.micro_batch_size}")
    print(f"  global_batch_size:   {cfg.global_batch_size}")
    print(f"  grad_accum_steps:    {grad_accum:.0f}")
    print(f"  FSDP sharding:       {cfg.sharding}")
    print(f"  grad checkpointing:  {cfg.gradient_checkpointing}")
    print(f"  enable_offload:      {cfg.enable_offload}")
    print(f"  num_gpus:            {cfg.num_gpus}")
    print(f"  num_envs:            {cfg.num_envs}")
    if spec.has_expert:
        print(f"  denoising_steps:     {cfg.num_denoising_steps}")
    print()

    gpu_sizes = [24, 40, 48, 80]

    for gp in gpu_peaks:
        components = [p.component for p in gp.processes]
        role_str = "+".join(components)
        print(f"GPU {gp.gpu_id} ({role_str}):")
        print(f"  Processes: {len(gp.processes)} (each with ~{fmt_bytes_short(CUDA_CONTEXT_OVERHEAD)} CUDA context)")
        print()

        # Per-process breakdown
        for proc in gp.processes:
            print(f"  [{proc.component.upper()} process]")
            print(f"    {'CUDA context':<32} {fmt_bytes(proc.cuda_context):>10}")
            for label, mem in proc.details.items():
                print(f"    {label:<32} {fmt_bytes(mem):>10}")
            print(f"    {'--- Active total':<32} {fmt_bytes(proc.mem_active()):>10}")
            if proc.sync_spike > 0:
                print(f"    {'--- Sync spike total':<32} {fmt_bytes(proc.mem_sync()):>10}")
            if proc.component != "env":
                print(f"    {'--- Offloaded total':<32} {fmt_bytes(proc.mem_offloaded()):>10}")
            print()

        # Phase analysis
        print(f"  Phase analysis (sum of all processes on this GPU):")
        for phase_name, phase_mem in sorted(gp.phase_breakdown.items(), key=lambda x: -x[1]):
            marker = " <<<" if phase_name == gp.peak_phase else ""
            print(f"    {phase_name:<32} {fmt_bytes(phase_mem):>10}{marker}")

        print()
        print(f"  PEAK (+ 10% fragmentation):    {fmt_bytes(gp.peak)}")
        print(f"  Limiting phase:                {gp.peak_phase}")
        print()

        # Fit check
        print(f"  Fit check:")
        for gs in gpu_sizes:
            ok = "OK" if gp.peak <= gs * GB else "OOM"
            margin = gs * GB - gp.peak
            margin_str = f"+{fmt_bytes_short(margin)}" if margin > 0 else f"{fmt_bytes_short(margin)}"
            print(f"    {gs:>3} GB: {ok:<4} ({margin_str})")
        print()


# ---------------------------------------------------------------------------
# YAML config loader
# ---------------------------------------------------------------------------

def load_from_yaml(path: str) -> TrainingConfig:
    """Best-effort extraction of training config from a Hydra-style YAML."""
    if yaml is None:
        raise ImportError("PyYAML is required for --config. Install with: pip install pyyaml")

    with open(path) as f:
        raw = yaml.safe_load(f)

    actor = raw.get("actor", {})
    model = actor.get("model", {})
    fsdp = actor.get("fsdp_config", {})
    rollout = raw.get("rollout", {})
    env_train = raw.get("env", {}).get("train", {})
    cluster = raw.get("cluster", {}).get("component_placement", {})

    model_type = model.get("model_type", "openvla_oft")

    # Determine num_gpus from actor placement
    actor_range = str(cluster.get("actor", "0-3"))
    num_gpus = len(parse_gpu_range(actor_range))

    num_action_chunks = model.get("num_action_chunks", 8)
    action_dim = model.get("action_dim", 7)

    cfg = TrainingConfig(
        model_type=model_type,
        num_gpus=num_gpus,
        micro_batch_size=actor.get("micro_batch_size", 8),
        global_batch_size=actor.get("global_batch_size", 128),
        sharding=fsdp.get("sharding_strategy", "shard_grad_op"),
        gradient_checkpointing=fsdp.get("gradient_checkpointing", False),
        is_lora=model.get("is_lora", False),
        lora_rank=model.get("lora_rank", 32),
        add_value_head=model.get("add_value_head", False),
        num_envs=env_train.get("total_num_envs", 16),
        max_steps_per_epoch=env_train.get("max_steps_per_rollout_epoch", 50),
        enable_offload=actor.get("enable_offload", False)
        or rollout.get("enable_offload", False),
        env_gpus=str(cluster.get("env", "0-1")),
        rollout_gpus=str(cluster.get("rollout", "2-3")),
        actor_gpus=actor_range,
        num_denoising_steps=model.get("num_steps", 4),
        num_action_chunks=num_action_chunks,
        action_dim=action_dim,
    )
    return cfg


# ---------------------------------------------------------------------------
# Suggest mode
# ---------------------------------------------------------------------------

def suggest_max_batch(spec: ModelSpec, cfg: TrainingConfig, gpu_mem_gb: float):
    """Binary search for max micro_batch_size that fits in gpu_mem_gb."""
    target = gpu_mem_gb * GB

    # First check if even batch=1 fits
    test_cfg = TrainingConfig(**{**cfg.__dict__, "micro_batch_size": 1})
    procs = compute_processes(spec, test_cfg)
    peaks = compute_gpu_peaks(procs, test_cfg)
    min_peak = max(gp.peak for gp in peaks)

    if min_peak > target:
        print(f"\nCannot fit even batch_size=1 in {gpu_mem_gb:.0f} GB GPUs!")
        print(f"  Minimum peak VRAM (batch=1): {fmt_bytes(min_peak)}")
        _print_bottleneck(peaks, target)
        return

    low, high = 1, 256
    best = 0

    while low <= high:
        mid = (low + high) // 2
        test_cfg = TrainingConfig(**{**cfg.__dict__, "micro_batch_size": mid})
        procs = compute_processes(spec, test_cfg)
        peaks = compute_gpu_peaks(procs, test_cfg)
        max_peak = max(gp.peak for gp in peaks)

        if max_peak <= target:
            best = mid
            low = mid + 1
        else:
            high = mid - 1

    print(f"\nSuggested max micro_batch_size for {gpu_mem_gb:.0f} GB GPUs: {best}")
    if best > 0:
        test_cfg = TrainingConfig(**{**cfg.__dict__, "micro_batch_size": best})
        procs = compute_processes(spec, test_cfg)
        peaks = compute_gpu_peaks(procs, test_cfg)
        max_peak = max(gp.peak for gp in peaks)
        print(f"  Peak VRAM at batch={best}: {fmt_bytes(max_peak)}")
        grad_accum = cfg.global_batch_size / (best * cfg.num_gpus)
        print(f"  Gradient accumulation steps: {grad_accum:.1f}")
        if grad_accum != int(grad_accum):
            print(f"  WARNING: global_batch_size={cfg.global_batch_size} not evenly "
                  f"divisible by micro_batch={best} * num_gpus={cfg.num_gpus}")

        _print_bottleneck(peaks, target)


def _print_bottleneck(peaks: list[GPUPeakResult], target: float):
    """Identify and print the bottleneck GPU and phase."""
    worst = max(peaks, key=lambda gp: gp.peak)
    roles = "+".join(p.component for p in worst.processes)

    print(f"\n  Bottleneck: GPU {worst.gpu_id} ({roles})")
    print(f"    Limiting phase: {worst.peak_phase}")
    for phase, mem in sorted(worst.phase_breakdown.items(), key=lambda x: -x[1]):
        marker = " <<<" if phase == worst.peak_phase else ""
        print(f"      {phase:<30} {fmt_bytes(mem):>10}{marker}")

    # Identify what dominates in the peak phase
    if len(worst.processes) > 1:
        print(f"\n  Per-process contribution in '{worst.peak_phase}' phase:")
        for proc in worst.processes:
            if proc.component == "env":
                mem = proc.mem_active()
            elif "weight_sync" in worst.peak_phase:
                if proc.component in worst.peak_phase:
                    mem = proc.mem_sync()
                else:
                    mem = proc.mem_offloaded()
            elif "training" in worst.peak_phase:
                mem = proc.mem_active() if proc.component in ("actor", "env") else proc.mem_offloaded()
            elif "rollout" in worst.peak_phase:
                mem = proc.mem_active() if proc.component in ("rollout", "env") else proc.mem_offloaded()
            else:
                mem = proc.mem_active()
            print(f"      {proc.component:<12} {fmt_bytes(mem):>10}")

    # Actionable suggestions
    batch_dependent = any(p.component == "actor" for p in worst.processes)
    env_on_gpu = any(p.component == "env" for p in worst.processes)

    if "weight_sync" in worst.peak_phase:
        print(f"\n  The bottleneck is weight sync, NOT batch size.")
        print(f"  Reducing micro_batch_size won't help.")
        print(f"  Possible fixes:")
        if env_on_gpu:
            print(f"    - Reduce num_envs on this GPU (currently using significant memory)")
        print(f"    - Move env and rollout/actor to non-overlapping GPUs")
        print(f"    - Modify collective_group.py to recv weights on CPU")
    elif env_on_gpu and not batch_dependent:
        print(f"\n  This GPU is env-only; reduce num_envs to free memory.")
    elif env_on_gpu:
        env_proc = next(p for p in worst.processes if p.component == "env")
        print(f"\n  Env process uses {fmt_bytes(env_proc.mem_active())} on this GPU.")
        print(f"  This leaves only {fmt_bytes(target - env_proc.mem_active())} for the actor process.")
        print(f"  Possible fixes:")
        print(f"    - Reduce num_envs (fewer envs on this GPU)")
        print(f"    - Move env to GPUs that don't overlap with actor")
        print(f"    - Reduce micro_batch_size (shrinks actor activations)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Estimate GPU VRAM usage for RLinf PPO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, help="Path to a PPO YAML config file")
    parser.add_argument("--model", type=str, choices=["openvla_oft", "openpi"],
                        help="Model type")
    parser.add_argument("--num_gpus", type=int, help="Number of GPUs")
    parser.add_argument("--micro_batch_size", type=int, help="Micro batch size")
    parser.add_argument("--global_batch_size", type=int, help="Global batch size")
    parser.add_argument("--sharding", type=str,
                        choices=["full_shard", "shard_grad_op", "no_shard"],
                        help="FSDP sharding strategy")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--lora", action="store_true", help="Enable LoRA")
    parser.add_argument("--lora_rank", type=int, help="LoRA rank")
    parser.add_argument("--num_envs", type=int, help="Total number of environments")
    parser.add_argument("--enable_offload", action="store_true",
                        help="Enable actor/rollout CPU offload")
    parser.add_argument("--env_gpus", type=str, help="GPU range for env (e.g., '0-1')")
    parser.add_argument("--rollout_gpus", type=str, help="GPU range for rollout (e.g., '2-3')")
    parser.add_argument("--actor_gpus", type=str, help="GPU range for actor (e.g., '0-3')")
    parser.add_argument("--suggest", action="store_true",
                        help="Suggest max micro_batch_size")
    parser.add_argument("--gpu_mem", type=float, default=48,
                        help="GPU memory in GB (for --suggest mode)")

    args = parser.parse_args()

    # Load config
    if args.config:
        cfg = load_from_yaml(args.config)
    else:
        cfg = TrainingConfig()

    # CLI overrides
    if args.model is not None:
        cfg.model_type = args.model
    if args.num_gpus is not None:
        cfg.num_gpus = args.num_gpus
    if args.micro_batch_size is not None:
        cfg.micro_batch_size = args.micro_batch_size
    if args.global_batch_size is not None:
        cfg.global_batch_size = args.global_batch_size
    if args.sharding is not None:
        cfg.sharding = args.sharding
    if args.gradient_checkpointing:
        cfg.gradient_checkpointing = True
    if args.lora:
        cfg.is_lora = True
    if args.lora_rank is not None:
        cfg.lora_rank = args.lora_rank
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs
    if args.enable_offload:
        cfg.enable_offload = True
    if args.env_gpus is not None:
        cfg.env_gpus = args.env_gpus
    if args.rollout_gpus is not None:
        cfg.rollout_gpus = args.rollout_gpus
    if args.actor_gpus is not None:
        cfg.actor_gpus = args.actor_gpus

    spec = MODEL_SPECS[cfg.model_type]

    # Apply OpenPI-specific overrides from config
    if spec.has_expert:
        spec = ModelSpec(**{**spec.__dict__})  # shallow copy
        spec.num_denoising_steps = cfg.num_denoising_steps
        spec.action_tokens = cfg.num_action_chunks * cfg.action_dim

    processes = compute_processes(spec, cfg)
    gpu_peaks = compute_gpu_peaks(processes, cfg)
    print_report(spec, cfg, processes, gpu_peaks)

    if args.suggest:
        suggest_max_batch(spec, cfg, args.gpu_mem)


if __name__ == "__main__":
    main()
