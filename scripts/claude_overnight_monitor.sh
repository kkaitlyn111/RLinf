#!/bin/bash
#SBATCH --job-name=claude_monitor
#SBATCH --output=/juice5b/scr5b/kaitwang/cs234/RLinf/logs/claude_monitor_%j.out
#SBATCH --error=/juice5b/scr5b/kaitwang/cs234/RLinf/logs/claude_monitor_%j.err
#SBATCH --partition=john
#SBATCH --account=nlp
#SBATCH --cpus-per-task=15
#SBATCH --time=12:00:00

claude --dangerously-skip-permissions --verbose --output-format stream-json --include-partial-messages -p "
You are monitoring an overnight RL training job for a cs234 research project. Read /sailhome/kaitwang/.claude/projects/-juice5b-scr5b-kaitwang/memory/job_monitoring_skill.md and /juice5b/scr5b/kaitwang/cs234/CLAUDE.md fully before doing anything else.

CURRENT STATE:
- Last job 14768207 FAILED with: RuntimeError: expected scalar type Float but found BFloat16
- This happens in the rollout worker SigLIP vision encoder: LayerNorm has Float32 weights but input is BFloat16
- Root call chain: MultiStepRolloutWorker.generate -> predict_action_batch -> sample_actions -> embed_prefix -> SigLIP LayerNorm -> crash
- The model loading code in rlinf/models/embodiment/openpi/__init__.py calls model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16') which does NOT cast all params (SigLIP LayerNorm stays Float32)
- Fix needed: ensure SigLIP LayerNorm weights are cast to bfloat16 before inference in the rollout worker

YOUR TASK:
1. Diagnose the bfloat16 fix: read rlinf/models/embodiment/openpi/__init__.py and .venv/lib/python3.11/site-packages/openpi/models_pytorch/gemma_pytorch.py to understand what to_bfloat16_for_selected_params does and why SigLIP LayerNorm is excluded
2. Apply the minimal fix. Most likely option: after the to_bfloat16_for_selected_params call in __init__.py, add a line to cast the full vision tower to bfloat16. Read the actual code first to confirm the right attribute path before editing.
3. Submit job: sbatch /juice5b/scr5b/kaitwang/cs234/RLinf/scripts/run_ppo_pi05.sh
4. Monitor log efficiently: poll squeue every 30s; once running get log path via scontrol show job [JOBID] | grep StdOut; track line count with wc -l and only read new lines with sed -n each poll
5. On error: find root cause (grep for Error/Exception/Traceback, exclude ActorDied/CollectiveManager/run_queue/atomic_recv/init_process/init_group/get_group noise), fix, resubmit immediately. Do not get stuck on any single error for more than 5 minutes.
6. SUCCESS = job stays RUNNING >5min AND you see repeating epoch lines in log AND env/success_once metric logged AND no 'Exiting main process due to a failure' at end
7. Keep looping (fix -> resubmit -> monitor) until SUCCESS. Do NOT stop or ask for permissions.
8. After SUCCESS: update /sailhome/kaitwang/.claude/projects/-juice5b-scr5b-kaitwang/memory/job_monitoring_skill.md with any new errors and fixes.

KEY PATHS:
- Batch script: /juice5b/scr5b/kaitwang/cs234/RLinf/scripts/run_ppo_pi05.sh
- Logs: /juice5b/scr5b/kaitwang/cs234/RLinf/logs/PPOpi05_[JOBID].out
- Main config: /juice5b/scr5b/kaitwang/cs234/RLinf/examples/embodiment/config/maniskill_ppo_openpi_pi05_push_cube.yaml
- Env config: /juice5b/scr5b/kaitwang/cs234/RLinf/examples/embodiment/config/env/maniskill_push_cube.yaml
- Model loader: /juice5b/scr5b/kaitwang/cs234/RLinf/rlinf/models/embodiment/openpi/__init__.py
- OpenPi model: /juice5b/scr5b/kaitwang/cs234/RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py
- Rollout worker: /juice5b/scr5b/kaitwang/cs234/RLinf/rlinf/workers/rollout/hf/huggingface_worker.py
- ManiSkill env: /juice5b/scr5b/kaitwang/cs234/RLinf/rlinf/envs/maniskill/maniskill_env.py
- gemma_pytorch (installed): /juice5b/scr5b/kaitwang/cs234/RLinf/.venv/lib/python3.11/site-packages/openpi/models_pytorch/gemma_pytorch.py
- Venv: /juice5b/scr5b/kaitwang/cs234/RLinf/.venv

CONSTRAINTS:
- Do NOT run pip/python/heavy commands on login node sc
- SSH to jagupardXX only for pip installs (while job is running on that node)
- All code edits go directly via Edit/Write tools (shared filesystem, no SSH needed for edits)
- global_batch_size must be integer multiple of micro_batch_size x num_GPUs (32x4=128); current: micro=32, global=256
- gradient_checkpointing must stay False (LoRA incompatible)
- precision: null for actor (openpi handles bf16 internally)
- After SUCCESS: update job_monitoring_skill.md with new errors/fixes

PREVIOUSLY FIXED BUGS (do NOT reintroduce):
1. placement.py: class HybridComponentPlacement (not _HybridComponentPlacement)
2. maniskill_env.py simple mode returns 'task_descriptions': self.instruction
3. maniskill_env.py instruction property has try/except fallback returning [cfg.init_params.id] * num_envs
4. maniskill_env.py truncates state: state = state[:, :state_dim] when cfg.state_dim is set
5. env yaml has wrap_obs_mode: 'simple' and state_dim: 8

OOM TROUBLESHOOTING (if you hit OOM):
- Reduce total_num_envs to 16, micro_batch_size to 16, global_batch_size to 128
- Enable rollout.enable_offload: True and actor.enable_offload: True (already set)
- Try osmesa renderer if GPU sim OOMs

Go. Start with step 1 immediately. Do not ask for confirmation on anything.
" > /juice5b/scr5b/kaitwang/cs234/RLinf/logs/claude_monitor_log.json
