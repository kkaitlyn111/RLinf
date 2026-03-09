from typing import Optional, List, Tuple
from dataclasses import dataclass
import subprocess
import shlex
from datetime import datetime
from pathlib import Path

import submitit
from submitit.core.core import Job


@dataclass
class ResourceConfig:
    """
    Dataclass for defining the resources that worker uses
    """

    log_dir: str
    # Slurm
    account: str = "nlp"
    partition: str = "sphinx"
    gres: str = "gpu:0"
    mem: str = "128G"
    time: str = "12:00:00"
    cpus_per_task: int = 8
    exclude: str = ""
    constraints: Optional[str] = None
    # Parallelism
    jobs_per_node: int = 1  # If job can be split into parallel jobs on the cluster side
    # ^ Required for distributed training, set this to number of gpus per node
    # Environment variables
    node_list: Optional[str] = None
    exclusive: bool = False


# ---------------------------------
# Submitit utils
# ---------------------------------


def time_string_to_minutes(time_str: str) -> int:
    """
    Convert time string in format 'DD-HH:MM:SS' or 'HH:MM:SS' to minutes.

    We use this for submitit because it doesn't like DD-HH:MM:SS

    Args:
        time_str: Time string in format 'DD-HH:MM:SS' or 'HH:MM:SS'
    Returns:
        Total minutes as integer
    """
    if not time_str:
        return 0

    # Handle DD-HH:MM:SS format
    if "-" in time_str:
        days_part, time_part = time_str.split("-", 1)
        days = int(days_part)
    else:
        days = 0
        time_part = time_str

    # Parse HH:MM:SS
    time_components = time_part.split(":")
    if len(time_components) == 3:
        hours, minutes, _ = map(int, time_components)
    elif len(time_components) == 2:
        hours, minutes = map(int, time_components)
    else:
        raise ValueError(f"Invalid time format: {time_str}")

    # Convert to total minutes
    total_minutes = days * 24 * 60 + hours * 60 + minutes

    return total_minutes


def get_submitit_executor(
    resource_config: ResourceConfig,
    log_dir: Optional[str] = None,
) -> submitit.AutoExecutor:
    """
    Get submitit executor from reousrced_config.

    If lop_dir is None, then resource_config.log_dir is used.
    """

    # Get the path that Python is being run from
    # calling_dir = os.getcwd()

    # convert time to minutes as submitit premting doesn't support time strings
    time = time_string_to_minutes(resource_config.time)

    if log_dir is None:
        log_dir = resource_config.log_dir

    executor = submitit.AutoExecutor(folder=log_dir)
    executor.update_parameters(
        slurm_account=resource_config.account,
        slurm_partition=resource_config.partition,
        slurm_gres=resource_config.gres,
        slurm_mem=resource_config.mem,
        slurm_time=time,
        slurm_cpus_per_task=resource_config.cpus_per_task,
        slurm_exclude=resource_config.exclude,
        slurm_ntasks_per_node=resource_config.jobs_per_node,
        slurm_constraint=resource_config.constraints,
    )

    if resource_config.node_list is not None:
        executor.update_parameters(
            slurm_nodelist=resource_config.node_list,
        )

    if resource_config.exclusive:
        executor.update_parameters(
            slurm_exclusive=True,
        )

    return executor


def cleanup_submitit_job(job: Job):
    """
    Job will be cancelled and associated files destroyed. If Job is already cancelled then nothing
    will happen.

    This means after running log files will be destroyed.
    """

    path: Path
    for path in [
        job.paths.stderr,
        job.paths.stdout,
        job.paths.submission_file,
        job.paths.submitted_pickle,
        job.paths.result_pickle,
    ]:
        if path.exists():
            path.unlink()


def get_slurm_job_start_and_end_times(
    job_id: str,
) -> List[Tuple[str, float, Optional[float]]]:
    """
    Returns a list of (state, start_ts, end_ts) per attempt (chronological).
    Timestamps are Unix seconds as floats. end_ts is None if not finished yet.

    The return here is a list because if a single slurm job is requeued there
    will be multiple start and end times.
    """

    out = (
        subprocess.check_output(
            shlex.split(f"sacct -j {job_id} -n -P -X -D -o JobID,State,Start,End"),
            text=True,
        )
        .strip()
        .splitlines()
    )

    attempts = []
    for line in out:
        jid, state, start, end = (line.split("|") + ["", "", "", ""])[:4]
        if jid != job_id:
            continue

        def to_ts(s: str) -> Optional[float]:
            if not s or s in ("Unknown", "N/A"):
                return None
            # sacct gives ISO-like "YYYY-MM-DDTHH:MM:SS"
            return datetime.fromisoformat(s).timestamp()

        start_ts = to_ts(start)
        if start_ts is None:
            continue  # skip attempts without a real start yet
        end_ts = to_ts(end)  # None if still running or not recorded

        attempts.append((state, start_ts, end_ts))

    attempts.sort(key=lambda x: x[1])  # chronological by start
    return attempts


# ---------------------------------
# Training job
# ---------------------------------

def train():
    import os
    import subprocess

    os.environ["TMPDIR"] = "/juice5b/scr5b/kaitwang/tmp"
    os.environ["RAY_TMPDIR"] = "/juice5b/scr5b/kaitwang/tmp/ray"
    os.environ["UV_CACHE_DIR"] = "/juice5b/scr5b/kaitwang/.uv_cache"
    os.environ["RAY_ADDRESS"] = "local"

    repo = "/juice5b/scr5b/kaitwang/cs234/RLinf"
    venv_bin = f"{repo}/.venv/bin"
    env = {**os.environ, "PATH": f"{venv_bin}:{os.environ['PATH']}"}

    # Install openpi + maniskill deps into the venv if not already present
    subprocess.run(
        ["bash", "requirements/install.sh", "embodied", "--model", "openpi", "--env", "maniskill_libero"],
        cwd=repo,
        env=env,
        check=True,
    )

    subprocess.run(
        ["bash", "examples/embodiment/run_embodiment.sh", "maniskill_ppo_openpi_pi05_push_cube"],
        cwd=repo,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    resource_config = ResourceConfig(
        log_dir="/juice5b/scr5b/kaitwang/cs234/RLinf/logs/PPOpi05subm",
        account="nlp",
        partition="jag-standard",
        gres="gpu:4",
        mem="80G",
        time="7-00:00:00",
        cpus_per_task=20,
        constraints="48G",
    )

    executor = get_submitit_executor(resource_config)
    job = executor.submit(train)
    print(f"Submitted job {job.job_id}")