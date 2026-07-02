import os
import time
from typing import Optional

from torch.utils.tensorboard import SummaryWriter

from deepspec.utils import ensure_dir, is_global_main_process, print_on_global_main
from deepspec.utils.metrics import add_metric, flush, reset

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


_writer: Optional[SummaryWriter] = None
_logging_steps: int = 1
_session_start_wall: Optional[float] = None
_session_start_step: int = 0
_wandb_enabled: bool = False


def init(
    *,
    logging_steps: int,
    tensorboard_dir: Optional[str] = None,
    wandb_project: Optional[str] = None,
    wandb_name: Optional[str] = None,
    wandb_config: Optional[dict] = None,
) -> None:
    global _writer, _logging_steps, _wandb_enabled
    _logging_steps = int(logging_steps)
    if is_global_main_process():
        if tensorboard_dir is not None:
            ensure_dir(tensorboard_dir)
            _writer = SummaryWriter(tensorboard_dir)
        if _WANDB_AVAILABLE and wandb_project is not None and os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true"):
            _wandb.init(
                project=wandb_project,
                name=wandb_name,
                config=wandb_config or {},
                resume="allow",
            )
            _wandb_enabled = True


def start_session(*, global_step: int) -> None:
    global _session_start_wall, _session_start_step
    reset()
    _session_start_wall = time.time()
    _session_start_step = int(global_step)


def on_optimizer_step(
    *,
    global_step: int,
    next_micro_step: int,
    micro_batches_per_epoch: int,
    max_train_steps: int,
    learning_rate: float,
    grad_norm: float,
):
    add_metric("lr", learning_rate, reduction="last", tag="train")
    add_metric("grad_norm", grad_norm, reduction="last", tag="train")

    if global_step % _logging_steps != 0:
        return None

    summary = flush()
    if is_global_main_process():
        _write_scalars(summary, global_step=global_step)
        _print_summary(
            summary=summary,
            global_step=global_step,
            next_micro_step=next_micro_step,
            micro_batches_per_epoch=micro_batches_per_epoch,
            max_train_steps=max_train_steps,
        )
    return summary


def close() -> None:
    global _writer, _wandb_enabled
    if _writer is not None:
        _writer.close()
        _writer = None
    if _wandb_enabled and _WANDB_AVAILABLE:
        _wandb.finish()
        _wandb_enabled = False


def _write_scalars(summary, *, global_step: int) -> None:
    if _writer is not None:
        for key, value in summary.items():
            _writer.add_scalar(key, value, global_step)
    if _wandb_enabled and _WANDB_AVAILABLE:
        _wandb.log({k: float(v) for k, v in summary.items()}, step=global_step)


def _print_summary(
    *,
    summary,
    global_step: int,
    next_micro_step: int,
    micro_batches_per_epoch: int,
    max_train_steps: int,
) -> None:
    session_start_wall = _session_start_wall
    if session_start_wall is None:
        session_start_wall = time.time()
    current_epoch = next_micro_step // micro_batches_per_epoch + 1
    session_elapsed = time.time() - session_start_wall
    completed_session_steps = global_step - _session_start_step
    remaining_steps = max(max_train_steps - global_step, 0)
    remaining_min = (
        session_elapsed * remaining_steps / max(completed_session_steps, 1)
    ) / 60
    loss_text = ""
    if "train/loss" in summary:
        loss_text = f" loss={summary['train/loss']:.4f}"
    print_on_global_main(
        f"epoch={current_epoch} "
        f"step={global_step}/{max_train_steps}"
        f"{loss_text} "
        f"| elapsed={session_elapsed / 60:.1f}min"
        f" | remaining={remaining_min:.1f}min"
    )
