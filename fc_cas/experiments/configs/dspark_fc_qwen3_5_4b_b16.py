import os

from deepspec.trainer.dspark_trainer import Qwen3_5DSparkTrainer

# FC-CAS domain draft — larger block for patent-friendly τ gap (v1).
BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "deepspec"
exp_name = "dspark_fc_cas_qwen3_5_4b_b16"
seed = 42

model = dict(
    target_model_name_or_path="/home/caden/models/Qwen3_5-4B",
    block_size=16,
    num_draft_layers=5,
    target_layer_ids=[1, 8, 16, 23, 30],
    mask_token_id=248063,
    num_anchors=64,
    markov_rank=256,
    markov_head_type="vanilla",
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=Qwen3_5DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=32,
    num_train_epochs=20,
    max_train_steps=200,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=100,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=2048,
    num_workers=2,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    logging_cfg["wandb_project"] = project_name
    logging_cfg["wandb_name"] = exp_name
    cfg["logging"] = logging_cfg
    return cfg
