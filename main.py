import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
import torch

# torch.autograd.set_detect_anomaly(True)
from torch.distributed import init_process_group, destroy_process_group
import os
from config import MyConfig
from trainer import Trainer
from datetime import timedelta

cs = ConfigStore.instance()
cs.store(name="my_config", node=MyConfig)


def ddp_setup():
    print(f"-------------- Setting up ddp {os.environ['LOCAL_RANK']}...")
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


@hydra.main(version_base=None, config_path="configs", config_name="debug")
def main(cfg: MyConfig) -> None:
    use_ddp = (cfg.hardware.multi_gpus == "ddp") and torch.cuda.is_available()

    if use_ddp:
        ddp_setup()

    num_gpus = torch.cuda.device_count()
    ## add num_gpus in hardward config
    if "num_gpus" not in cfg["hardware"]:
        OmegaConf.update(cfg, "hardware.num_gpus", num_gpus, force_add=True)

    trainer = Trainer(cfg)

    print("starting trainer.train()...")
    trainer.train()

    if use_ddp:
        destroy_process_group()


if __name__ == "__main__":
    main()
