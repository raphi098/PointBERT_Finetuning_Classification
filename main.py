from omegaconf import OmegaConf

# register resolvers at import time
OmegaConf.register_new_resolver(
    "floordiv", lambda a, b: int(a) // int(b), use_cache=True
)
import hydra
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from omegaconf import DictConfig
from run_model.pointbert import PointBERT
import os
from utils.misc import set_random_seed
import sys

@hydra.main(version_base="1.1", config_path="config", config_name="main")
def main(cfg:DictConfig):
    log_dir = os.getcwd()
    print(f"Logging to: {log_dir}")

    if cfg.mode == "test":
        current_dir = get_original_cwd()
        test_path = cfg.test_path
        cfg = OmegaConf.load(os.path.join(current_dir, test_path, ".hydra", "config.yaml"))
        OmegaConf.set_struct(cfg, True)
        cfg["mode"] = "test"
        cfg["test_path"] = test_path

    if cfg.mode == "train":
        current_dir = get_original_cwd()
        test_path = cfg.test_path

    print(OmegaConf.to_yaml(cfg))
    set_random_seed(42, deterministic=True)
    pointbert = PointBERT(cfg, log_dir)

    if cfg.mode == "train":
        pointbert.run_training()
    elif cfg.mode == "test":
        pointbert.run_testing()

if __name__ == "__main__":
    with initialize(version_base=None, config_path="config", job_name="test_app"):
        cfg = compose(config_name="main", overrides=sys.argv[1:],return_hydra_config=True)
        run_dir = cfg.hydra.run.dir 

    if cfg.mode == "test":
        sys.argv.extend([f'hydra.run.dir={cfg.test_path}', 
                 'hydra.output_subdir=null', 
                 'hydra/hydra_logging=disabled', 
                 'hydra/job_logging=disabled'])
                 
    if cfg.mode == "train" and os.path.exists(run_dir):
        raise ValueError(f"Output directory {run_dir} already exists. Please choose a different directory or remove the existing one.")
    main()