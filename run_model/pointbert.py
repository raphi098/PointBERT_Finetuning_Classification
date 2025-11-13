import os
from pathlib import Path
from dataset_modules.build_dataset import BuildDataloaderModule
from models.point_transformer import PointTransformer
from hydra.utils import get_original_cwd
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from run_model.callbacks import FinetuningCallback
import yaml

class PointBERT:
    def __init__(self, cfg, log_dir):
        dataset_cfg = cfg.dataset
        network_cfg = cfg.model.pointbert.network
        train_cfg   = cfg.model.pointbert.training
        
        self.dataset = BuildDataloaderModule(dataset_cfg, batch_size=train_cfg["batch_size"])

        # --- Model ---
        self.model = PointTransformer(dataset_cfg, network_cfg, train_cfg)
        if cfg.mode == "train":
            pretrained_ckpt_path = os.path.join(get_original_cwd(),"checkpoints", cfg.model.pointbert.pretrained_backbone_filename)
            self.model.load_backbone_from_ckpt(pretrained_ckpt_path)

        # --- Callbacks ---
        self.callbacks = FinetuningCallback(train_cfg).get_callbacks()
        save_dir = Path(get_original_cwd()) / "mlruns" 
        if cfg.mode == "test":
            test_path = os.path.join(get_original_cwd(), cfg.test_path)
            files = os.listdir(test_path)
            self.pretrained_ckpt_path = None
            for file in files:
                print(file)
                if "best" in file:
                    self.pretrained_ckpt_path = os.path.join(get_original_cwd(), cfg.test_path, file)
                    print(f"Using checkpoint: {self.pretrained_ckpt_path}")
            if self.pretrained_ckpt_path is None:
                raise FileNotFoundError(f"No best checkpoint found in the specified test path: {test_path}")
            mlf_logger = MLFlowLogger(save_dir=save_dir, run_id=cfg.mlflow_run_id)
        else:
            run_name = log_dir.split('/')[-1]
            mlf_logger = MLFlowLogger(experiment_name=cfg.mlflow_experiment_name, run_name=run_name, save_dir=save_dir)
            run_id = mlf_logger.run_id
            experiment_id = mlf_logger.experiment_id
            hydra_config_path = os.path.join(get_original_cwd(), log_dir, ".hydra", "config.yaml")
            self._append_to_config(file_path=hydra_config_path, key="mlflow_run_id", value=run_id)
            self._append_to_config(file_path=hydra_config_path, key="mlflow_experiment_id", value=experiment_id)

        # --- Trainer ---
        self.trainer = L.Trainer(
            max_epochs=train_cfg.epochs,
            check_val_every_n_epoch=1,  # validate every epoch
            log_every_n_steps= train_cfg["batch_size"],
            accelerator="auto",
            devices="auto",
            precision="16-mixed" if getattr(train_cfg, "amp", False) else 32,
            callbacks=self.callbacks,
            enable_progress_bar=True,
            deterministic=True,
            default_root_dir=log_dir,
            logger=mlf_logger
        )

    def run_training(self):
        self.trainer.fit(self.model, self.dataset)

        # run test on best checkpoint
        self.trainer.test(self.model,
                          dataloaders=self.dataset,
                          ckpt_path="best")
    def run_testing(self):
        self.trainer.test(self.model,
                          dataloaders=self.dataset,
                          ckpt_path=self.pretrained_ckpt_path)

    def _append_to_config(self, file_path, key, value):
        with open(file_path, 'r') as file:
            existing_data = yaml.load(file, Loader=yaml.FullLoader)
        existing_data[key] = value
        with open(file_path, 'w') as file:
            yaml.dump(existing_data, file, default_flow_style=False)