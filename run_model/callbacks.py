import os
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

class FinetuningCallback():
    def __init__(self, train_config):
        self.ckpt_cb = ModelCheckpoint(
            dirpath=os.getcwd(),
            monitor="val/loss",           
            mode="min",
            save_last=True,
            save_top_k=1,
            filename="pt-{epoch:02d}-best-{val/loss:.4f}",
            auto_insert_metric_name=False
        )
        self.early_cb = EarlyStopping(
            monitor="val/loss",
            mode="min",
            patience=train_config.early_stop_patience
        )
        self.lr_cb = LearningRateMonitor(logging_interval="epoch")

    def get_callbacks(self):
        return [self.ckpt_cb, self.early_cb, self.lr_cb]

