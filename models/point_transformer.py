import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassAveragePrecision,MulticlassAccuracy,MulticlassConfusionMatrix, MulticlassRecall, MulticlassF1Score, BinaryConfusionMatrix, BinaryAveragePrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy
import lightning as L
from models.utils import get_missing_parameters_message, get_unexpected_parameters_message, add_weight_decay
from models.modules import Group, Encoder, TransformerEncoder
from lightning.pytorch.loggers import MLFlowLogger
import json
import os
import matplotlib.pyplot as plt
from hydra.utils import get_original_cwd
from torchmetrics import MetricCollection

class PointTransformer(L.LightningModule):
    def __init__(self, dataset_cfg, network_cfg, train_cfg, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.train_cfg = train_cfg
        self.dataset_cfg = dataset_cfg

        self.trans_dim = network_cfg.trans_dim
        self.depth = network_cfg.depth 
        self.drop_path_rate = network_cfg.drop_path_rate 
        self.num_classes = network_cfg.num_classes 
        self.num_heads = network_cfg.num_heads 
        self.group_size = network_cfg.group_size
        self.num_group = network_cfg.num_group
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dims =  network_cfg.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Linear(self.encoder_dims,  self.trans_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
            nn.Linear(self.trans_dim * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, self.num_classes)
        )

        self.loss_ce = nn.CrossEntropyLoss()
        dataset_path = self.dataset_cfg.data_path
        category_map_path = os.path.join(get_original_cwd(), dataset_path, "cat_dict.json")
        self.class_names = []
        with open(category_map_path, 'r') as f:
            cat_dict = json.load(f)
            for key in cat_dict.keys():
                self.class_names.append((cat_dict[key], key))
        
        self.val_metrics_dict = {}

        if self.num_classes == 2:
            self.val_metrics_dict["precision"] = BinaryAveragePrecision()
            self.val_metrics_dict["recall"] = BinaryRecall()
            self.val_metrics_dict["accuracy"] = BinaryAccuracy()    
            self.val_metrics_dict["f1_score"] = BinaryF1Score()
            self.confusion_matrix = BinaryConfusionMatrix()
        else:
            self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_classes)
            self.val_metrics_dict["precision"] = MulticlassAveragePrecision(num_classes=self.num_classes, average="weighted")
            self.val_metrics_dict["recall"] = MulticlassRecall(num_classes=self.num_classes, average="weighted")
            self.val_metrics_dict["accuracy"] = MulticlassAccuracy(num_classes=self.num_classes, average="weighted")
            self.val_metrics_dict["f1_score"] = MulticlassF1Score(num_classes=self.num_classes, average="weighted")

        self.val_metrics = MetricCollection(self.val_metrics_dict)
        self.test_metrics_dict = self.val_metrics_dict.copy()
        self.test_metrics_dict["confusion_matrix"] = self.confusion_matrix
        self.test_metrics = MetricCollection(self.test_metrics_dict)

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, train=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, train=False)
        return {"val/loss": loss}

    def _common_step(self, batch, train=True):
        points, class_label = batch
        # dtypes
        class_label  = class_label
        
        # forward
        class_pred = self.forward(points)  # [B, C] log-probs
        label_id = class_pred.argmax(dim=1)   # [B, N]
        class_label_flat = class_label.reshape(-1)             # [B]
        class_pred_flat  = class_pred.reshape(-1, self.num_classes)  # [B, C]

        # metrics
        loss = {}
        loss["train/loss" if train else "val/loss"] = self.loss_ce(class_pred_flat, class_label_flat)
        self.log("train/loss" if train else "val/loss", loss["train/loss" if train else "val/loss"])       # [B, C] logits

        if not train:
            for name, metric in self.val_metrics.items():
                metric.update(class_pred_flat, class_label_flat)
                self.log(f"val/{name}", metric, on_step=False, on_epoch=True)

        return loss["train/loss" if train else "val/loss"]

    def test_step(self, batch, batch_idx):
        points, class_label = batch  # [B, N, C], [B, N], [B]

        # forward
        class_pred, _ = self.forward(points)          # [B, N, C] logits
        class_pred_flat     = class_pred.reshape(-1, self.num_classes)
        class_label_flat    = class_label.reshape(-1)
        
        for name, metric in self.test_metrics.items():
            metric.update(class_pred_flat, class_label_flat)
            self.log(f"test/{name}", metric, on_step=False, on_epoch=True)

        # logging
        for key, value in metrics.items():
            self.log(f"test/{key}", value, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        return {"test_loss": metrics["test/loss"]}

    def on_test_epoch_end(self):
        cm = self.confusion_matrix.compute()
        fig, ax = self.confusion_matrix.plot(cm, labels=self.class_names, cmap="Blues")
        fig.set_size_inches(10, 10)
        ax.tick_params(axis='both', labelsize=14)

        # Get attached MLFlow logger
        mlf_logger = self._get_mlf_logger()
        if mlf_logger is not None:
            mlf_logger.experiment.log_figure(
                figure=fig,
                artifact_file=f"confusion_matrix/test_confusion_matrix_.png",
                run_id=mlf_logger.run_id
            )

        # Save locally as well
        plt.savefig(f"test_confusion_matrix.png")
        plt.close(fig)
        self.confusion_matrix.reset()

    def load_backbone_from_ckpt(self, bert_ckpt_path):
        ckpt = torch.load(bert_ckpt_path, weights_only=False)
        print("Checkpoint keys:", ckpt.keys())
        if 'base_model' in ckpt:
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        else:
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
            elif k.startswith('base_model'):
                base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            del base_ckpt[k]

        incompatible = self.load_state_dict(base_ckpt, strict=False)

        if incompatible.missing_keys:
            print('missing_keys')
            print(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            print('unexpected_keys')
            print(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

        print(f'[PointTransformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, pts):
        # divide the point clo  ud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  #  B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)  
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)  
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:,0], x[:, 1:].max(1)[0]], dim = -1)
        ret = self.cls_head_finetune(concat_f)
        return ret
    
    
    def configure_optimizers(self):
        # No cosine optimizer with warmup in torch
        param_groups = add_weight_decay(
            self,
            weight_decay=self.train_cfg.decay_rate
        )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.train_cfg.lr
        )

        warmup_epochs = self.train_cfg.warmup_epochs
        total_epochs = self.train_cfg.epochs

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-6 / self.train_cfg.lr,  # starting LR relative to base LR
            total_iters=warmup_epochs
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=1e-6
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }    

    def _get_mlf_logger(self):
        # trainer can hold multiple loggers; pick the MLflow one
        if isinstance(self.trainer.logger, MLFlowLogger):
            return self.trainer.logger
        for lg in self.trainer.loggers if isinstance(self.trainer.loggers, (list, tuple)) else []:
            if isinstance(lg, MLFlowLogger):
                return lg
        return None