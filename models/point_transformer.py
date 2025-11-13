import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchmetrics import JaccardIndex
from torchmetrics.classification import MulticlassAveragePrecision,MulticlassAccuracy,MulticlassConfusionMatrix, MulticlassRecall, MulticlassF1Score, BinaryConfusionMatrix, BinaryAveragePrecision, BinaryRecall, BinaryF1Score, BinaryAccuracy
import lightning as L
from models.utils import get_missing_parameters_message, get_unexpected_parameters_message, add_weight_decay
from models.modules import Group, Encoder, TransformerEncoder
from lightning.pytorch.loggers import MLFlowLogger

class PointTransformer(L.LightningModule):
    def __init__(self, dataset_cfg, network_cfg, train_cfg, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.train_cfg = train_cfg

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
        if self.num_classes == 2:
            self.precision = BinaryAveragePrecision()
            self.recall = BinaryRecall()
            self.accuracy = BinaryAccuracy()    
            self.f1_score = BinaryF1Score()
        else:
            self.precision = MulticlassAveragePrecision(num_classes=self.num_classes, average="weighted")
            self.recall = MulticlassRecall(num_classes=self.num_classes, average="weighted")
            self.accuracy = MulticlassAccuracy(num_classes=self.num_classes, average="weighted")
            self.f1_score = MulticlassF1Score(num_classes=self.num_classes, average="weighted")
            
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, train=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, train=False)
        return loss

    def _common_step(self, batch, train=True):
        points, class_label = batch
        # dtypes
        class_label  = class_label
        
        # forward
        class_pred = self.forward(points)  # [B, C] log-probs
        label_id = class_pred.argmax(dim=1)   # [B, N]
         
        # metrics
        metrics = {}

        metrics["train/acc" if train else "val/acc"]   = (label_id == class_label).float().mean()
        # loss
        class_label_flat = class_label.reshape(-1)             # [B]
 
        metrics["train/loss" if train else "val/loss"] = self.loss_ce(class_pred, class_label_flat)       # [B, C] logits

        for key, value in metrics.items():
            self.log(key, value, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return metrics["train/loss" if train else "val/loss"]
    
    def test_step(self, batch, batch_idx):
        points, class_label = batch  # [B, N, C], [B, N], [B]

        # dtypes
        class_label  = class_label

        # forward
        class_pred, _ = self.forward(points)          # [B, N, C] logits
  
        # metrics

        class_pred_flat     = class_pred.reshape(-1, self.num_classes)

        #metrics
        metrics = {}
        metrics["test/acc"]   = self.accuracy(class_pred_flat, class_label)
        metrics["test/f1_score"] = self.f1_score(class_pred_flat, class_label)
        metrics["test/precision"] = self.precision(class_pred_flat, class_label)
        metrics["test/recall"] = self.recall(class_pred_flat, class_label)
        metrics["test/loss"] = self.loss_ce(class_pred_flat, class_label)
        # logging
        for key, value in metrics.items():
            self.log(f"test/{key}", value, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"test_loss": metrics["test/loss"]}
    
    def _get_mlf_logger(self):
        # trainer can hold multiple loggers; pick the MLflow one
        if isinstance(self.trainer.logger, MLFlowLogger):
            return self.trainer.logger
        for lg in self.trainer.loggers if isinstance(self.trainer.loggers, (list, tuple)) else []:
            if isinstance(lg, MLFlowLogger):
                return lg
        return None
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