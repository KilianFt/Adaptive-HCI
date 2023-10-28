import lightning.pytorch as pl
import torch
from torch.functional import F
from torchmetrics import ExactMatch, F1Score
from vit_pytorch import ViT


class EMGViT(ViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x: torch.Tensor):
        x.unsqueeze_(axis=1)
        return self.forward(x)


class PLModel(pl.LightningModule):
    def __init__(self, model, n_labels, lr, n_frozen_layers: int, threshold: float):
        super(PLModel, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.MSELoss()
        self.exact_match = ExactMatch(task="multilabel", num_labels=n_labels, threshold=threshold)
        self.f1_score = F1Score(task="multilabel", num_labels=n_labels, threshold=threshold)
        self.freeze_layers(n_frozen_layers)

    def freeze_layers(self, n_frozen_layers: int) -> None:
        # reset grad
        for param in self.model.parameters():
            param.requires_grad = True

        # freeze desired ones
        if n_frozen_layers >= 1:
            for param in self.model.to_patch_embedding.parameters():
                param.requires_grad = False
            for param in self.model.dropout.parameters():
                param.requires_grad = False

        # FIXME adjust for other models than ViT
        if n_frozen_layers >= 2:
            for layer_idx in range(min((n_frozen_layers - 1), len(self.model.transformer.layers))):
                for param in self.model.transformer.layers[layer_idx].parameters():
                    param.requires_grad = False

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)

        val_acc = self.exact_match(outputs, targets)
        val_f1 = self.f1_score(outputs, targets)
        val_loss = F.mse_loss(outputs, targets)
        self.log('validation/loss', val_loss, prog_bar=True)
        self.log('validation/acc', val_acc, prog_bar=True)
        self.log('validation/f1', val_f1, prog_bar=True)

        return val_loss, val_acc, val_f1

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
