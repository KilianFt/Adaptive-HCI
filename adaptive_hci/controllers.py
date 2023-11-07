import re
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics import ExactMatch, F1Score, Accuracy
from vit_pytorch import ViT


class EMGViT(ViT):
    def __init__(self, config, *args, **kwargs):
        super().__init__(image_size=config.window_size,
            patch_size=config.base_model.patch_size,
            num_classes=config.num_classes,
            dim=config.base_model.dim,
            depth=config.base_model.depth,
            heads=config.base_model.heads,
            mlp_dim=config.base_model.mlp_dim,
            dropout=config.base_model.dropout,
            emb_dropout=config.base_model.emb_dropout,
            channels=config.base_model.channels,
            *args, **kwargs)

    def __call__(self, x: torch.Tensor):
        x.unsqueeze_(axis=1)
        return self.forward(x)

    def freeze_layers(self, n_frozen_layers: int) -> None:
        # reset grad
        for param in self.parameters():
            param.requires_grad = True

        # freeze desired ones
        if n_frozen_layers >= 1:
            for param in self.to_patch_embedding.parameters():
                param.requires_grad = False
            for param in self.dropout.parameters():
                param.requires_grad = False

        if n_frozen_layers >= 2:
            for layer_idx in range(min((n_frozen_layers - 1), len(self.transformer.layers))):
                for param in self.transformer.layers[layer_idx].parameters():
                    param.requires_grad = False


class EMGLSTM(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # TODO feature encoder?
        # TODO input last action?

        self.lstm = nn.LSTM(config.base_model.input_size, config.base_model.hidden_size,
                            config.base_model.num_layers, dropout=config.base_model.dropout,
                            batch_first=True)
        self.fc = nn.Linear(config.base_model.hidden_size, config.num_classes)

        # TODO check out how to best initialize weights

    def __call__(self, x: torch.Tensor):
        x, _ = self.lstm(x.swapaxes(1,2))
        out = self.fc(x[:, -1, :])
        return out

    def freeze_layers(self, n_frozen_layers: int) -> None:
        for param in self.lstm.named_parameters():
            param[1].requires_grad = False

        if n_frozen_layers > 0:
            pattern = re.compile(r'(weight|bias)_(ih|hh)_l[0-{0}]'.format((n_frozen_layers-1)))

            if n_frozen_layers > self.lstm.num_layers:
                print('WARN: trying to freeze more layers than present')
                n_frozen_layers = self.lstm.num_layers

            for param in self.lstm.named_parameters():
                if pattern.match(param[0]):
                    param[1].requires_grad = False


class PLModel(pl.LightningModule):
    def __init__(self, model, n_labels, lr, n_frozen_layers: int, threshold: float, metric_prefix: str = ''):
        super(PLModel, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.threshold = threshold
        self.metric_prefix = metric_prefix
        self.criterion = torch.nn.MSELoss()
        self.exact_match = ExactMatch(task="multilabel", num_labels=n_labels, threshold=threshold)
        self.f1_score = F1Score(task="multilabel", num_labels=n_labels, threshold=threshold)
        self.accuracy_metric = Accuracy(task='binary')

        if n_frozen_layers is not None:
            self.model.freeze_layers(n_frozen_layers)



    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        self.log(f"{self.metric_prefix}train/loss", loss)
        return loss

    def get_per_label_accuracies(self, outputs, targets):
        num_targets = targets.shape[1]

        binary_outputs = (outputs >= self.threshold).int()
        binary_targets = (targets >= self.threshold).int()

        per_labels_accuracies = []

        for label_idx in range(num_targets):
            label_acc = self.accuracy_metric(binary_outputs[:,label_idx], binary_targets[:,label_idx])
            per_labels_accuracies.append(label_acc)

        return torch.tensor(per_labels_accuracies)

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)

        val_acc = self.exact_match(outputs, targets)
        val_f1 = self.f1_score(outputs, targets)
        val_loss = F.mse_loss(outputs, targets)
        self.log(f'{self.metric_prefix}validation/loss', val_loss, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/acc', val_acc, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/f1', val_f1, prog_bar=True)

        per_label_accuracies = self.get_per_label_accuracies(outputs, targets)
        for label_idx, per_label_acc in enumerate(per_label_accuracies):
            self.log(f'{self.metric_prefix}validation/acc_label_{label_idx}', per_label_acc, prog_bar=True)

        return val_loss, val_acc, val_f1, per_label_accuracies

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
