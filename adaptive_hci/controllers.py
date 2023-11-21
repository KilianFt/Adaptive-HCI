import logging

import lightning.pytorch as pl
import torch
from torch import nn
from torch.functional import F
from torchmetrics import ExactMatch, F1Score, Accuracy
from vit_pytorch import ViT

from adaptive_hci.tcn import TCN
from common import GeneralModelEnum


def make_model(config) -> nn.Module:
    model_name = config.general_model_name
    if model_name == GeneralModelEnum.TCN:
        return EMGTCN(windows_size = config.window_size,
                      in_channels = config.num_channels,
                      out_channels = config.general_model_config.out_channels,
                      kernel_size = config.general_model_config.kernel_size,
                      channels = config.general_model_config.channels,
                      layers = config.general_model_config.layers,
                      bias = config.general_model_config.bias,
                      class_hidden_size = config.general_model_config.classification_hidden_size,
                      class_n_layers = config.general_model_config.classification_n_layers,
                      output_size = config.num_classes)
    
    elif model_name == GeneralModelEnum.ViT:
        return EMGViT(image_size=config.window_size,
                      patch_size=config.general_model_config.patch_size,
                      num_classes=config.num_classes,
                      dim=config.general_model_config.dim,
                      depth=config.general_model_config.depth,
                      heads=config.general_model_config.heads,
                      mlp_dim=config.general_model_config.mlp_dim,
                      dropout=config.general_model_config.dropout,
                      emb_dropout=config.general_model_config.emb_dropout,
                      channels=config.general_model_config.channels,)

    elif model_name == GeneralModelEnum.MLP:
        hidden_sizes = [config.general_model_config.hidden_size for _ in range(config.general_model_config.n_layers)]
        return MLP(input_size=(config.window_size * config.num_channels),
                   hidden_sizes=hidden_sizes,
                   output_size=config.num_classes)
    
    else:
        logging.error("Count not find model {model_name}")


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU(inplace=True))

        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def freeze_layers(self, n_frozen_layers: int) -> None:
        pass


class EMGTCN(nn.Module):
    def __init__(self, windows_size, in_channels, out_channels, kernel_size, channels, layers,
            bias, class_hidden_size, class_n_layers, output_size):
        super(EMGTCN, self).__init__()
        self.tcn = TCN(in_channels = in_channels,
                       out_channels = out_channels,
                       kernel_size = kernel_size,
                       channels = channels,
                       layers = layers,
                       bias = bias,
                       fwd_time = True)
        
        mlp_input_size = out_channels *  windows_size
        hidden_sizes = [class_hidden_size for _ in range(class_n_layers)]

        self.classification_head = MLP(input_size=mlp_input_size, hidden_sizes=hidden_sizes,
                                       output_size=output_size)
        self.out_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tcn(x)
        x = self.classification_head(x.flatten(start_dim=1))
        out = self.out_activation(x)
        return out

    def freeze_layers(self, n_frozen_layers: int) -> None:
        pass


class EMGViT(ViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x: torch.Tensor):
        x.unsqueeze_(axis=1)
        return self.forward(x)

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
        self.step_count = 0
        if n_frozen_layers is not None:
            self.model.freeze_layers(n_frozen_layers)

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        self.log(f"{self.metric_prefix}train/loss", loss)
        self.log(f"{self.metric_prefix}train/step", self.step_count)
        self.step_count += 1
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

        # FIXME remove 'Rest' class for validation?
        val_acc = self.exact_match(outputs, targets)
        val_f1 = self.f1_score(outputs, targets)
        val_loss = F.mse_loss(outputs, targets)
        self.log(f'{self.metric_prefix}validation/loss', val_loss, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/acc', val_acc, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/f1', val_f1, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/step', self.step_count, prog_bar=True)

        per_label_accuracies = self.get_per_label_accuracies(outputs, targets)
        for label_idx, per_label_acc in enumerate(per_label_accuracies):
            self.log(f'{self.metric_prefix}validation/acc_label_{label_idx}', per_label_acc, prog_bar=True)

        return val_loss, val_acc, val_f1, per_label_accuracies

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
