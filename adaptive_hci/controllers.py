import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics import ExactMatch, F1Score, Accuracy
from vit_pytorch import ViT

from adaptive_hci.encoders import ENCODER_MODELS

from common import EncoderModelEnum # FIXME

class EMGViT(ViT):
    def __init__(self, config, *args, **kwargs):
        super().__init__(image_size=config.window_size,
            patch_size=config.general_model.patch_size,
            num_classes=config.num_classes,
            dim=config.general_model.dim,
            depth=config.general_model.depth,
            heads=config.general_model.heads,
            mlp_dim=config.general_model.mlp_dim,
            dropout=config.general_model.dropout,
            emb_dropout=config.general_model.emb_dropout,
            channels=config.general_model.channels,
            *args, **kwargs)

    def __call__(self, observations: torch.Tensor, actions: torch.Tensor):
        # TODO only consider last action?
        observations.unsqueeze_(axis=1)
        return self.forward(observations)

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


class ContextClassifier(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super(ContextClassifier, self).__init__(*args, **kwargs)

        self.append_action = config.append_action

        # TODO add option to only condition on states?
        hist_input_size = config.feat_size + config.num_classes
        if config.encoder_name is not None:
            # FIXME
            # self.history_encoder = ENCODER_MODELS[config.encoder_name](hist_input_size, config.encoder)
            self.history_encoder = ENCODER_MODELS[EncoderModelEnum.TCN](hist_input_size, config.encoder)
        else:
            self.history_encoder = None

        class_input_size = hist_input_size + config.feat_size

        if self.append_action:
            class_input_size += config.num_classes

        self.classification_head = MLP(input_size=class_input_size,
                                       hidden_sizes=config.general_model.hidden_sizes,
                                       output_size=config.num_classes)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor):

        if self.history_encoder is not None:
            hist = torch.cat([observations, actions], dim=1)
            hist_state = self.history_encoder(hist) # output: [batch, seq_len, feat_size+num_actions]

            # TODO what is the hist_state? do I just want to take the last? or mean?
            concat_state = torch.cat([hist_state[:,:,-1:], observations[:,:,-1:]], dim=1)
        else:
            concat_state = observations[:,:,-1:]

        # only consider last observation and action in classification [batch, hist_size+feat_size+num_actions, 1]
        if self.append_action:
            classification_input = torch.cat([concat_state, actions[:,:,-1:]], dim=1)            
        else:
            classification_input = concat_state

        out = self.classification_head(classification_input.squeeze(dim=2))  # [B, num_actions]
        return out
    
    def freeze_layers(self, n_frozen_layers: int) -> None:
        print('WARN: not implemeneted')


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
