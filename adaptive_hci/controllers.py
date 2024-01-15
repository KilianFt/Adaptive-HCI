import math

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.functional import F

import lightning.pytorch as pl
from torchmetrics import ExactMatch, F1Score, Accuracy
from vit_pytorch import ViT


class EMGViT(ViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x: torch.Tensor):
        x.unsqueeze_(axis=1)
        return self.forward(x)


def get_criterion(criterion_key):
    print(f"Using {criterion_key} loss")
    # how do we deal with keys? can we use enum in sweep config?
    if criterion_key == 'mse':
        return torch.nn.MSELoss()
    elif criterion_key == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"{criterion_key} loss not supported")


def multilabel_at_least_one_match(y_true, y_pred):
    ''' Is correct if at least one label is predicted'''
    intersection = (y_true * y_pred).sum(dim=-1)
    at_least_one_correct = (intersection > 0).float()
    return at_least_one_correct.mean()


class PLModel(pl.LightningModule):
    def __init__(self, model, n_labels, lr, n_frozen_layers: int, threshold: float, metric_prefix: str = '',
                 criterion_key: str = 'bce'):
        super(PLModel, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.threshold = threshold
        self.metric_prefix = metric_prefix
        self.criterion = get_criterion(criterion_key)
        self.exact_match = ExactMatch(task="multilabel", num_labels=n_labels, threshold=threshold)
        self.f1_score = F1Score(task="multilabel", num_labels=n_labels, threshold=threshold)
        self.accuracy_metric = Accuracy(task='binary')
        self.step_count = 0
        if n_frozen_layers is not None:
            self.freeze_layers(n_frozen_layers)

    def freeze_layers(self, n_frozen_layers: int) -> None:
        # reset grad
        # this is fine for ViT as by default all params require grad
        for param in self.model.parameters():
            param.requires_grad = True

        # freeze desired ones
        if n_frozen_layers >= 1:
            for param in self.model.to_patch_embedding.parameters():
                param.requires_grad = False
            for param in self.model.dropout.parameters():
                param.requires_grad = False

        if n_frozen_layers >= 2:
            for layer_idx in range(min((n_frozen_layers - 1), len(self.model.transformer.layers))):
                for param in self.model.transformer.layers[layer_idx].parameters():
                    param.requires_grad = False

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
            label_acc = self.accuracy_metric(binary_outputs[:, label_idx], binary_targets[:, label_idx])
            per_labels_accuracies.append(label_acc)

        return torch.tensor(per_labels_accuracies)

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)

        val_acc = self.exact_match(outputs, targets)
        val_f1 = self.f1_score(outputs, targets)
        one_match = multilabel_at_least_one_match(y_true=targets, y_pred=outputs)
        val_loss = F.mse_loss(outputs, targets)
        self.log(f'{self.metric_prefix}validation/loss', val_loss, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/acc', val_acc, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/f1', val_f1, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/one_match', one_match, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/step', self.step_count, prog_bar=True)

        per_label_accuracies = self.get_per_label_accuracies(outputs, targets)
        for label_idx, per_label_acc in enumerate(per_label_accuracies):
            self.log(f'{self.metric_prefix}validation/acc_label_{label_idx}', per_label_acc, prog_bar=True)

        return val_loss, val_acc, val_f1, per_label_accuracies

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# drawing model

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5, device: str = 'cuda'):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.device = device

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output