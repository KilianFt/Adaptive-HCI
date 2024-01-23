import lightning.pytorch as pl
import torch
from torch.functional import F
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
    elif criterion_key == 'ce':
        return torch.nn.CrossEntropyLoss()
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
        val_loss = self.criterion(outputs, targets)
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


class SingleLabelPlModel(PLModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accuracy_metric = Accuracy(task="multiclass", num_classes=kwargs["n_labels"])


    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)
        val_loss = self.criterion(outputs, targets)

        out_preds = F.softmax(outputs, dim=-1).argmax(-1).type(torch.long)
        val_acc = self.accuracy_metric(out_preds, targets)

        self.log(f'{self.metric_prefix}validation/loss', val_loss, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/acc', val_acc, prog_bar=True)
        self.log(f'{self.metric_prefix}validation/step', self.step_count, prog_bar=True)

        return val_loss, val_acc
