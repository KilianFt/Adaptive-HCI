import abc
import torch
from torch import Tensor
from torch.functional import F
from stable_baselines3 import PPO
from torchmetrics import ExactMatch, F1Score, Accuracy
from vit_pytorch import ViT
import lightning.pytorch as pl


class BaseController:
    @abc.abstractmethod
    def deterministic_forward(self, x) -> torch.Tensor:
        pass

    def sl_update(self, states, optimal_actions):
        device = next(self.policy.parameters()).device
        if states.device != device:
            states = states.to(device)
        if optimal_actions.device != device:
            optimal_actions = optimal_actions.to(device)

        self.policy.train()
        self.optimizer.zero_grad()
        predicted_action = self.policy(states)
        target_action = torch.clip(optimal_actions, -1, 1)

        loss = self.criterion(predicted_action, target_action)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class RLSLController(BaseController, PPO):
    def __init__(self, env):
        super(RLSLController, self).__init__(
            "MlpPolicy",
            env,
            n_steps=50,
            verbose=1,
            tensorboard_log="tmp/fetch_reach_tensorboard/",
            learning_rate=5e-3,
        )

    def deterministic_forward(self, x):
        dist = self.policy.get_distribution(x)
        return dist.distribution.mean


class EMGViT(ViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x: torch.Tensor):
        x.unsqueeze_(axis=1)
        return self.forward(x)


class PLModel(pl.LightningModule):
    def __init__(self, model, n_labels, lr=1e-3, n_frozen_layers=None):
        super(PLModel, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.MSELoss()
        self.exact_match = ExactMatch(task="multilabel", num_labels=n_labels, threshold=0.5)
        self.f1_score = F1Score(task="multilabel", num_labels=n_labels, threshold=0.5)
        self.accuracy_metric = Accuracy(task='binary')

        if n_frozen_layers is not None:
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

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def get_per_label_accuracies(self, outputs, targets, threshold = 0.5):
        num_targets = targets.shape[1]
        per_labels_accuracies = torch.zeros(num_targets)

        binary_outputs = (outputs >= threshold).int()
        binary_targets = (targets >= threshold).int()

        for label_idx in range(num_targets):
            label_acc = self.accuracy_metric(binary_outputs[:,label_idx], binary_targets[:,label_idx])
            per_labels_accuracies[label_idx] = label_acc

        return per_labels_accuracies

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)

        val_acc = self.exact_match(outputs, targets)
        val_f1 = self.f1_score(outputs, targets)
        val_loss = F.mse_loss(outputs, targets)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_f1', val_f1, prog_bar=True)

        per_label_accuracies = self.get_per_label_accuracies(outputs, targets)
        self.log('per_label_accuracies', per_label_accuracies, prog_bar=True)

        return val_loss, val_acc, val_f1, per_label_accuracies

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
