import abc
import numpy as np
import torch
from torch.functional import F
from stable_baselines3 import PPO
from torch.utils.data import DataLoader
from torchmetrics import ExactMatch, F1Score
from vit_pytorch import ViT
import lightning.pytorch as pl

# from train_general_model import main
from .training import train_model
from .datasets import EMGWindowsAdaptationDataset
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


class SLOnlyController(BaseController):
    def __init__(self, model_path, device='cpu', lr=1e-3, batch_size=64, epochs=1, n_frozen_layers=0):
        self.device = device
        self.batch_size = batch_size
        self.epochs = epochs

        if model_path is not None:
            self.policy = torch.load(model_path).to(self.device)
        else:
            raise NotImplementedError('policy cannot be trained in controller')
            # self.policy = main()

        if n_frozen_layers >= 1:
            for i, param in enumerate(self.policy.to_patch_embedding.parameters()):
                param.requires_grad = False
            for i, param in enumerate(self.policy.dropout.parameters()):
                param.requires_grad = False

        if n_frozen_layers >= 2:
            for layer_idx in range(min((n_frozen_layers - 1), 4)):
                for i, param in enumerate(self.policy.transformer.layers[layer_idx].parameters()):
                    param.requires_grad = False

        self.model_name = model_path.split('/')[-1].split('.')[0]

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

    def deterministic_forward(self, emg_window):
        emg_window_tensor = emg_window.unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.policy(emg_window_tensor)
        return outputs

    def sl_update(self, states, optimal_actions):
        train_dataset = EMGWindowsAdaptationDataset(windows=states, labels=optimal_actions)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,)

        # train model
        # TODO change to lightning
        self.policy, history = train_model(self.policy,
                                    optimizer=self.optimizer,
                                    criterion=self.criterion,
                                    train_dataloader=train_dataloader,
                                    model_name=self.model_name,
                                    device=self.device,
                                    epochs=self.epochs,)

        return history['train_loss'][-1]


class EMGViT(ViT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, x: torch.Tensor):
        x.unsqueeze_(axis=1)
        return self.forward(x)

# TODO combine with controller above
class PLModel(pl.LightningModule):
    def __init__(self, model, n_labels, lr=1e-3):
        super(PLModel, self).__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = lr
        self.criterion = torch.nn.MSELoss()
        self.exact_match = ExactMatch(task="multilabel", num_labels=n_labels, threshold=0.5)
        self.f1_score = F1Score(task="multilabel", num_labels=n_labels, threshold=0.5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)
        loss = self.criterion(outputs, targets)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, targets = batch
        outputs = self.model(data)

        val_acc = self.exact_match(outputs, targets)
        val_f1 = self.f1_score(outputs, targets)
        val_loss = F.mse_loss(outputs, targets)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)
        self.log('val_f1', val_f1, prog_bar=True)

        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
