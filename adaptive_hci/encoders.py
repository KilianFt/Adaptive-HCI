import re

import torch
from torch import nn

from .tcn import TCN
from common import EncoderModelEnum

class EMGLSTM(nn.Module):
    def __init__(self, input_size, enc_config) -> None:
        super().__init__()
        # TODO feature encoder?
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=enc_config.hidden_size,
                            num_layers=enc_config.num_layers, dropout=enc_config.dropout,
                            batch_first=True)

        # default gru initialization is uniform, not recommended
        # https://smerity.com/articles/2016/orthogonal_init.html orthogonal has eigenvalue = 1
        # to prevent grad explosion or vanishing
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

        self.linear = nn.Linear(enc_config.hidden_size, input_size)

    def forward(self, x: torch.Tensor):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

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


class EMGTCN(TCN):
    def __init__(self, input_size, enc_config):
        super().__init__(in_channels=input_size,
                         out_channels=input_size,
                         kernel_size=enc_config.kernel_size,
                         channels=enc_config.channels,
                         layers=enc_config.num_layers,
                         bias=enc_config.bias,)

    def freeze_layers(self, n_frozen_layers: int) -> None:
        raise NotImplementedError
    

ENCODER_MODELS = {
    EncoderModelEnum.TCN: EMGTCN,
    EncoderModelEnum.LSTM: EMGLSTM,
    # TODO 'Permutation-invariant':
}

