import torch


class Chomp1d(torch.nn.Module):
    """
    Removes leading or trailing elements of a time series.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L - s`) where `s`
    is the number of elements to remove.

    Args:
        chomp_size : Number of elements to remove.
        last : If True, removes the last elements in the time dimension,
            If False, removes the fist elements.
    """

    def __init__(self, chomp_size: int, last: bool = True):
        super().__init__()
        self.chomp_size = chomp_size
        self.last = last

    def forward(self, x):
        if self.last:
            x_chomped = x[:, :, :-self.chomp_size]
        else:
            x_chomped = x[:, :, self.chomp_size:]

        return x_chomped


class TCNBlock(torch.nn.Module):
    """
    Temporal Convolutional Network block.

    Composed sequentially of two causal convolutions (with leaky ReLU activation functions),
    and a parallel residual connection.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).

    Args:
        in_channels : Number of input channels.
        out_channels : Number of output channels.
        kernel_size : Kernel size of the applied non-residual convolutions.
        dilation : Dilation parameter of non-residual convolutions.
        bias : If True, adds a learnable bias to the convolutions.
        fwd_time : If True, the network "causal" direction is from past to future (forward),
            if False, the relation is from future to past (backward).
        final : If True, the last activation function is disabled.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        bias: bool = True,
        fwd_time: bool = True,
        final: bool = False,
    ):

        super().__init__()

        in_channels = int(in_channels)
        kernel_size = int(kernel_size)
        out_channels = int(out_channels)
        dilation = int(dilation)

        # Computes left padding so that the applied convolutions are causal
        padding = int((kernel_size - 1) * dilation)

        # First causal convolution
        conv1_pre = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        # conv1 = torch.nn.utils.weight_norm(conv1_pre, name ="weight")
        conv1 = conv1_pre

        # The truncation makes the convolution causal
        chomp1 = Chomp1d(chomp_size=padding, last=fwd_time)

        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2_pre = torch.nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        # conv2 = torch.nn.utils.weight_norm(conv2_pre, name="weight")
        conv2 = conv2_pre
        chomp2 = Chomp1d(padding)
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        self.causal = torch.nn.Sequential(conv1, chomp1, relu1, conv2, chomp2, relu2)

        # Residual connection
        self.upordownsample = (torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        ) if in_channels != out_channels else None)

        # Final activation function
        self.activation = torch.nn.LeakyReLU() if final else None

    def forward(self, x):
        out_causal = self.causal(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        if self.activation is None:
            return out_causal + res
        else:
            return self.activation(out_causal + res)


class TCN(torch.nn.Module):
    """
    Temporal Convolutional Network.

    Composed of a sequence of causal convolution blocks.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    Args:
        in_channels : Number of input channels.
        out_channels : Number of output channels.
        kernel_size : Kernel size of the applied non-residual convolutions.
        channels : Number of channels processed in the network and of output
            channels.
        layers : Depth of the network.
        bias : If True, adds a learnable bias to the convolutions.
        fwd_time : If True the network is the relation relation if from past to future (forward),
            if False, the relation from future to past (backward).
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 channels: int,
                 layers: int,
                 bias: bool = True,
                 fwd_time: bool = True):
        super().__init__()

        layers = int(layers)

        net_layers = []  # List of sequential TCN blocks
        dilation_size = 1  # Initial dilation size

        for i in range(layers):
            in_channels_block = in_channels if i == 0 else channels
            net_layers.append(
                TCNBlock(
                    in_channels=in_channels_block,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation_size,
                    bias=bias,
                    fwd_time=fwd_time,
                    final=False,
                ))
            dilation_size *= 2  # Doubles the dilation size at each step

        # Last layer
        net_layers.append(
            TCNBlock(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation_size,
                bias=bias,
                fwd_time=fwd_time,
                final=True,
            ))

        self.network = torch.nn.Sequential(*net_layers)

    def forward(self, x):
        # x = x.flatten(2).permute(0, 2, 1)
        y = self.network(x)
        # y = y.permute(0, 2, 1)
        return y