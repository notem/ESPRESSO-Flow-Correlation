import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DModel(nn.Module):
    def __init__(self, input_channels, input_size, feature_dim=64, **kwargs):
        super(Conv1DModel, self).__init__()

        self.dummy_param = nn.Parameter(torch.empty(0))

        self.input_size = input_size

        # Define common parameters
        self.filter_nums = [32, 64, 128, 256]
        self.kernel_size = 8
        self.conv_stride_size = 1
        self.pool_stride_size = 4
        self.pool_size = 8
        self.dropout_rate = 0.1

        # Define layers using loops
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        in_channels = input_channels
        for i, out_channels in enumerate(self.filter_nums):
            self.conv_layers.append(nn.Conv1d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = self.kernel_size,
                stride = self.conv_stride_size,
                padding = "same",
            ))
            self.conv_layers.append(nn.Conv1d(
                in_channels = out_channels,
                out_channels = out_channels,
                kernel_size = self.kernel_size,
                stride = self.conv_stride_size,
                padding = "same",
            ))
            self.pool_layers.append(nn.MaxPool1d(
                kernel_size=self.pool_size,
                stride=self.pool_stride_size,
                padding=self.pool_size//2
            ))
            self.dropout_layers.append(nn.Dropout(self.dropout_rate))
            in_channels = out_channels

        flattened_size = self.__stage_size(input_size)[-1] * self.filter_nums[-1]
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(flattened_size, feature_dim)

    def __stage_size(self, input_size):
        """Calculate the sequence size after stages within the model (as a function of input_size)
        """
        fmap_size = [input_size]
        for i in range(len(self.filter_nums)):
            fmap_size.append(fmap_size[-1] // self.pool_stride_size + 1)
        return fmap_size[1:]

    def get_device(self):
        return self.dummy_param.get_device()

    def forward(self, x):
        """
        """
        # add channel dim if necessary
        if len(x.shape) < 3:
            x = x.unsqueeze(1)

        # clip sample length to maximum supported size and pad with zeros if necessary
        size_dif = x.shape[-1] - self.input_size
        if x.shape[-1] > self.input_size:
            x = x[..., :self.input_size]
        elif size_dif < 0:
            x = F.pad(x, (0,abs(size_dif)))

        for i in range(len(self.filter_nums)):
            if i > 0:
                x = F.relu(self.conv_layers[2 * i](x))
                x = F.relu(self.conv_layers[2 * i + 1](x))
            else:
                x = F.elu(self.conv_layers[2 * i](x))
                x = F.elu(self.conv_layers[2 * i + 1](x))
            x = self.pool_layers[i](x)
            x = self.dropout_layers[i](x)

        x = self.flatten(x)
        x = self.dense(x)
        return x
