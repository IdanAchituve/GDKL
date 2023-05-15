import torch.nn as nn
import torch.nn.functional as F
from GDKL.nn import spectral_norm_fc, spectral_norm_conv, SpectralBatchNorm2d


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.identity_layer = nn.Identity()

    def forward(self, x, *args, **kwargs):
        return self.identity_layer(x)


class FCNet(nn.Module):
    def __init__(
        self,
        input_dim,
        layers,
        dropout_rate=0.0,
        activation="relu",
    ):
        super().__init__()
        self.first = nn.Linear(input_dim, layers[0])
        self.layers = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers[:-2]))]
        )
        self.last = nn.Linear(layers[-2], layers[-1])
        self.dropout = nn.Dropout(dropout_rate)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        elif activation == "tanh":
            self.activation = F.tanh
        else:
            raise ValueError("Activation is unknown")

    def forward(self, x):
        x = self.dropout(self.activation(self.first(x)))
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))

        x = self.last(x)
        return x


# Based on: https://github.com/y0ast/DUE/blob/main/due/wide_resnet.py
class WideBasic(nn.Module):
    def __init__(
        self,
        wrapped_conv,
        wrapped_batchnorm,
        input_size,
        in_c,
        out_c,
        stride,
        dropout_rate,
    ):
        super().__init__()
        self.bn1 = wrapped_batchnorm(in_c)
        self.conv1 = wrapped_conv(input_size, in_c, out_c, 3, stride)
        input_size = (input_size - 1) // stride + 1

        self.bn2 = wrapped_batchnorm(out_c)
        self.conv2 = wrapped_conv(input_size, out_c, out_c, 3, 1)

        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)

        if stride != 1 or in_c != out_c:
            self.shortcut = wrapped_conv(input_size, in_c, out_c, 1, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))

        if self.dropout_rate > 0:
            out = self.dropout(out)

        out = self.conv2(out)
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(
        self,
        input_size=32,
        spectral_conv=False,
        spectral_bn=False,
        depth=28,
        widen_factor=10,
        num_classes=None,
        dropout_rate=0.3,
        coeff=3,
        n_power_iterations=1
    ):
        super().__init__()

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"

        self.dropout_rate = dropout_rate

        def wrapped_bn(num_features):
            if spectral_bn:
                bn = SpectralBatchNorm2d(num_features, coeff)
            else:
                bn = nn.BatchNorm2d(num_features)

            return bn

        self.wrapped_bn = wrapped_bn

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_conv:
                return conv

            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                input_dim = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(
                    conv, coeff, input_dim, n_power_iterations
                )

            return wrapped_conv

        self.wrapped_conv = wrapped_conv

        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]
        strides = [1, 1, 2, 2]

        self.conv1 = wrapped_conv(input_size, 3, nStages[0], 3, strides[0])
        self.layer1, input_size = self._wide_layer(
            nStages[0:2], n, strides[1], input_size
        )
        self.layer2, input_size = self._wide_layer(
            nStages[1:3], n, strides[2], input_size
        )
        self.layer3, input_size = self._wide_layer(
            nStages[2:4], n, strides[3], input_size
        )

        self.bn1 = self.wrapped_bn(nStages[3])
        self.num_classes = num_classes

    def _wide_layer(self, channels, num_blocks, stride, input_size):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        in_c, out_c = channels

        for stride in strides:
            layers.append(
                WideBasic(
                    self.wrapped_conv,
                    self.wrapped_bn,
                    input_size,
                    in_c,
                    out_c,
                    stride,
                    self.dropout_rate,
                )
            )
            in_c = out_c
            input_size = (input_size - 1) // stride + 1

        return nn.Sequential(*layers), input_size

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.flatten(1)

        if self.num_classes is not None:
            out = self.linear(out)
            out = F.log_softmax(out, dim=1)

        return out