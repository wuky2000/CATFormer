import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    LayerNorm supporting 'channels_first' (B, C, H, W) or 'channels_last'.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class NormDownsample(nn.Module):
    """ ResNet-style Downsampling block with optional normalization """
    def __init__(self, in_ch, out_ch, scale=0.5, use_norm=False):
        super(NormDownsample, self).__init__()
        stride = 2 if scale == 0.5 else 1
        self.use_norm = use_norm
        norm_layer = nn.BatchNorm2d if self.use_norm else None
        bias = not self.use_norm

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm1 = norm_layer(out_ch) if norm_layer else nn.Identity()
        self.activation = nn.PReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=bias)
        self.norm2 = norm_layer(out_ch) if norm_layer else nn.Identity()

        self.downsample_skip = None
        if stride != 1 or in_ch != out_ch:
            skip = [nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=bias)]
            if norm_layer: skip.append(norm_layer(out_ch))
            self.downsample_skip = nn.Sequential(*skip)

    def forward(self, x):
        identity = self.downsample_skip(x) if self.downsample_skip else x
        out = self.norm2(self.conv2(self.activation(self.norm1(self.conv1(x)))))
        return self.activation(out + identity)

class NormUpsample(nn.Module):
    """ Upsampling block with PReLU and optional Normalization """
    def __init__(self, in_ch, out_ch, scale=2, use_norm=False):
        super(NormUpsample, self).__init__()
        self.use_norm = use_norm
        if self.use_norm: self.norm = LayerNorm(out_ch)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale)
        )
        self.up = nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False)

    def forward(self, x, y):
        x = self.up_scale(x)
        x = torch.cat([x, y], dim=1)
        x = self.prelu(self.up(x))
        return self.norm(x) if self.use_norm else x