import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from .SwinUMamba import VSSMEncoder
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
import re


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True)
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1,
                 norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True)
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            norm_layer(out_channels)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads,
                                    qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(inplace=True),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x, H, W, attn_mask=None):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        Hp, Wp = x.shape[1], x.shape[2]

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=64, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, size=res.shape[-2:], mode='bilinear', align_corners=False)

        weights = F.relu(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x



class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = ConvBN(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.psp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(decode_channels, decode_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, res):
        x = F.interpolate(x, size=res.shape[-2:], mode='bilinear', align_corners=False)
        weights = F.relu(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + 1e-8)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        x = x * self.psp(x)
        return x


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 256, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = WF(encoder_channels[-2], decode_channels)

        self.b2 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = WF(encoder_channels[-3], decode_channels)

        self.b1 = Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            nn.Conv2d(decode_channels, num_classes, kernel_size=1)
        )

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.pre_conv(res4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.b4(x, H, W)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        x = self.p3(x, res3)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.b3(x, H, W)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        x = self.p2(x, res2)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.b2(x, H, W)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        x = self.p1(x, res1)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.b1(x, H, W)
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        return x


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1).contiguous())
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FusionAttention(nn.Module):
    def __init__(self, dim=256, ssmdims=256, num_heads=8, qkv_bias=False, window_size=8):
        super().__init__()
        self.dim = dim
        self.ssmdims = ssmdims
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.kv = nn.Conv2d(ssmdims, dim * 2, kernel_size=1, bias=qkv_bias)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.window_size = window_size

    def forward(self, x, y):
        B, C, H, W = x.shape
        ws = self.window_size
        pad_l = pad_t = 0
        pad_r = (ws - W % ws) % ws
        pad_b = (ws - H % ws) % ws
        x_ = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
        y_ = F.pad(y, (pad_l, pad_r, pad_t, pad_b))
        _, _, Hp, Wp = x_.shape

        q = self.q(x_)
        kv = self.kv(y_)
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (h d) hp wp -> b h (hp wp) d', h=self.num_heads)
        k = rearrange(k, 'b (h d) hp wp -> b h (hp wp) d', h=self.num_heads)
        v = rearrange(v, 'b (h d) hp wp -> b h (hp wp) d', h=self.num_heads)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        out = rearrange(out, 'b h (hp wp) d -> b (h d) hp wp', hp=Hp, wp=Wp)

        out = self.proj(out)
        out = out[:, :, :H, :W].contiguous()
        return out


class FusionBlock(nn.Module):
    def __init__(self, dim=256, ssmdims=256, num_heads=16, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.normx = norm_layer(dim)
        self.normy = norm_layer(ssmdims)
        self.attn = FusionAttention(dim, ssmdims, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
                       act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.normx(x), self.normy(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RegionAttention(nn.Module):
    """Label-guided region attention (LGRA).


    """

    def __init__(self, n_feat: int, num_heads: int = 8, bias: bool = True,
                 mask_mode: str = "hard", mask_beta: float = 4.0, attn_drop: float = 0.0):
        super().__init__()
        assert n_feat % num_heads == 0, f"n_feat({n_feat}) must be divisible by num_heads({num_heads})."
        assert mask_mode in {"hard", "soft"}, f"mask_mode must be 'hard' or 'soft', got {mask_mode}"
        self.num_heads = num_heads
        self.head_dim = n_feat // num_heads
        self.scale = self.head_dim ** -0.5

        self.mask_mode = mask_mode
        self.mask_beta = float(mask_beta)

        self.qkv = nn.Conv2d(n_feat, n_feat * 3, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.project_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

    def _prep_mask(self, mask: torch.Tensor, H: int, W: int, device: torch.device, dtype: torch.dtype):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.dim() != 4:
            raise ValueError(f"Invalid mask shape: {mask.shape}")
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask.float(), size=(H, W), mode="nearest")
        mask = mask.to(device=device, dtype=dtype)
        mask = mask.flatten(2)  # [B,1,L]
        return mask

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (h d) h1 w1 -> b h (h1 w1) d", h=self.num_heads)
        k = rearrange(k, "b (h d) h1 w1 -> b h (h1 w1) d", h=self.num_heads)
        v = rearrange(v, "b (h d) h1 w1 -> b h (h1 w1) d", h=self.num_heads)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale  # [B,heads,L,L]

        roi = self._prep_mask(mask, H, W, device=attn.device, dtype=attn.dtype)  # [B,1,L]
        roi_k = roi.unsqueeze(1)  # [B,1,1,L]

        if self.mask_mode == "hard":
            bias = (1.0 - roi_k) * (-1e4)
        else:
            bias = (roi_k - 0.5) * (2.0 * self.mask_beta)

        attn = attn + bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = rearrange(out, "b h (h1 w1) d -> b (h d) h1 w1", h1=H, w1=W)
        out = self.project_out(out)
        return out


class DegradeBlock(nn.Module):
    """SSRAE ：downsample -> blur -> (noise) -> upsample back.
    """

    def __init__(self,
                 in_channels: int,
                 scale_factor: float = 0.5,
                 blur_kernel: int = 3,
                 blur_sigma: float = 1.0,
                 noise_std: float = 0.01,
                 noise_train_only: bool = True,
                 blur_learnable: bool = False):
        super().__init__()
        assert 0 < scale_factor <= 1.0, f"scale_factor must be in (0,1], got {scale_factor}"
        assert blur_kernel % 2 == 1, "blur_kernel must be odd"
        self.in_channels = int(in_channels)
        self.scale_factor = float(scale_factor)
        self.blur_kernel = int(blur_kernel)
        self.blur_sigma = float(blur_sigma)
        self.noise_std = float(noise_std)
        self.noise_train_only = bool(noise_train_only)
        self.blur_learnable = bool(blur_learnable)

        self.blur = nn.Conv2d(self.in_channels, self.in_channels,
                              kernel_size=self.blur_kernel,
                              padding=self.blur_kernel // 2,
                              groups=self.in_channels,
                              bias=False)

        with torch.no_grad():
            k = self._make_gaussian_kernel(self.blur_kernel, self.blur_sigma)
            w = k.view(1, 1, self.blur_kernel, self.blur_kernel).repeat(self.in_channels, 1, 1, 1)
            self.blur.weight.copy_(w)

        if not self.blur_learnable:
            for p in self.blur.parameters():
                p.requires_grad_(False)

    @staticmethod
    def _make_gaussian_kernel(ksize: int, sigma: float):
        ax = torch.arange(ksize) - ksize // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if self.scale_factor < 1.0:
            x_lr = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        else:
            x_lr = x

        x_lr = self.blur(x_lr)

        if (not self.noise_train_only) or self.training:
            if self.noise_std > 0:
                x_lr = x_lr + torch.randn_like(x_lr) * self.noise_std

        if x_lr.shape[-2:] != (H, W):
            x_lr = F.interpolate(x_lr, size=(H, W), mode="bilinear", align_corners=False)
        return x_lr


class SRDecoder(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels=(384, 192, 96, 48),
                 dropout: float = 0.0):
        super().__init__()
        chs = [int(in_channels)] + [int(c) for c in hidden_channels]
        self.blocks = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.blocks.append(nn.Sequential(
                ConvBNReLU(chs[i], chs[i + 1], kernel_size=3, stride=1),
                nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            ))
        self.proj = nn.Conv2d(int(chs[-1]), int(out_channels), kernel_size=1, bias=True)

    def forward(self, bottleneck: torch.Tensor, target_size):
        x = bottleneck
        for blk in self.blocks:
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = blk(x)
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.proj(x)
        return x


class FaRMamba(nn.Module):
    """FaRMamba: Frequency-aware Region-guided Mamba segmentation with SSRAE (paper-aligned).

      - use_msfm / msfm_type (WT/FFT/DCT/NONE)
      - use_ssrae
      - share_mamba_backbone
      - use_region_attention (LGRA)
      - use_fusion (CNN<->Mamba fusion)
    """

    def __init__(self,
                 num_classes: int = 4,
                 in_chans: int = 1,
                 # seg decoder
                 decode_channels: int = 64,
                 dropout: float = 0.1,
                 window_size: int = 8,
                 # cnn backbone
                 backbone_name: str = "swsl_resnet18",
                 pretrained: bool = True,
                 # msfm
                 use_msfm: bool = True,
                 msfm_type: str = "WT",  # WT / FFT / DCT / NONE
                 msfm_in_planes: int = 48,
                 # mamba
                 vssm_patch_size: int = 2,
                 vssm_in_chans: int = 48,
                 vssm_depths=(2, 2, 9, 2),
                 vssm_dims=(96, 192, 384, 768),
                 share_mamba_backbone: bool = True,
                 # fusion
                 use_fusion: bool = True,
                 # ssrae
                 use_ssrae: bool = True,
                 degrade_scale_factor: float = 0.5,
                 degrade_blur_kernel: int = 3,
                 degrade_blur_sigma: float = 1.0,
                 degrade_noise_std: float = 0.01,
                 degrade_noise_train_only: bool = True,
                 degrade_blur_learnable: bool = False,
                 # lgra
                 use_region_attention: bool = True,
                 region_attn_heads: int = 8,
                 region_attn_mode: str = "hard",  # hard / soft
                 region_attn_beta: float = 4.0,
                 region_focus_classes=None,
                 # sr decoder
                 sr_decoder_dropout: float = 0.0):
        super().__init__()

        self.num_classes = int(num_classes)
        self.in_chans = int(in_chans)

        # CNN backbone
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            output_stride=32,
            out_indices=(1, 2, 3, 4),
            pretrained=pretrained,
            in_chans=in_chans,
        )
        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.act1 = self.backbone.act1
        self.maxpool = self.backbone.maxpool
        self.layers = nn.ModuleList([self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4])
        encoder_channels = self.backbone.feature_info.channels()

        # shallow stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, msfm_in_planes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(msfm_in_planes, eps=1e-5, affine=True),
        )

        # MSFM
        self.use_msfm = bool(use_msfm)
        self.msfm_type = str(msfm_type).upper()
        self.msfm = self._build_msfm(self.msfm_type, msfm_in_planes) if self.use_msfm else nn.Identity()

        # shared Mamba
        self.share_mamba_backbone = bool(share_mamba_backbone)
        self.vssm_encoder = VSSMEncoder(
            patch_size=vssm_patch_size,
            in_chans=vssm_in_chans,
            depths=list(vssm_depths),
            dims=list(vssm_dims),
        )
        if not self.share_mamba_backbone:
            self.sr_encoder = VSSMEncoder(
                patch_size=vssm_patch_size,
                in_chans=vssm_in_chans,
                depths=list(vssm_depths),
                dims=list(vssm_dims),
            )
        else:
            self.sr_encoder = None
        # VSSMEncoder: patch_size=2 + 3次merge(4 stages) => 总体要求输入 H/W 可被 16 整除
        self.vssm_input_multiple = int(vssm_patch_size * (2 ** (len(vssm_depths) - 1)))
        # usually 16；if change  stages/patch_size，will auto follow

        # fusion
        self.use_fusion = bool(use_fusion)
        ssm_dims = list(vssm_dims)
        if self.use_fusion:
            self.Fuse = nn.ModuleList([FusionBlock(encoder_channels[i], ssm_dims[i], window_size=window_size) for i in range(4)])
        else:
            self.Fuse = None

        # seg decoder
        self.decoder = Decoder(
            encoder_channels=encoder_channels,
            decode_channels=decode_channels,
            dropout=dropout,
            window_size=window_size,
            num_classes=num_classes,
        )

        # SSRAE
        self.use_ssrae = bool(use_ssrae)
        self.region_focus_classes = region_focus_classes

        if self.use_ssrae:
            self.degrade_block = DegradeBlock(
                in_channels=msfm_in_planes,
                scale_factor=degrade_scale_factor,
                blur_kernel=degrade_blur_kernel,
                blur_sigma=degrade_blur_sigma,
                noise_std=degrade_noise_std,
                noise_train_only=degrade_noise_train_only,
                blur_learnable=degrade_blur_learnable,
            )

            self.use_region_attention = bool(use_region_attention)
            if self.use_region_attention:
                self.region_attention = RegionAttention(
                    n_feat=int(ssm_dims[-1]),
                    num_heads=int(region_attn_heads),
                    mask_mode=str(region_attn_mode).lower(),
                    mask_beta=float(region_attn_beta),
                )
            else:
                self.region_attention = None

            self.sr_decoder = SRDecoder(
                in_channels=int(ssm_dims[-1]),
                out_channels=int(msfm_in_planes),
                hidden_channels=(int(ssm_dims[-2]), int(ssm_dims[-3]), int(ssm_dims[-4]), int(msfm_in_planes)),
                dropout=sr_decoder_dropout,
            )
        else:
            self.degrade_block = None
            self.use_region_attention = False
            self.region_attention = None
            self.sr_decoder = None

    @staticmethod
    def _build_msfm(msfm_type: str, in_planes: int):
        msfm_type = str(msfm_type).upper()
        if msfm_type in {"NONE", "IDENTITY"}:
            return nn.Identity()

        if msfm_type == "WT":
            from Module.WT_stable import CombinedModule as _CM
            return _CM(in_planes=in_planes)

        if msfm_type == "FFT":
            from Module.FFTtransformer import CombinedModule as _CM
            return _CM(in_planes=in_planes)

        if msfm_type == "DCT":
            from Module.DCTtransformer import CombinedModule as _CM
            return _CM(in_planes=in_planes)

        raise ValueError(f"Unknown msfm_type: {msfm_type}. Use WT / FFT / DCT / NONE.")

    def _pad_to_multiple(self, x: torch.Tensor, multiple: int):
        """Pad bottom/right so that H,W are divisible by `multiple`."""
        H, W = x.shape[-2:]
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)
        return x, pad_h, pad_w

    def _crop_hw(self, x: torch.Tensor, H: int, W: int):
        """Crop to target H,W (assumes padding only on bottom/right)."""
        return x[..., :H, :W].contiguous()

    def _make_roi_mask(self, labels: torch.Tensor, target_hw, device):
        if labels is None:
            return None

        if labels.dim() == 4 and labels.shape[1] > 1:
            labels = labels.argmax(dim=1)
        elif labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels[:, 0]
        elif labels.dim() != 3:
            raise ValueError(f"Invalid labels shape: {labels.shape}")

        if self.region_focus_classes is None:
            roi = (labels > 0).float()
        else:
            roi = torch.zeros_like(labels, dtype=torch.float32)
            for cid in self.region_focus_classes:
                roi = roi + (labels == int(cid)).float()
            roi = (roi > 0).float()

        roi = roi.unsqueeze(1)
        roi = F.interpolate(roi, size=target_hw, mode="nearest")
        roi = roi.to(device=device)
        return roi

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None, return_aux: bool = False):
        h, w = x.shape[-2:]

        # shallow + MSFM
        f0 = self.stem(x)      # [B,48,H/2,W/2]
        f0 = self.msfm(f0)     # MSFM

        # seg mamba feats
        f0_for_mamba, ph0, pw0 = self._pad_to_multiple(f0, self.vssm_input_multiple)
        seg_feats = self.vssm_encoder(f0_for_mamba)

        # SSRAE (optional)
        aux = {}
        if self.use_ssrae:
            degraded = self.degrade_block(f0)
            degraded_for_mamba, phd, pwd = self._pad_to_multiple(degraded, self.vssm_input_multiple)

            enc = self.vssm_encoder if self.share_mamba_backbone else self.sr_encoder
            sr_feats = enc(degraded_for_mamba)

            region_mask = None
            sr_bottleneck = sr_feats[-1]
            if self.use_region_attention and (labels is not None):
                region_mask = self._make_roi_mask(labels, target_hw=sr_bottleneck.shape[-2:], device=sr_bottleneck.device)
                sr_bottleneck = self.region_attention(sr_bottleneck, region_mask)

            sr_pred = self.sr_decoder(sr_bottleneck, target_size=f0.shape[-2:])

            aux.update({
                "sr_pred": sr_pred,
                "sr_target": f0.detach(),
                "degraded": degraded,
                "sr_bottleneck": sr_bottleneck,
                "region_mask": region_mask,
            })

        # CNN backbone + fusion
        ress = []
        xb = self.conv1(x)
        xb = self.bn1(xb)
        xb = self.act1(xb)
        xb = self.maxpool(xb)

        for i, layer in enumerate(self.layers):
            xb = layer(xb)
            if self.use_fusion:
                s = seg_feats[i + 1]
                if s.shape[-2:] != xb.shape[-2:]:
                    s = self._crop_hw(s, xb.shape[-2], xb.shape[-1])
                xb = self.Fuse[i](xb, s)

            ress.append(xb)

        seg_output = self.decoder(ress[0], ress[1], ress[2], ress[3], h, w)

        if (not return_aux) and (not self.use_ssrae):
            return seg_output
        return seg_output, aux


# Backward-compatible alias (if your old training code imports `sota`)
sota = FaRMamba


if __name__ == "__main__":
    import os
    import traceback
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    torch.set_printoptions(precision=4, sci_mode=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # ---------------------------
    # Basic test config
    # ---------------------------
    B = 2
    C = 1
    H, W = 256, 256
    num_classes = 4

    x = torch.randn(B, C, H, W, device=device)
    labels = torch.randint(0, num_classes, (B, H, W), device=device)

    # ---------------------------
    # Test 1: Full path (SSRAE ON, shared mamba ON, LGRA ON, Fusion ON)
    # ---------------------------
    try:
        msfm_try = "WT"  #  "FFT" / "DCT"
        print(f"\n[TEST-1] Full path, try msfm={msfm_try} (fallback to NONE if import fails)")

        try:
            model = FaRMamba(
                num_classes=num_classes,
                in_chans=C,
                pretrained=False,
                use_msfm=True,
                msfm_type=msfm_try,
                use_ssrae=True,
                share_mamba_backbone=True,
                use_region_attention=True,
                region_attn_mode="hard",
                use_fusion=True,
            ).to(device)
            print(f"  [MSFM] enabled: {msfm_try}")
        except Exception as e_msfm:
            print(f"  [MSFM] failed to enable ({msfm_try}): {repr(e_msfm)}")
            print("  [MSFM] fallback to NONE for self-contained sanity test.")
            model = FaRMamba(
                num_classes=num_classes,
                in_chans=C,
                pretrained=False,
                use_msfm=False,
                msfm_type="NONE",
                use_ssrae=True,
                share_mamba_backbone=True,
                use_region_attention=True,
                region_attn_mode="hard",
                use_fusion=True,
            ).to(device)

        model.train()
        y, aux = model(x, labels=labels, return_aux=True)

        print(f"  seg_out: {tuple(y.shape)} (expect: {(B, num_classes, H, W)})")
        for k in ["sr_pred", "sr_target", "degraded", "sr_bottleneck", "region_mask"]:
            v = aux.get(k, None)
            if v is None:
                print(f"  aux[{k}]: None")
            else:
                print(f"  aux[{k}]: {tuple(v.shape)}")

        # Dummy losses + backward (sanity gradient flow)
        seg_loss = F.cross_entropy(y, labels)
        sr_loss = F.l1_loss(aux["sr_pred"], aux["sr_target"])
        loss = seg_loss + 0.1 * sr_loss
        loss.backward()

        # check some grads exist
        grad_ok = False
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad_ok = True
                print(f"  [GRAD] ok: {name} grad_mean={p.grad.abs().mean().item():.6f}")
                break
        print(f"  total_loss={loss.item():.6f} (seg={seg_loss.item():.6f}, sr={sr_loss.item():.6f}), grad_ok={grad_ok}")

        # Eval path (no labels -> no region guidance; degrade noise should be disabled if noise_train_only=True)
        model.eval()
        with torch.no_grad():
            y2, aux2 = model(x, labels=None, return_aux=True)
        print(f"  [EVAL] seg_out: {tuple(y2.shape)}")
        if "sr_pred" in aux2:
            print(f"  [EVAL] sr_pred: {tuple(aux2['sr_pred'].shape)}")

    except Exception as e:
        print("[TEST-1][ERROR]", repr(e))
        traceback.print_exc()

    # ---------------------------
    # Test 2: Seg-only path (SSRAE OFF)
    # ---------------------------
    try:
        print("\n[TEST-2] SSRAE=OFF (seg-only), fusion=ON, msfm=NONE")
        model2 = FaRMamba(
            num_classes=num_classes,
            in_chans=C,
            pretrained=False,
            use_msfm=False,
            msfm_type="NONE",
            use_ssrae=False,
            share_mamba_backbone=True,
            use_region_attention=False,
            use_fusion=True,
        ).to(device)

        model2.eval()
        with torch.no_grad():
            y3 = model2(x)  # should return only seg_output
        print(f"  seg_out: {tuple(y3.shape)} (expect: {(B, num_classes, H, W)})")

    except Exception as e:
        print("[TEST-2][ERROR]", repr(e))
        traceback.print_exc()

    # ---------------------------
    # Test 3: Non-square / non-multiple-of-32 shape robustness
    # ---------------------------
    try:
        print("\n[TEST-3] shape robustness (H,W not aligned), SSRAE=ON")
        H2, W2 = 250, 330
        x4 = torch.randn(B, C, H2, W2, device=device)
        labels4 = torch.randint(0, num_classes, (B, H2, W2), device=device)

        model3 = FaRMamba(
            num_classes=num_classes,
            in_chans=C,
            pretrained=False,
            use_msfm=False,
            msfm_type="NONE",
            use_ssrae=True,
            share_mamba_backbone=True,
            use_region_attention=True,
            use_fusion=True,
        ).to(device)

        model3.eval()
        with torch.no_grad():
            y4, aux4 = model3(x4, labels=labels4, return_aux=True)
        print(f"  seg_out: {tuple(y4.shape)} (expect: {(B, num_classes, H2, W2)})")
        print(f"  sr_pred: {tuple(aux4['sr_pred'].shape)} ; sr_target: {tuple(aux4['sr_target'].shape)}")

    except Exception as e:
        print("[TEST-3][ERROR]", repr(e))
        traceback.print_exc()

    print("\n[DONE] All tests finished.")
