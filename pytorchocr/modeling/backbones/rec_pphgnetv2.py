from __future__ import absolute_import, division, print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor  # 明确导入 Tensor
from torch.nn.parameter import Parameter
from collections import OrderedDict
from typing import Tuple, List, Dict, Union, Callable, Any, Optional, Type

# 从 ppocr.modeling.backbones.rec_donut_swin import DonutSwinModelOutput
# 我们在文件顶部定义自己的 DonutSwinModelOutput
# from paddle.nn.initializer import Uniform # PyTorch 有自己的初始化
# from paddle.regularizer import L2Decay # L2Decay 在 PyTorch 优化器中处理
# from paddle import ParamAttr # PyTorch 中参数属性通过 Parameter 或直接定义


# --- DonutSwinModelOutput ---
class DonutSwinModelOutput(OrderedDict):
    """基于 Transformers SwinModelOutput 的输出类，用于兼容性。"""

    last_hidden_state: Optional[torch.Tensor] = None
    pooler_output: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    reshaped_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, k):
        return dict(self.items())[k] if isinstance(k, str) else self.to_tuple()[k]

    def __setattr__(self, name, value):
        if name in self.keys() and value is not None:
            super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any, ...]:
        return tuple(self[k] for k in self.keys() if self[k] is not None)


# --- DiverseBranchBlock 及其辅助模块的 PyTorch 实现 ---
class IdentityBasedConv1x1(nn.Conv2d):  # 与 Paddle 类名一致
    def __init__(self, channels: int, groups: int = 1):
        super().__init__(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias=False,
        )
        assert channels % groups == 0
        input_dim = channels // groups
        id_value_np = np.zeros((channels, input_dim, 1, 1), dtype=np.float32)
        for i in range(channels):
            id_value_np[i, i % input_dim, 0, 0] = 1
        self.register_buffer("id_tensor", torch.from_numpy(id_value_np))
        with torch.no_grad():
            self.weight.zero_()

    def forward(
        self, input_tensor: Tensor
    ) -> Tensor:  # 参数名改为 input_tensor 避免与关键字冲突
        kernel = self.weight + self.id_tensor
        return F.conv2d(
            input_tensor,
            kernel,
            None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def get_actual_kernel(self) -> Tensor:
        return self.weight + self.id_tensor


class BNAndPad(nn.Module):  # 与 Paddle 类名一致
    def __init__(
        self,
        pad_pixels: int,
        num_features: int,
        epsilon: float = 1e-5,  # eps for PyTorch
        momentum: float = 0.1,
        last_conv_bias: Optional[Tensor] = None,
        bn: Type[nn.BatchNorm2d] = nn.BatchNorm2d,
    ):  # bn 参数名与 Paddle 一致
        super().__init__()
        self.bn_layer = bn(
            num_features, eps=epsilon, momentum=momentum
        )  # PyTorch BatchNorm2d 使用 eps
        self.pad_pixels = pad_pixels
        self.last_conv_bias = last_conv_bias  # Paddle 属性名

    def forward(self, input_tensor: Tensor) -> Tensor:
        output = self.bn_layer(input_tensor)
        if self.pad_pixels > 0:
            # Paddle: bias_val = -self.bn._mean ; if self.last_conv_bias: bias_val += self.last_conv_bias
            # pad_values = self.bn.bias + self.bn.weight * (bias_val / sqrt(var+eps))
            bias_val = -self.bn_layer.running_mean
            if self.last_conv_bias is not None:
                bias_val = bias_val + self.last_conv_bias.to(bias_val.device)

            pad_values = self.bn_layer.bias + self.bn_layer.weight * (
                bias_val / torch.sqrt(self.bn_layer.running_var + self.bn_layer.eps)
            )

            N, C, H, W = output.shape
            values_to_pad = pad_values.view(1, -1, 1, 1)
            padding_h_tensor = values_to_pad.expand(N, -1, self.pad_pixels, W)
            output = torch.cat([padding_h_tensor, output, padding_h_tensor], dim=2)
            H_padded = H + 2 * self.pad_pixels
            padding_w_tensor = values_to_pad.expand(N, -1, H_padded, self.pad_pixels)
            output = torch.cat([padding_w_tensor, output, padding_w_tensor], dim=3)
        return output

    @property
    def weight(self):
        return self.bn_layer.weight

    @property
    def bias(self):
        return self.bn_layer.bias

    @property
    def _mean(self):
        return self.bn_layer.running_mean

    @property
    def _variance(self):
        return self.bn_layer.running_var

    @property
    def _epsilon(self):
        return self.bn_layer.eps


def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    padding_mode: str = "zeros",
) -> nn.Sequential:
    # Paddle: bias_attr=False. PyTorch: bias=False.
    # Paddle padding_mode: PyTorch Conv2d 不直接接受此参数，"zeros" 是默认行为。
    conv_layer = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias=False,
    )
    bn_layer = nn.BatchNorm2d(out_channels)
    # 使用 OrderedDict 来确保子模块名称 'conv' 和 'bn' 与 Paddle 一致
    return nn.Sequential(OrderedDict([("conv", conv_layer), ("bn", bn_layer)]))


def transI_fusebn(
    kernel: Tensor, bn_module: Union[nn.BatchNorm2d, BNAndPad]
) -> Tuple[Tensor, Tensor]:
    gamma = bn_module.weight
    beta = bn_module.bias
    mean = bn_module._mean
    var = bn_module._variance
    eps = bn_module._epsilon  # 使用属性访问
    std_dev = torch.sqrt(var + eps)
    fused_kernel = kernel * (gamma / std_dev).view(-1, 1, 1, 1)
    fused_bias = beta - (mean * gamma / std_dev)
    return fused_kernel, fused_bias


def transII_addbranch(
    kernels: Tuple[Any, ...], biases: Tuple[Any, ...]
) -> Tuple[Tensor, Tensor]:
    valid_kernels = [k for k in kernels if isinstance(k, Tensor)]
    valid_biases = [b for b in biases if isinstance(b, Tensor)]
    # 确保在空列表时返回一个与第一个有效张量相同设备和类型的零张量，或默认CPU float
    dev = valid_kernels[0].device if valid_kernels else torch.device("cpu")
    dt = valid_kernels[0].dtype if valid_kernels else torch.float32
    sum_k = (
        sum(valid_kernels) if valid_kernels else torch.tensor(0.0, device=dev, dtype=dt)
    )
    sum_b = (
        sum(valid_biases) if valid_biases else torch.tensor(0.0, device=dev, dtype=dt)
    )
    return sum_k, sum_b


def transIII_1x1_kxk(
    k1: Tensor, b1: Tensor, k2: Tensor, b2: Tensor, groups: int
) -> Tuple[Tensor, Tensor]:
    # 之前的PyTorch DBB融合逻辑，需要严格测试
    if groups == 1:
        k_fused = F.conv2d(
            k2, k1.permute(1, 0, 2, 3), padding="same" if k1.size(2) == 1 else 0
        )
        b_hat = (k2 * b1.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        C_mid_total_k1 = k1.shape[0]
        C_in_per_group_k1 = k1.shape[1]
        C_out_total_k2 = k2.shape[0]
        C_mid_per_group_k2 = k2.shape[1]
        k1_g = k1.view(groups, C_mid_total_k1 // groups, C_in_per_group_k1, 1, 1)
        b1_g = b1.view(groups, C_mid_total_k1 // groups)
        k2_g = k2.view(
            groups, C_out_total_k2 // groups, C_mid_per_group_k2, k2.size(2), k2.size(3)
        )
        for g_idx in range(groups):
            k1_s = k1_g[g_idx]
            b1_s = b1_g[g_idx]
            k2_s = k2_g[g_idx]
            k_f_s = F.conv2d(
                k2_s,
                k1_s.permute(1, 0, 2, 3),
                padding="same" if k1_s.size(2) == 1 else 0,
            )
            k_slices.append(k_f_s)
            b_h_s = (k2_s * b1_s.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))
            b_slices.append(b_h_s)
        k_fused = torch.cat(k_slices, dim=0)
        b_hat = torch.cat(b_slices, dim=0)
    return k_fused, b_hat + b2


def transIV_depthconcat(
    kernels: List[Tensor], biases: List[Tensor]
) -> Tuple[Tensor, Tensor]:
    return torch.cat(kernels, dim=0), torch.cat(biases, dim=0)


def transV_avg(channels: int, kernel_size: int, groups: int) -> Tensor:
    input_dim = channels // groups  # Paddle: input_dim = channels // groups
    k = torch.zeros(
        (channels, input_dim, kernel_size, kernel_size), dtype=torch.float32
    )
    val = 1.0 / (kernel_size * kernel_size)
    for i in range(channels):
        k[i, i % input_dim, :, :] = val
    return k


def transVI_multiscale(kernel: Tensor, target_kernel_size: int) -> Tensor:
    pad_h = (target_kernel_size - kernel.shape[2]) // 2
    pad_w = (target_kernel_size - kernel.shape[3]) // 2
    return F.pad(kernel, [pad_w, pad_w, pad_h, pad_h])


class DiverseBranchBlock(nn.Module):  # 与 Paddle 类名一致
    def __init__(
        self,
        num_channels: int,
        num_filters: int,
        filter_size: int,
        stride: int = 1,
        groups: int = 1,
        act: Optional[str] = None,  # Paddle act 是 None 或 ReLU 实例
        is_repped: bool = False,
        single_init: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        padding = (filter_size - 1) // 2
        in_channels, out_channels, kernel_size = num_channels, num_filters, filter_size

        self.is_repped = is_repped
        self.nonlinear = (
            nn.ReLU() if isinstance(act, nn.ReLU) or act == "relu" else nn.Identity()
        )  # 检查 act 类型

        # 存储参数，避免与 nn.Module 内置属性冲突，并与 Paddle 实例属性对应
        self.kernel_size_prop = kernel_size  # Paddle: self.kernel_size
        self.out_channels_prop = out_channels  # Paddle: self.out_channels
        self.groups_prop = groups  # Paddle: self.groups
        self.in_channels_prop = in_channels  # 额外存储，方便重参数化
        self.stride_prop = stride
        self.padding_prop = padding

        assert padding == kernel_size // 2

        _internal_channels = kwargs.get("internal_channels_1x1_3x3", None)
        if _internal_channels is None:
            _internal_channels = (
                in_channels if groups < out_channels else 2 * in_channels
            )

        if is_repped:
            self.dbb_reparam = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=True,
            )  # Paddle: bias_attr=True
        else:
            self.dbb_origin = conv_bn(
                in_channels, out_channels, kernel_size, stride, padding, groups=groups
            )
            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_module(
                    "conv",
                    nn.Conv2d(
                        in_channels, out_channels, 1, 1, 0, groups=groups, bias=False
                    ),
                )
                self.dbb_avg.add_module(
                    "bn", BNAndPad(pad_pixels=padding, num_features=out_channels)
                )
                self.dbb_avg.add_module(
                    "avg", nn.AvgPool2d(kernel_size, stride, padding=0)
                )
                self.dbb_1x1 = conv_bn(
                    in_channels, out_channels, 1, stride, 0, groups=groups
                )
            else:
                self.dbb_avg.add_module(
                    "avg", nn.AvgPool2d(kernel_size, stride, padding=padding)
                )
            self.dbb_avg.add_module("avgbn", nn.BatchNorm2d(out_channels))

            self.dbb_1x1_kxk = nn.Sequential()
            if _internal_channels == in_channels:
                self.dbb_1x1_kxk.add_module(
                    "idconv1", IdentityBasedConv1x1(in_channels, groups)
                )
            else:
                self.dbb_1x1_kxk.add_module(
                    "conv1",
                    nn.Conv2d(
                        in_channels,
                        _internal_channels,
                        1,
                        1,
                        0,
                        groups=groups,
                        bias=False,
                    ),
                )
            self.dbb_1x1_kxk.add_module(
                "bn1", BNAndPad(pad_pixels=padding, num_features=_internal_channels)
            )
            self.dbb_1x1_kxk.add_module(
                "conv2",
                nn.Conv2d(
                    _internal_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    0,
                    groups=groups,
                    bias=False,
                ),
            )
            self.dbb_1x1_kxk.add_module("bn2", nn.BatchNorm2d(out_channels))

        if single_init and not is_repped:
            self.single_init()

    def forward(self, inputs: Tensor) -> Tensor:
        if self.is_repped:
            return self.nonlinear(self.dbb_reparam(inputs))
        out = self.dbb_origin(inputs)
        if hasattr(self, "dbb_1x1"):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value: float):
        if hasattr(self, "dbb_origin"):
            nn.init.constant_(self.dbb_origin._modules["bn"].weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            nn.init.constant_(self.dbb_1x1._modules["bn"].weight, gamma_value)
        if hasattr(self, "dbb_avg") and hasattr(self.dbb_avg, "avgbn"):
            nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk") and hasattr(self.dbb_1x1_kxk, "bn2"):
            nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            nn.init.constant_(self.dbb_origin._modules["bn"].weight, 1.0)

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        k_origin, b_origin = transI_fusebn(
            self.dbb_origin._modules["conv"].weight, self.dbb_origin._modules["bn"]
        )
        dev = k_origin.device
        k_1x1, b_1x1 = torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)
        if hasattr(self, "dbb_1x1"):
            k_f, b_f = transI_fusebn(
                self.dbb_1x1._modules["conv"].weight, self.dbb_1x1._modules["bn"]
            )
            k_1x1 = transVI_multiscale(k_f, self.kernel_size_prop)
            b_1x1 = b_f

        k1_1x1kxk_w = (
            self.dbb_1x1_kxk.idconv1.get_actual_kernel()
            if hasattr(self.dbb_1x1_kxk, "idconv1")
            else self.dbb_1x1_kxk.conv1.weight
        )
        k1_f, b1_f = transI_fusebn(k1_1x1kxk_w, self.dbb_1x1_kxk.bn1)  # bn1 is BNAndPad
        k2_f, b2_f = transI_fusebn(self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1kxk, b_1x1kxk = transIII_1x1_kxk(k1_f, b1_f, k2_f, b2_f, self.groups_prop)

        k_avg_op = transV_avg(
            self.out_channels_prop, self.kernel_size_prop, self.groups_prop
        ).to(dev)
        k_avg_bn, b_avg_bn = transI_fusebn(k_avg_op, self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, "conv"):
            k_avg_c1, b_avg_c1 = transI_fusebn(
                self.dbb_avg.conv.weight, self.dbb_avg.bn
            )  # bn is BNAndPad
            k_avg_m, b_avg_m = transIII_1x1_kxk(
                k_avg_c1, b_avg_c1, k_avg_bn, b_avg_bn, self.groups_prop
            )
        else:
            k_avg_m, b_avg_m = k_avg_bn, b_avg_bn
        return transII_addbranch(
            (k_origin, k_1x1, k_1x1kxk, k_avg_m), (b_origin, b_1x1, b_1x1kxk, b_avg_m)
        )

    def re_parameterize(self):
        if self.is_repped:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2d(
            self.in_channels_prop,
            self.out_channels_prop,
            self.kernel_size_prop,
            self.stride_prop,
            self.padding_prop,
            groups=self.groups_prop,
            bias=True,
        )
        self.dbb_reparam.weight.data = kernel
        self.dbb_reparam.bias.data = bias
        for name_to_del in ["dbb_origin", "dbb_avg", "dbb_1x1", "dbb_1x1_kxk"]:
            if hasattr(self, name_to_del):
                self.__delattr__(name_to_del)
        self.is_repped = True


# --- TheseusLayer 和 PPHGNetV2 子模块的 PyTorch 实现 ---
class Identity(nn.Module):  # 与 Paddle 类名一致
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: Tensor) -> Tensor:
        return input_tensor


class TheseusLayer(nn.Module):  # 与 Paddle 类名一致 (简化版)
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Paddle TheseusLayer 的属性，如果 PPHGNetV2 子模块不直接用，可以简化
        self.res_dict = {}
        # self.res_name = "some_name" # PyTorch nn.Module 没有 full_name()
        self.pruner = None
        self.quanter = None
        # self.init_net(*args, **kwargs) # 原始 init_net 逻辑复杂，这里跳过
        # 除非 PPHGNetV2 的参数里有 stages_pattern 等
        # 并且这些参数被 kwargs 传递到这里


class LearnableAffineBlock(TheseusLayer):  # 与 Paddle 类名一致
    def __init__(
        self,
        scale_value: float = 1.0,
        bias_value: float = 0.0,
        lr_mult: float = 1.0,
        lab_lr: float = 0.01,
    ):
        super().__init__()
        self.lr_mult = lr_mult
        self.lab_lr = lab_lr  # 供优化器使用
        self.scale = Parameter(torch.tensor([scale_value], dtype=torch.float32))
        self.bias = Parameter(torch.tensor([bias_value], dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class ConvBNAct(TheseusLayer):  # 与 Paddle 类名一致
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, str] = 1,
        groups: int = 1,
        use_act: bool = True,
        use_lab: bool = False,
        lr_mult: float = 1.0,
    ):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab

        actual_padding: Union[int, Tuple[int, int]]
        k_is_int = isinstance(kernel_size, int)
        pad_is_str = isinstance(padding, str)

        if pad_is_str and padding.lower() == "same":
            if k_is_int:
                actual_padding = (kernel_size - 1) // 2
            elif isinstance(kernel_size, tuple):
                actual_padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
            else:
                actual_padding = 0
            # print(f"警告: ConvBNAct 收到 padding='SAME', 解释为 {actual_padding}。")
        elif isinstance(padding, int):
            actual_padding = padding
        elif isinstance(padding, tuple):
            actual_padding = padding  # 已经是元组
        else:  # Fallback, e.g. padding not string, kernel_size is int
            actual_padding = (kernel_size - 1) // 2 if k_is_int else 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=actual_padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

        self.act: Optional[nn.Module] = None  # 属性名与 Paddle 匹配
        self.lab: Optional[LearnableAffineBlock] = None  # 属性名与 Paddle 匹配
        if use_act:
            self.act = nn.ReLU()
            if use_lab:
                self.lab = LearnableAffineBlock(lr_mult=lr_mult)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.act(x)
        if self.lab:
            x = self.lab(x)
        return x


class LightConvBNAct(TheseusLayer):  # 与 Paddle 类名一致
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_lab: bool = False,
        lr_mult: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)  # 传递 kwargs 给 TheseusLayer (如果它处理)
        self.conv1 = ConvBNAct(
            in_channels,
            out_channels,
            1,
            padding=0,
            use_act=False,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        dw_padding = (kernel_size - 1) // 2
        self.conv2 = ConvBNAct(
            out_channels,
            out_channels,
            kernel_size,
            groups=out_channels,
            padding=dw_padding,
            use_act=True,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(self.conv1(x))


class StemBlock(TheseusLayer):  # 与 Paddle 类名一致
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        use_lab: bool = False,
        lr_mult: float = 1.0,
        text_rec: bool = False,
    ):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels, mid_channels, 3, 2, padding=1, use_lab=use_lab, lr_mult=lr_mult
        )
        self.stem2a = ConvBNAct(
            mid_channels,
            mid_channels // 2,
            2,
            1,
            padding="SAME",
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.stem2b = ConvBNAct(
            mid_channels // 2,
            mid_channels,
            2,
            1,
            padding="SAME",
            use_lab=use_lab,
            lr_mult=lr_mult,
        )

        stem3_stride_val = 1 if text_rec else 2
        self.stem3 = ConvBNAct(
            mid_channels * 2,
            mid_channels,
            3,
            stride=stem3_stride_val,
            padding=1,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.stem4 = ConvBNAct(
            mid_channels,
            out_channels,
            1,
            1,
            padding=0,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=1, padding=1, ceil_mode=True
        )  # 尝试近似 SAME for K=2,S=1

    def forward(self, x: Tensor) -> Tensor:
        x_s1 = self.stem1(x)
        x1_p = self.pool(x_s1)  # (N,C,H_s1+1, W_s1+1) with padding=1
        x2_s = self.stem2a(x_s1)
        x2_s = self.stem2b(x2_s)  # (N,C_mid, H_s1, W_s1)

        # 尺寸对齐，如果 MaxPool2d(padding=1) 导致尺寸变化
        if x1_p.shape[2:] != x2_s.shape[2:]:
            x1_p = F.adaptive_avg_pool2d(x1_p, (x2_s.size(2), x2_s.size(3)))

        x_cat = torch.cat([x1_p, x2_s], dim=1)
        x_out = self.stem3(x_cat)
        x_out = self.stem4(x_out)
        return x_out


class HGV2_Block(TheseusLayer):  # 与 Paddle 类名一致
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        layer_num: int = 6,
        identity: bool = False,
        light_block: bool = True,
        use_lab: bool = False,
        lr_mult: float = 1.0,
    ):
        super().__init__()
        self.identity = identity  # Paddle 属性名
        self.layers = nn.ModuleList()  # Paddle 属性名
        block_constructor = LightConvBNAct if light_block else ConvBNAct
        current_c = in_channels
        for i in range(layer_num):
            self.layers.append(
                block_constructor(
                    current_c if i == 0 else mid_channels,  # Paddle 逻辑
                    mid_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    use_lab=use_lab,
                    lr_mult=lr_mult,
                )
            )
            # current_c = mid_channels # Paddle 中这里没有更新 current_c, 因为下一层输入总是 mid_channels
            # (除了第一层是 in_channels)

        total_concat_c = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(
            total_concat_c,
            out_channels // 2,
            1,
            padding=0,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )
        self.aggregation_excitation_conv = ConvBNAct(
            out_channels // 2,
            out_channels,
            1,
            padding=0,
            use_lab=use_lab,
            lr_mult=lr_mult,
        )

    def forward(self, x: Tensor) -> Tensor:
        identity_map = x
        output_list = [x]  # Paddle: output = []; output.append(x)
        current_path_feat = x
        for layer_module in self.layers:  # 使用 self.layers
            current_path_feat = layer_module(current_path_feat)
            output_list.append(current_path_feat)

        concatenated = torch.cat(output_list, dim=1)  # Paddle: axis=1
        aggregated = self.aggregation_squeeze_conv(concatenated)
        aggregated = self.aggregation_excitation_conv(aggregated)

        if self.identity:  # 使用 self.identity
            aggregated = aggregated + identity_map
        return aggregated


class HGV2_Stage(TheseusLayer):  # 与 Paddle 类名一致
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        block_num: int,
        layer_num: int = 6,
        is_downsample: bool = True,
        light_block: bool = True,
        kernel_size: int = 3,
        use_lab: bool = False,
        stride: Union[int, Tuple[int, int]] = 2,
        lr_mult: float = 1.0,
    ):
        super().__init__()
        self.is_downsample = is_downsample  # Paddle 属性名
        self.downsample: Optional[ConvBNAct] = None  # Paddle 属性名
        if self.is_downsample:
            self.downsample = ConvBNAct(
                in_channels,
                in_channels,
                3,
                stride=stride,
                padding=1,
                groups=in_channels,
                use_act=False,
                use_lab=use_lab,
                lr_mult=lr_mult,
            )

        blocks_list_internal = []
        current_block_in_c = in_channels  # 输入给第一个 HGV2_Block 的通道数
        for i in range(block_num):
            blocks_list_internal.append(
                HGV2_Block(
                    in_channels=current_block_in_c
                    if i == 0
                    else out_channels,  # Paddle 逻辑
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    layer_num=layer_num,
                    identity=(i > 0),
                    light_block=light_block,
                    use_lab=use_lab,
                    lr_mult=lr_mult,
                )
            )
            # 第一个 block 的输入是 stage 的 in_channels (可能已下采样)
            # 后续 block 的输入是前一个 block 的 out_channels (即 stage 的 out_channels)
        self.blocks = nn.Sequential(*blocks_list_internal)  # Paddle 属性名

    def forward(self, x: Tensor) -> Tensor:
        if self.is_downsample and self.downsample:
            x = self.downsample(x)
        return self.blocks(x)


# --- PPHGNetV2 主干网络 (PyTorch 实现) ---
class PPHGNetV2(TheseusLayer):  # 与 Paddle 类名一致
    def __init__(
        self,
        stage_config: Dict[str, List[Any]],
        stem_channels: List[int] = [3, 32, 64],
        use_lab: bool = False,
        use_last_conv: bool = True,
        class_expand: int = 2048,
        dropout_prob: float = 0.0,
        class_num: int = 1000,
        lr_mult_list: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0],
        det: bool = False,
        text_rec: bool = False,
        out_indices: Optional[List[int]] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.det = det
        self.text_rec = text_rec
        self.use_lab = use_lab
        self.use_last_conv = use_last_conv
        self.class_expand = class_expand
        self.class_num = class_num

        # out_indices 与 Paddle 保持一致
        self.out_indices = (
            out_indices if out_indices is not None else ([0, 1, 2, 3] if det else [])
        )

        # self.out_channels 属性与 Paddle 保持一致
        self.out_channels: Union[List[int], int] = []  # det 时是列表，否则是标量

        self.stem = StemBlock(
            stem_channels[0],
            stem_channels[1],
            stem_channels[2],
            use_lab,
            lr_mult_list[0],
            text_rec,
        )

        self.stages = nn.ModuleList()
        # current_in_c_for_stage = stem_channels[2] # Paddle 中 Stage 的 in_channels 直接来自 config

        stage_keys_ordered = list(stage_config.keys())
        if not isinstance(stage_config, OrderedDict) and all(
            k.startswith("stage") for k in stage_keys_ordered
        ):
            stage_keys_ordered.sort(key=lambda x: int(x.replace("stage", "")))

        last_stage_final_out_channels = 0  # 用于分类头的输入通道
        for i, stage_key in enumerate(stage_keys_ordered):
            s_cfg = stage_config[stage_key]
            s_in, s_mid, s_out, s_nbl, s_isds, s_lb, s_k, s_ln, s_std = s_cfg

            # 在 PyTorch nn.ModuleList 中，子模块通常是按索引访问，或通过迭代
            # 如果要按名称访问 (如 self.stages.stage1)，则应使用 nn.ModuleDict 或直接设为属性
            # Paddle nn.LayerList 可以通过索引或名称 (如果 add_sublayer 时提供了名称)
            # 这里我们保持 ModuleList，并通过迭代或索引访问
            current_stage_module = HGV2_Stage(
                in_channels=s_in,
                mid_channels=s_mid,
                out_channels=s_out,
                block_num=s_nbl,
                layer_num=s_ln,
                is_downsample=s_isds,
                light_block=s_lb,
                kernel_size=s_k,
                use_lab=use_lab,
                stride=s_std,  # PyTorch ConvBNAct 会处理 stride tuple
                lr_mult=lr_mult_list[i + 1] if (i + 1) < len(lr_mult_list) else 1.0,
            )
            self.stages.append(current_stage_module)

            if (
                i in self.out_indices and self.det
            ):  # Paddle: self.out_channels.append(out_channels)
                if isinstance(self.out_channels, list):
                    self.out_channels.append(s_out)
            last_stage_final_out_channels = s_out

        if not self.det:  # Paddle: self.out_channels = stage_config["stage4"][2]
            # 确保 stage_config 中有 "stage4" 并且它有正确的索引
            if "stage4" in stage_config and len(stage_config["stage4"]) > 2:
                self.out_channels = stage_config["stage4"][2]
            else:  # Fallback to the last configured stage's output
                self.out_channels = last_stage_final_out_channels
                print(
                    f"警告: PPHGNetV2 非det模式，'stage4' 未在 stage_config 中找到或格式不符。"
                    f"self.out_channels 回退到最后一个 stage 的输出通道数: {last_stage_final_out_channels}"
                )

        # 分类头
        # Paddle: self.avg_pool = AdaptiveAvgPool2D(1)
        # PyTorch: AdaptiveAvgPool2D 类已定义
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 与 Paddle AdaptiveAvgPool2D 行为一致

        if (
            self.use_last_conv
            and not self.det
            and not self.text_rec
            and self.class_num > 0
        ):
            # Paddle: self.last_conv = Conv2D(in_channels=out_channels, ...)
            # 这里的 out_channels 是 for 循环中最后一个 stage 的 out_channels
            self.last_conv = nn.Conv2d(
                last_stage_final_out_channels, self.class_expand, 1, 1, 0, bias=False
            )
            self.act = nn.ReLU()
            self.lab: Optional[LearnableAffineBlock] = (
                LearnableAffineBlock() if use_lab else None
            )
            self.dropout = nn.Dropout(
                p=dropout_prob
            )  # Paddle mode="downscale_in_infer" 是 PyTorch 默认
        else:
            self.last_conv, self.act, self.lab, self.dropout = None, None, None, None

        # Paddle: self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.flatten = nn.Flatten(start_dim=1)  # PyTorch stop_dim 默认是 -1

        if not self.det and not self.text_rec and self.class_num > 0:
            # Paddle: self.fc = nn.Linear(self.class_expand if self.use_last_conv else out_channels, self.class_num)
            fc_in = (
                self.class_expand
                if self.use_last_conv and self.last_conv
                else last_stage_final_out_channels
            )
            self.fc = nn.Linear(fc_in, self.class_num)
        else:
            self.fc = None

        self._init_weights()

    def _init_weights(self):  # 与 Paddle 一致
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Paddle: zeros_(m.bias), kaiming_normal_ for weight if from ConvBNAct
                # Paddle PPHGNetV2 _init_weights 中 Linear 偏置为0, 权重默认 (xavier for some)
                nn.init.normal_(m.weight, 0, 0.01)  # 保持之前的初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # 与 Paddle 一致

    def forward(self, x: Tensor) -> Union[Tensor, List[Tensor], DonutSwinModelOutput]:
        x = self.stem(x)

        outputs_det = []
        current_stage_idx = 0
        for stage_module in self.stages:  # 直接迭代 ModuleList
            x = stage_module(x)
            if self.det and current_stage_idx in self.out_indices:
                outputs_det.append(x)
            current_stage_idx += 1

        if self.det:
            return outputs_det

        if self.text_rec:
            if self.training:
                x = F.adaptive_avg_pool2d(x, (1, 40))
            else:  # 推理时 avg_pool2d([3,2])
                k_h = 3 if x.size(2) >= 3 else x.size(2)
                k_w = 2 if x.size(3) >= 2 else x.size(3)
                if k_h > 0 and k_w > 0:
                    x = F.avg_pool2d(x, kernel_size=(k_h, k_w), stride=(k_h, k_w))
            return x  # 返回特征图

        # 分类模式 (not det, not text_rec)
        if self.fc:  # 只有当分类头存在时才执行
            if self.avg_pool:
                x = self.avg_pool(x)
            if self.last_conv:
                x = self.last_conv(x)
                if self.act:
                    x = self.act(x)
                if self.lab:
                    x = self.lab(x)
                if self.dropout:
                    x = self.dropout(x)
            if self.flatten:
                x = self.flatten(x)
            x = self.fc(x)
        return x  # 返回 logits 或最后一个 stage 的特征（如果分类头不存在）


# --- 工厂函数 (与 Paddle 对应) ---
MODEL_URLS = {  # 这些 URL 指向 Paddle 参数，转换后不能直接用
    "PPHGNetV2_B0": "...",
    "PPHGNetV2_B1": "...",
    "PPHGNetV2_B2": "...",
    "PPHGNetV2_B3": "...",
    "PPHGNetV2_B4": "...",
    "PPHGNetV2_B5": "...",
    "PPHGNetV2_B6": "...",
}
__all__ = list(MODEL_URLS.keys())  # 导出的模型名称
__all__.append("PPHGNetV2")  # 也导出主类
__all__.extend(["PPHGNetV2_B4_Formula", "PPHGNetV2_B6_Formula"])


def PPHGNetV2_B0(pretrained=False, use_ssld=False, **kwargs):
    stage_config = OrderedDict(
        [  # 使用 OrderedDict 保证顺序
            (
                "stage1",
                [16, 16, 64, 1, False, False, 3, 3, 2],
            ),  # stride for downsample if is_downsample=True
            ("stage2", [64, 32, 256, 1, True, False, 3, 3, 2]),
            ("stage3", [256, 64, 512, 2, True, True, 5, 3, 2]),
            ("stage4", [512, 128, 1024, 1, True, True, 5, 3, 2]),
        ]
    )
    # Paddle PPHGNetV2 kwargs 会覆盖默认值，例如 class_num
    # PyTorch 实现中，kwargs 也会传递给 PPHGNetV2 构造函数
    model = PPHGNetV2(
        stem_channels=[3, 16, 16], stage_config=stage_config, use_lab=True, **kwargs
    )
    # pretrained 和 use_ssld 的处理需要单独的权重加载逻辑
    return model


def PPHGNetV2_B1(pretrained=False, use_ssld=False, **kwargs):
    stage_config = OrderedDict(
        [
            ("stage1", [32, 32, 64, 1, False, False, 3, 3, 2]),
            ("stage2", [64, 48, 256, 1, True, False, 3, 3, 2]),
            ("stage3", [256, 96, 512, 2, True, True, 5, 3, 2]),
            ("stage4", [512, 192, 1024, 1, True, True, 5, 3, 2]),
        ]
    )
    model = PPHGNetV2(
        stem_channels=[3, 24, 32], stage_config=stage_config, use_lab=True, **kwargs
    )
    return model


def PPHGNetV2_B2(pretrained=False, use_ssld=False, **kwargs):
    stage_config = OrderedDict(
        [
            ("stage1", [32, 32, 96, 1, False, False, 3, 4, 2]),
            ("stage2", [96, 64, 384, 1, True, False, 3, 4, 2]),
            ("stage3", [384, 128, 768, 3, True, True, 5, 4, 2]),
            ("stage4", [768, 256, 1536, 1, True, True, 5, 4, 2]),
        ]
    )
    model = PPHGNetV2(
        stem_channels=[3, 24, 32], stage_config=stage_config, use_lab=True, **kwargs
    )
    return model


def PPHGNetV2_B3(pretrained=False, use_ssld=False, **kwargs):
    stage_config = OrderedDict(
        [
            ("stage1", [32, 32, 128, 1, False, False, 3, 5, 2]),
            ("stage2", [128, 64, 512, 1, True, False, 3, 5, 2]),
            ("stage3", [512, 128, 1024, 3, True, True, 5, 5, 2]),
            ("stage4", [1024, 256, 2048, 1, True, True, 5, 5, 2]),
        ]
    )
    model = PPHGNetV2(
        stem_channels=[3, 24, 32], stage_config=stage_config, use_lab=True, **kwargs
    )
    return model


def PPHGNetV2_B4(pretrained=False, use_ssld=False, det=False, text_rec=False, **kwargs):
    # Paddle stage_config 中 stride 是第9个元素
    stage_config_rec = OrderedDict(
        [
            ("stage1", [48, 48, 128, 1, True, False, 3, 6, [2, 1]]),
            ("stage2", [128, 96, 512, 1, True, False, 3, 6, [1, 2]]),
            ("stage3", [512, 192, 1024, 3, True, True, 5, 6, [2, 1]]),
            ("stage4", [1024, 384, 2048, 1, True, True, 5, 6, [2, 1]]),
        ]
    )
    stage_config_det = OrderedDict(
        [
            (
                "stage1",
                [48, 48, 128, 1, False, False, 3, 6, 2],
            ),  # is_downsample=False, stride is for HGV2_Stage if true
            ("stage2", [128, 96, 512, 1, True, False, 3, 6, 2]),
            ("stage3", [512, 192, 1024, 3, True, True, 5, 6, 2]),
            ("stage4", [1024, 384, 2048, 1, True, True, 5, 6, 2]),
        ]
    )
    current_stage_config = (
        stage_config_rec if text_rec else stage_config_det
    )  # 优先 rec (如果 text_rec=True)
    if not text_rec and not det:  # 如果是分类模式，也用 det 的配置（通常通道数更通用）
        current_stage_config = stage_config_det

    model = PPHGNetV2(
        stem_channels=[3, 32, 48],
        stage_config=current_stage_config,
        use_lab=False,
        det=det,
        text_rec=text_rec,
        **kwargs,
    )
    return model


def PPHGNetV2_B5(pretrained=False, use_ssld=False, **kwargs):
    stage_config = OrderedDict(
        [
            ("stage1", [64, 64, 128, 1, False, False, 3, 6, 2]),
            ("stage2", [128, 128, 512, 2, True, False, 3, 6, 2]),
            ("stage3", [512, 256, 1024, 5, True, True, 5, 6, 2]),
            ("stage4", [1024, 512, 2048, 2, True, True, 5, 6, 2]),
        ]
    )
    model = PPHGNetV2(
        stem_channels=[3, 32, 64], stage_config=stage_config, use_lab=False, **kwargs
    )
    return model


def PPHGNetV2_B6(pretrained=False, use_ssld=False, **kwargs):
    stage_config = OrderedDict(
        [
            ("stage1", [96, 96, 192, 2, False, False, 3, 6, 2]),
            ("stage2", [192, 192, 512, 3, True, False, 3, 6, 2]),
            ("stage3", [512, 384, 1024, 6, True, True, 5, 6, 2]),
            ("stage4", [1024, 768, 2048, 3, True, True, 5, 6, 2]),
        ]
    )
    model = PPHGNetV2(
        stem_channels=[3, 48, 96], stage_config=stage_config, use_lab=False, **kwargs
    )
    return model


# --- Formula Recognizer specific wrappers ---
class PPHGNetV2_B4_Formula(nn.Module):  # 直接继承 nn.Module
    def __init__(
        self, in_channels: int = 3, class_num: int = 1000, **kwargs_for_backbone
    ):
        super().__init__()
        # self.in_channels = in_channels # PPHGNetV2_B4 会处理输入通道
        # self.out_channels = 2048 # 由 PPHGNetV2_B4 的 stage_config 决定

        # Formula Rec 通常是 text_rec=True, det=False
        # class_num 传递给 PPHGNetV2, 但如果 text_rec=True, PPHGNetV2 不会创建 fc 分类头
        self.pphgnet_b4 = PPHGNetV2_B4(
            text_rec=True, det=False, class_num=class_num, **kwargs_for_backbone
        )
        # 获取 backbone 输出的实际通道数
        self.feature_out_channels = self.pphgnet_b4.final_stage_out_channels

    def forward(
        self, input_data: Union[Tensor, List[Tensor], Dict[str, Tensor]]
    ) -> DonutSwinModelOutput:
        pixel_values: Tensor
        if (
            isinstance(input_data, list) and input_data
        ):  # 假设训练时是 (pixel_values, label, attention_mask)
            pixel_values = input_data[0]
        elif isinstance(input_data, dict):  # HuggingFace 风格
            pixel_values = input_data["pixel_values"]
        elif isinstance(input_data, Tensor):
            pixel_values = input_data
        else:
            raise ValueError(
                f"PPHGNetV2_B4_Formula 不支持的输入类型: {type(input_data)}"
            )

        if pixel_values.size(1) == 1:  # 单通道转三通道
            pixel_values = pixel_values.repeat(1, 3, 1, 1)

        # pphgnet_b4 在 text_rec=True 时返回 (N, C, H_pooled, W_pooled)
        features = self.pphgnet_b4(pixel_values)

        N, C, H, W = features.shape
        # 转换为 (N, SeqLen, HiddenDim) for DonutSwinModelOutput
        # SeqLen = H * W, HiddenDim = C
        last_hidden_state = features.flatten(2).permute(0, 2, 1)  # (N, H*W, C)

        output = DonutSwinModelOutput(
            last_hidden_state=last_hidden_state,
            # 其他字段可根据需要填充，通常 backbone 只提供 last_hidden_state
        )
        # 训练时，外部的 Trainer/Model 会处理 label 和 attention_mask
        return output


class PPHGNetV2_B6_Formula(nn.Module):  # 直接继承 nn.Module
    def __init__(
        self, in_channels: int = 3, class_num: int = 1000, **kwargs_for_backbone
    ):
        super().__init__()
        self.pphgnet_b6 = PPHGNetV2_B6(
            text_rec=True, det=False, class_num=class_num, **kwargs_for_backbone
        )
        self.feature_out_channels = self.pphgnet_b6.final_stage_out_channels

    def forward(
        self, input_data: Union[Tensor, List[Tensor], Dict[str, Tensor]]
    ) -> DonutSwinModelOutput:
        pixel_values: Tensor
        if isinstance(input_data, list) and input_data:
            pixel_values = input_data[0]
        elif isinstance(input_data, dict):
            pixel_values = input_data["pixel_values"]
        elif isinstance(input_data, Tensor):
            pixel_values = input_data
        else:
            raise ValueError(
                f"PPHGNetV2_B6_Formula 不支持的输入类型: {type(input_data)}"
            )

        if pixel_values.size(1) == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)

        features = self.pphgnet_b6(pixel_values)
        N, C, H, W = features.shape
        last_hidden_state = features.flatten(2).permute(0, 2, 1)

        return DonutSwinModelOutput(last_hidden_state=last_hidden_state)
