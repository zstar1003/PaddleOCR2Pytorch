from __future__ import absolute_import, division, print_function

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Union

# 外部依赖：来自 necks 目录
from ..necks.rnn import (
    Im2Seq,  # 需要 PyTorch 版本的 Im2Seq
    SequenceEncoder,  # 需要 PyTorch 版本的 SequenceEncoder
)

# 头部实现：从各自的文件导入 PyTorch 版本
# 假设您的 PyTorch 版本类名与 Paddle 版本一致，或者您已相应重命名
# 例如，如果 PyTorch 版本是 CTCHead_PT，则导入 CTCHead_PT as CTCHead
from .rec_ctc_head import CTCHead  # 假设这是您实现的 PyTorch CTCHead
from .rec_sar_head import SARHead  # 假设这是您实现的 PyTorch SARHead
from .rec_nrtr_head import (
    Transformer as NRTRTransformer,
)  # 将 NRTR 的 Transformer 导入并重命名以避免与 torch.nn.Transformer 冲突


# --- 辅助类 (与 Paddle MultiHead 直接相关) ---
def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    """截断正态初始化，用于 AddPos。"""

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("警告: trunc_normal_ 均值与 [a,b] 区间距离 > 2 std")
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def zeros_(tensor: Tensor) -> Tensor:
    """用零填充张量。"""
    with torch.no_grad():
        tensor.zero_()
        return tensor


class FCTranspose(nn.Module):
    """对应 Paddle 的 FCTranspose，用于 NRTR 的 before_gtc。"""

    def __init__(
        self, in_channels: int, out_channels: int, only_transpose: bool = False
    ):
        super().__init__()
        self.only_transpose = only_transpose
        if not self.only_transpose:
            self.fc = nn.Linear(
                in_channels, out_channels, bias=False
            )  # Paddle: bias_attr=False

    def forward(self, x: Tensor) -> Tensor:
        # 输入 x 假设计为 (N, C, S) from Flatten(2)
        x = x.permute(0, 2, 1)  # (N, S, C)
        return x if self.only_transpose else self.fc(x)


class AddPos(nn.Module):
    """对应 Paddle 的 AddPos，用于 NRTR 的 before_gtc 可选位置编码。"""

    def __init__(self, dim: int, w: int):  # w 是序列最大长度
        super().__init__()
        self.dec_pos_embed = nn.Parameter(torch.empty(1, w, dim))
        zeros_(self.dec_pos_embed)  # Paddle: default_initializer=zeros_
        trunc_normal_(self.dec_pos_embed)  # Paddle: trunc_normal_(self.dec_pos_embed)

    def forward(self, x: Tensor) -> Tensor:  # x: (N, S, dim)
        # Paddle: x = x + self.dec_pos_embed[:, : x.shape[1], :]
        return x + self.dec_pos_embed[:, : x.shape[1], :]


class MultiHead(nn.Module):
    """多头识别模型的 PyTorch 实现。"""

    def __init__(
        self, in_channels: int, out_channels_list: Dict[str, int], **kwargs: Any
    ):
        super().__init__()
        self.head_list_config = kwargs.pop("head_list")  # 从 kwargs 获取头部配置列表
        self.use_pool = kwargs.get("use_pool", False)
        self.use_pos = kwargs.get("use_pos", False)
        self.in_channels = in_channels

        if self.use_pool:
            # Paddle: nn.AvgPool2D(kernel_size=[3, 2], stride=[3, 2], padding=0)
            self.pool = nn.AvgPool2d(kernel_size=(3, 2), stride=(3, 2), padding=0)
        else:
            self.pool = None

        # Paddle: self.gtc_head = "sar" (作为实例属性追踪 GTC 类型)
        # PyTorch: 我们用一个变量来追踪，并在 forward 中使用
        self.active_gtc_head_module_name = "sar_head"  # 默认 GTC 对应的属性名

        assert len(self.head_list_config) >= 2, "MultiHead 至少需要 2 个头配置"

        # 初始化头部属性为 None
        self.sar_head: Optional[SARHead] = None
        self.nrtr_transformer: Optional[NRTRTransformer] = (
            None  # NRTR 的 Transformer 组件
        )
        self.nrtr_before_gtc: Optional[nn.Sequential] = None  # NRTR 的前置处理
        self.ctc_encoder: Optional[SequenceEncoder] = None
        self.ctc_head: Optional[CTCHead] = None
        # self.encoder_reshape: Optional[Im2Seq] = None # Paddle CTC 分支中的 Im2Seq

        # 根据配置列表实例化各个头部
        for head_config_entry in self.head_list_config:
            head_type_name_in_config = list(head_config_entry.keys())[
                0
            ]  # 例如 "SARHead", "NRTRHead", "CTCHead"
            specific_args = head_config_entry[head_type_name_in_config]

            if head_type_name_in_config == "SARHead":
                # 使用导入的 SARHead (PyTorch 版本)
                self.sar_head = SARHead(
                    in_channels=self.in_channels,
                    out_channels=out_channels_list["SARLabelDecode"],
                    **specific_args,  # 传递 SARHead 特定的所有配置参数
                )
                self.active_gtc_head_module_name = "sar_head"  # 对应的属性名
            elif head_type_name_in_config == "NRTRHead":
                nrtr_dim = specific_args.get("nrtr_dim", 256)
                # Paddle: pos_max_len=80 (硬编码在 AddPos 调用处)
                # PyTorch: 从配置获取或保持一致
                pos_max_len_for_addpos = specific_args.get("pos_max_len", 80)

                before_gtc_layers = [
                    nn.Flatten(start_dim=2),  # (N,C,H,W) -> (N,C,H*W)
                    FCTranspose(
                        self.in_channels, nrtr_dim
                    ),  # (N,C,H*W) -> (N,H*W,nrtr_dim)
                ]
                if self.use_pos:
                    before_gtc_layers.append(AddPos(nrtr_dim, pos_max_len_for_addpos))
                self.nrtr_before_gtc = nn.Sequential(*before_gtc_layers)

                # 使用导入的 NRTRTransformer (PyTorch 版本)
                # Paddle: self.gtc_head = Transformer(...)
                self.nrtr_head_transformer = NRTRTransformer(
                    d_model=nrtr_dim,
                    nhead=specific_args.get("nhead", nrtr_dim // 32),
                    # Paddle: num_encoder_layers=-1 表示无编码器
                    num_encoder_layers=specific_args.get("num_encoder_layers", 0)
                    if specific_args.get("num_encoder_layers", 0) >= 0
                    else 0,
                    num_decoder_layers=specific_args.get(
                        "num_decoder_layers", 6
                    ),  # Paddle 默认 6
                    max_len=specific_args.get("max_text_length", 25),
                    dim_feedforward=specific_args.get(
                        "dim_feedforward", nrtr_dim * 4
                    ),  # Paddle 默认 1024
                    out_channels=out_channels_list["NRTRLabelDecode"],  # 纯字符类别数
                    # 从 specific_args 获取 Transformer 的其他参数
                    attention_dropout_rate=specific_args.get(
                        "attention_dropout_rate", 0.0
                    ),
                    residual_dropout_rate=specific_args.get(
                        "residual_dropout_rate", 0.1
                    ),
                    scale_embedding=specific_args.get("scale_embedding", True),
                    beam_size=specific_args.get("beam_size", 0)
                    if specific_args.get("beam_size", 0) >= 0
                    else 0,  # Paddle -1 -> PyTorch 0
                    **{
                        k: v
                        for k, v in specific_args.items()
                        if k
                        not in [  # 传递其他未明确列出的参数
                            "nrtr_dim",
                            "nhead",
                            "num_encoder_layers",
                            "num_decoder_layers",
                            "dim_feedforward",
                            "max_text_length",
                            "attention_dropout_rate",
                            "residual_dropout_rate",
                            "scale_embedding",
                            "beam_size",
                            "pos_max_len",
                        ]
                    },
                )
                self.active_gtc_head_module_name = (
                    "nrtr_head_transformer"  # 对应的属性名
                )
            elif head_type_name_in_config == "CTCHead":
                # Paddle: self.encoder_reshape = Im2Seq(in_channels)
                # Im2Seq 通常在 SequenceEncoder 之前或内部处理
                # 这里我们假设 SequenceEncoder 会处理输入形状
                # 如果需要显式的 Im2Seq，可以在 SequenceEncoder 外部或内部添加

                neck_config = specific_args["Neck"]
                encoder_type_name = neck_config.pop(
                    "name"
                )  # 从 neck_config 中移除 'name'
                self.ctc_encoder = SequenceEncoder(
                    in_channels=self.in_channels,  # SequenceEncoder 输入是原始特征图通道
                    encoder_type=encoder_type_name,
                    **neck_config,
                )

                ctc_head_specific_args = specific_args.get("Head", {})
                # 使用导入的 CTCHead (PyTorch 版本)
                self.ctc_head = CTCHead(
                    in_channels=self.ctc_encoder.out_channels,  # CTCHead 输入来自 Neck
                    out_channels=out_channels_list["CTCLabelDecode"],
                    **ctc_head_specific_args,  # 传递 CTCHead 特定的所有配置参数
                )
            else:
                raise NotImplementedError(
                    f"头部类型 {head_type_name_in_config} 当前在 MultiHead 中尚不支持"
                )

        # 校验必要的头部是否已初始化
        if not (self.ctc_head and self.ctc_encoder):
            raise ValueError(
                "CTCHead 及其编码器 (Neck) 必须在 head_list_config 中定义。"
            )
        if self.active_gtc_head_module_name == "sar_head" and not self.sar_head:
            raise ValueError("SARHead 被设为 GTC 头，但未初始化。")
        if self.active_gtc_head_module_name == "nrtr_head_transformer" and not (
            self.nrtr_head_transformer and self.nrtr_before_gtc
        ):
            raise ValueError(
                "NRTRHead 被设为 GTC 头，但其组件 (Transformer 或 before_gtc) 未完全初始化。"
            )

    def forward(self, x: Tensor, targets: Optional[List[Any]] = None) -> Dict[str, Any]:
        # x: (N, C, H, W) 来自骨干网络的特征图
        # targets: 目标标签列表，结构依赖于训练的头部

        feature_map_for_ctc_neck_and_sar = x  # SAR 和 CTC Neck 通常直接用骨干特征
        if self.pool:
            # Paddle 的 reshape: x.reshape([0, 3, -1, self.in_channels]).transpose([0, 3, 1, 2])
            # 这个 reshape + transpose 之后，输入给 pool 的是 (N, C, 3, W_pooled_in)
            # 然后池化 (kernel=[3,2]) 作用于 H=3, W=W_pooled_in
            # PyTorch 中，如果 x 已经是 (N,C,H,W)，且 H=3, W 是可被2整除的宽度
            if x.size(2) == 3:  # 检查高度是否为3
                # Paddle 的 reshape([0,3,-1,C]) -> transpose([0,3,1,2]) 似乎是为了将 C 维度放到前面
                # 如果 x 已经是 (N, C, H, W), 且 H=3, 那么直接池化
                feature_map_for_ctc_neck_and_sar = self.pool(
                    x
                )  # (N, C, 1, W_pooled_out)
            else:
                # 如果原始 paddle 代码的 reshape 是必须的，那意味着 x 的输入形状可能不是标准的 (N,C,H,W)
                # 或者 self.in_channels 指的不是 C 而是 reshape 后的某个维度
                # 这里假设，如果 use_pool=True，那么输入 x 的 H 维度应该是 3
                print(
                    f"警告: use_pool=True 但输入特征图高度 {x.size(2)} != 3。池化可能不会按预期工作或被跳过。"
                )
                # 可以选择在这里报错，或者跳过池化，或者尝试自适应池化（但这会改变逻辑）
                # 为了安全，如果高度不为3，我们暂时不应用这个特定的池化
                # feature_map_for_ctc_neck_and_sar = x # 跳过池化

        # CTC 分支 (总是计算)
        # Paddle: ctc_encoder = self.ctc_encoder(x) # x 是可能经过 pool 的特征
        # PyTorch:
        ctc_encoder_output = self.ctc_encoder(feature_map_for_ctc_neck_and_sar)

        # Paddle: ctc_out = self.ctc_head(ctc_encoder, targets)
        # PyTorch CTCHead.forward(x, labels=None), targets 在 Paddle CTCHead 中未使用
        ctc_head_output = self.ctc_head(
            ctc_encoder_output
        )  # targets 不直接传给 PyTorch CTCHead

        output_dict: Dict[str, Any] = {"ctc_neck": ctc_encoder_output}
        if isinstance(
            ctc_head_output, tuple
        ):  # (feats, predicts) from CTCHead when return_feats=True
            output_dict["ctc_feats"] = ctc_head_output[0]
            output_dict["ctc"] = {"predict": ctc_head_output[1]}
        else:  # predicts from CTCHead
            output_dict["ctc"] = {"predict": ctc_head_output}

        if not self.training:  # 推理模式
            # Paddle: return ctc_out (ctc_head 的原始输出，可能是 logits 或 softmax 后概率)
            # PyTorch CTCHead 推理时已 softmax
            return output_dict["ctc"]

        # GTC 分支 (SAR 或 NRTR), 仅在训练时计算
        # Paddle: targets[1:] 用于 GTC head
        gtc_specific_targets = targets[1:] if targets and len(targets) > 1 else None

        if self.active_gtc_head_module_name == "sar_head":
            if not self.sar_head:
                raise RuntimeError("SARHead 未初始化。")
            # SARHead 的输入特征图通常是未经 before_gtc 处理的原始或池化后特征图
            sar_predictions = self.sar_head(
                feature_map_for_ctc_neck_and_sar, gtc_specific_targets
            )
            output_dict["sar"] = sar_predictions  # SARHead.forward 返回字典
        elif self.active_gtc_head_module_name == "nrtr_head_transformer":
            if not (self.nrtr_before_gtc and self.nrtr_head_transformer):
                raise RuntimeError("NRTR 组件未初始化。")

            # NRTR 的输入是经过 before_gtc 处理的特征
            # before_gtc 期望的输入是原始的 x (或池化后的 x)
            nrtr_encoder_equivalent_input = self.nrtr_before_gtc(
                feature_map_for_ctc_neck_and_sar
            )

            nrtr_predictions = self.nrtr_head_transformer(
                nrtr_encoder_equivalent_input, gtc_specific_targets
            )
            output_dict["gtc"] = nrtr_predictions  # NRTR Transformer.forward 返回字典
        else:
            raise RuntimeError(f"未知的 GTC 头类型: {self.active_gtc_head_module_name}")

        return output_dict
