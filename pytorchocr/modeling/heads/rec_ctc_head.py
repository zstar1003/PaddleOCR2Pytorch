import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Any, Tuple, Union  # 增加了 Union 和 Tuple


class CTCHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,  # 移除默认值，使其成为必需参数
        fc_decay: float = 0.0004,  # L2 decay，PyTorch中通常在优化器层面处理
        mid_channels: Optional[int] = None,
        return_feats: bool = False,
        apply_paddle_init: bool = False,  # 新增参数控制是否应用Paddle风格初始化
        **kwargs,
    ):
        super(CTCHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fc_decay = fc_decay  # 保存L2 decay值，可供外部优化器使用
        self.mid_channels = mid_channels
        self.return_feats = return_feats

        if mid_channels is None:
            self.fc = nn.Linear(
                in_channels, out_channels, bias=True
            )  # Paddle bias_attr 存在
            if apply_paddle_init:
                stdv = 1.0 / math.sqrt(in_channels * 1.0)
                nn.init.uniform_(self.fc.weight, -stdv, stdv)
                if self.fc.bias is not None:
                    nn.init.uniform_(self.fc.bias, -stdv, stdv)
        else:
            self.fc1 = nn.Linear(in_channels, mid_channels, bias=True)
            if apply_paddle_init:
                stdv1 = 1.0 / math.sqrt(in_channels * 1.0)
                nn.init.uniform_(self.fc1.weight, -stdv1, stdv1)
                if self.fc1.bias is not None:
                    nn.init.uniform_(self.fc1.bias, -stdv1, stdv1)

            self.fc2 = nn.Linear(mid_channels, out_channels, bias=True)
            if apply_paddle_init:
                stdv2 = 1.0 / math.sqrt(mid_channels * 1.0)
                nn.init.uniform_(self.fc2.weight, -stdv2, stdv2)
                if self.fc2.bias is not None:
                    nn.init.uniform_(self.fc2.bias, -stdv2, stdv2)

    def forward(
        self, x: Tensor, labels: Optional[Any] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # `labels` 参数 (对应 Paddle 的 `targets`) 在此模块的 forward 中未使用

        feats_for_return = x  # 如果没有 mid_channels 且需要返回特征，返回原始 x
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x_after_fc1 = self.fc1(x)
            predicts = self.fc2(x_after_fc1)
            feats_for_return = x_after_fc1  # 如果有 mid_channels，返回 fc1 的输出

        # 根据 Paddle 实现，推理时 result 总是被 softmax 后的 predicts 覆盖
        if not self.training:
            predicts_softmax = F.softmax(predicts, dim=2)
            return predicts_softmax  # 推理时，无论 return_feats 如何，都返回 softmax 后结果
        else:  # 训练模式
            if self.return_feats:
                return (feats_for_return, predicts)  # 返回 (特征, logits)
            else:
                return predicts  # 返回 logits
