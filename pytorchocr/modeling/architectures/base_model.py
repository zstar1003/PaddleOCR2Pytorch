from torch import nn

from ..backbones import build_backbone
from ..heads import build_head
from ..necks import build_neck


class BaseModel(nn.Module):
    def __init__(self, config, **kwargs):
        """
        the module for OCR.
        args:
            config (dict): the super parameters for module.
        """
        super(BaseModel, self).__init__()

        self.config = config  # 保存整个配置，方便后续使用

        in_channels_param = config.get("in_channels", 3)  # 初始输入通道，通常是3 (RGB)
        model_type = config["model_type"]

        # build backbone, backbone is need for det, rec and cls
        current_in_channels = in_channels_param
        if "Backbone" not in config or config["Backbone"] is None:
            self.use_backbone = False
            # 如果没有backbone，MultiHead的in_channels需要从config的in_channels_param获取
            # 但通常识别模型总是有backbone
        else:
            self.use_backbone = True
            config["Backbone"]["in_channels"] = (
                current_in_channels  # 设置backbone的输入通道
            )
            self.backbone = build_backbone(config["Backbone"], model_type)
            current_in_channels = (
                self.backbone.out_channels
            )  # 更新当前通道数为backbone的输出

        # build neck
        # for rec, neck can be cnn,rnn or reshape(None)
        # for det, neck can be FPN, BIFPN and so on.
        # for cls, neck should be none
        if "Neck" not in config or config["Neck"] is None:
            self.use_neck = False
        else:
            self.use_neck = True
            config["Neck"]["in_channels"] = current_in_channels  # 设置neck的输入通道
            self.neck = build_neck(config["Neck"])
            current_in_channels = self.neck.out_channels  # 更新当前通道数为neck的输出

        # build head, head is need for det, rec and cls
        if "Head" not in config or config["Head"] is None:
            self.use_head = False
        else:
            self.use_head = True
            head_config = config["Head"].copy()  # 复制一份，避免修改原始config
            head_config["in_channels"] = current_in_channels  # 设置head的输入通道

            # 特别处理 MultiHead 的 out_channels_list
            if head_config.get("name") == "MultiHead":
                # 从全局配置中获取字符字典路径或字符列表
                # 你需要确保 'character_dict_path' 在你的主 config 中定义
                char_dict_path = self.config.get("character_dict_path", None)
                if char_dict_path:
                    try:
                        with open(char_dict_path, "r", encoding="utf-8") as f:
                            # 移除空行和BOS/EOS（如果字典里有但模型不需要它们计数）
                            char_list = [line.strip() for line in f if line.strip()]
                            # 假设字典里不包含CTC blank, NRTR/SAR的特殊token
                            num_character_classes = len(char_list)
                    except FileNotFoundError:
                        print(
                            f"Warning: Character dictionary file not found at {char_dict_path}. Using default class count."
                        )
                        num_character_classes = 90  # 一个备用值
                else:
                    # 如果没有提供字符字典路径，你可能需要一个默认值或者从其他地方获取
                    # 例如，对于PaddleOCR预训练模型，类别数是固定的
                    # ch_PP-OCRv3 server_rec (MultiHead) 使用的字符集大小是 6623 + blank = 6624 (CTC)
                    # NRTR/SAR 可能需要不同的特殊token数量
                    print(
                        "Warning: 'character_dict_path' not found in config. Inferring class counts."
                    )
                    # 这是一个示例，你需要根据你的模型和字符集来确定这些值
                    # 例如，如果你的模型是基于中英文的，那么 num_character_classes 可能是 6623
                    num_character_classes = self.config.get(
                        "num_classes", 6623
                    )  # 尝试从config获取类别数

                out_channels_list_for_multihead = {
                    "CTCLabelDecode": num_character_classes + 1,  # +1 for CTC blank
                    "NRTRLabelDecode": num_character_classes
                    + 2,  # 示例: +2 for start/end tokens for NRTR
                    "SARLabelDecode": num_character_classes
                    + 3,  # 示例: +3 for start/end/unknown for SAR
                    # 根据你的MultiHead中实际包含的解码头来调整这些键和值
                }
                head_config["out_channels_list"] = out_channels_list_for_multihead

            self.head = build_head(
                head_config, **kwargs
            )  # **kwargs 是从BaseModel构造函数传来的额外参数

        self.return_all_feats = config.get("return_all_feats", False)

        self._initialize_weights()

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, data=None):  # 保持和 PaddleOCR 接口一致，data 对应 targets
        y = dict()
        if self.use_backbone:
            x = self.backbone(
                x
            )  # x 的形状可能是 (N, C, H, W) 或 (N, SeqLen, C) 取决于backbone

        # 对于识别模型，backbone 的输出通常是特征图 (N, C, H, W)
        # PPHGNetV2(text_rec=True) 输出 (N, C, H_pooled, W_pooled)
        # 这个 x 会直接传递给 MultiHead
        # MultiHead 内部的 CTCHead->Neck(svtr) 会处理这个特征图

        # 在你的原始 Paddle BaseModel 中，Neck 是可选的，并且在 Rec 模型中，
        # Neck 通常是集成在 Head 内部的（比如 CTCHead 包含一个 SVTR Neck）。
        # 所以这里的 self.neck 可能不用于基于 MultiHead 的识别模型。
        # 如果 MultiHead 期望直接从 backbone 获取特征，则不应使用外部的 self.neck。
        # 你的 MultiHead 实现中的 CTCHead 已经包含了 Neck (SequenceEncoder)，所以外部的 Neck 应该不被使用。

        # 检查模型类型，如果 rec 且 Head 是 MultiHead，可能跳过外部 Neck
        is_multihead_rec = (
            self.config.get("model_type") == "rec"
            and self.config.get("Head", {}).get("name") == "MultiHead"
        )

        if (
            self.use_neck and not is_multihead_rec
        ):  # 只有当不是 MultiHead Rec 或 Neck 明确配置时才使用
            if isinstance(x, dict):  # 如果backbone输出是字典，取出主要特征
                x_for_neck = x.get("backbone_out", x.get(list(x.keys())[0]))
            else:
                x_for_neck = x
            x_after_neck = self.neck(x_for_neck)
            if isinstance(x, dict):
                # 如果原始x是字典，将neck的输出（可能是字典或张量）合并或替换
                if isinstance(x_after_neck, dict):
                    x.update(x_after_neck)
                else:
                    x["neck_out"] = x_after_neck  # 将neck的输出放入字典
                # 更新主路径上的 x 为 neck 的主要输出（如果 neck 输出字典）
                if isinstance(x_after_neck, dict):
                    x = x_after_neck.get(
                        "neck_out", x_after_neck.get(list(x_after_neck.keys())[0])
                    )
                else:
                    x = x_after_neck  # 更新主路径上的 x
            else:  # 如果原始x是张量
                x = x_after_neck  # 更新主路径上的 x
            # y["neck_out"] = x # y字典的更新在后面统一处理

        # 将骨干网络的输出（可能经过外部Neck处理）存入y字典
        # MultiHead 期望的是一个 Tensor x
        if isinstance(x, dict):
            # 如果 x 变成了一个字典（例如backbone或neck返回字典），需要取出主特征给head
            # 假设主特征的键是 'backbone_out' 或 'neck_out'，或者字典的第一个元素
            main_feature_key = (
                "neck_out"
                if (self.use_neck and not is_multihead_rec)
                else "backbone_out"
            )
            if main_feature_key in x:
                head_input = x[main_feature_key]
            else:  # 尝试取字典的第一个值
                head_input = list(x.values())[0]
            y.update(x)  # 把 backbone/neck 的所有输出都先存起来
        else:
            head_input = x
            if self.use_neck and not is_multihead_rec:
                y["neck_out"] = x
            else:
                y["backbone_out"] = x

        if self.use_head:
            if data is None:  # forward(x)
                head_output = self.head(head_input)
            else:  # forward(x, targets)
                head_output = self.head(head_input, data)
        else:
            head_output = head_input  # 如果没有head，直接输出

        # 更新y字典，处理head的输出
        if isinstance(head_output, dict):
            # MultiHead 在训练时会返回类似 {'ctc': ctc_pred, 'gtc': gtc_pred, 'ctc_neck': ...} 的字典
            # PaddleOCR期望的最终输出（如果不是return_all_feats）是预测结果
            y.update(head_output)
            # 确保有一个主预测键，比如 'predict' 或 'head_out'
            if "predict" not in y and "ctc" in y:  # MultiHead输出的ctc本身也是一个dict
                if isinstance(y["ctc"], dict) and "predict" in y["ctc"]:
                    y["predict"] = y["ctc"]["predict"]  # 将CTC的预测作为主预测
                else:
                    y["predict"] = y["ctc"]  # 或者CTC的全部输出
            final_output_for_return = y.get("predict", head_output)  # 获取主预测
        else:
            y["head_out"] = head_output
            final_output_for_return = head_output

        if self.return_all_feats:
            return y  # 训练时或需要所有中间特征时返回整个y字典
        else:
            # 推理时，通常只返回最终的预测结果
            return final_output_for_return
