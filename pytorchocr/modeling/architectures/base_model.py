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
                num_character_classes = None  # 初始化
                char_dict_path = self.config.get("character_dict_path", None)

                if char_dict_path:
                    try:
                        with open(char_dict_path, "r", encoding="utf-8") as f:
                            char_list = [line.strip() for line in f if line.strip()]
                            num_character_classes = len(char_list)
                        print(
                            f"信息: 从 {char_dict_path} 加载字符字典成功, 找到 {num_character_classes} 个类别。"
                        )
                    except FileNotFoundError:
                        print(
                            f"警告: 字符字典文件未在 {char_dict_path} 找到。请确保路径正确且文件可访问。"
                        )
                    except Exception as e:
                        print(f"错误: 读取字符字典 {char_dict_path} 失败: {e}")

                if (
                    num_character_classes is None
                ):  # 如果 char_dict_path 未提供，或加载失败
                    if char_dict_path:  # 说明加载失败
                        print(
                            f"信息: 由于 '{char_dict_path}' 加载问题，尝试从配置中回退使用 'num_classes'。"
                        )
                    else:  # 说明 char_dict_path 未在 config 中提供
                        print(
                            "信息: 配置中未找到 'character_dict_path'。尝试使用 'num_classes'。"
                        )

                    if "num_classes" in self.config:
                        num_character_classes = self.config["num_classes"]
                        print(
                            f"信息: 从配置中使用 'num_classes': {num_character_classes}。"
                        )
                    else:
                        # 关键的后备：如果字典路径和 num_classes 都没有正确设置
                        default_fallback_classes = 6623  # PP-OCRv3/v4 的常见默认值
                        # 对于 PP-OCRv5，这个值是错误的。ppocrv5_dict.txt 大约有 18385 个字符。
                        print(
                            f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                        )
                        print(
                            f"!!! 严重警告: 模型配置中未找到 'character_dict_path' 且未找到 'num_classes'。"
                        )
                        print(f"!!! 将回退使用默认类别数: {default_fallback_classes}。")
                        print(f"!!! - 对于 PP-OCRv3/v4 模型，此默认值可能适用。")
                        print(
                            f"!!! - 对于 PP-OCRv5 系列模型 (如 ch_PP-OCRv5_rec_server), 此默认值几乎肯定是错误的。"
                        )
                        print(
                            f"!!!   PP-OCRv5 使用的 'ppocrv5_dict.txt' 约有 18385 个字符。"
                        )
                        print(
                            f"!!!   错误的类别数将导致 CTCHead 和 NRTRHead (GTC) 输出层维度不匹配，权重无法正确加载!"
                        )
                        print(f"!!! 请立即检查您的模型配置文件 (例如 .yml 或 .py):")
                        print(
                            f"!!! 1. 确保 'character_dict_path' 正确指向了您模型对应的字符字典文件 (例如 'ppocrv5_dict.txt')。"
                        )
                        print(
                            f"!!! 2. 或者，确保 'num_classes' 被正确设置为字典中的字符数量 (例如，对于 ppocrv5_dict.txt 约为 18385)。"
                        )
                        print(
                            f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                        )
                        num_character_classes = default_fallback_classes

                # 强制设置PP-OCRv5的num_classes为18384
                if self.config.get("model_name", "").lower().startswith("pp-ocrv5"):
                    num_character_classes = 18384
                    print(
                        f"信息: 检测到PP-OCRv5模型，强制设置num_classes为{num_character_classes}"
                    )

                # 确保 num_character_classes 是一个正整数
                if (
                    not isinstance(num_character_classes, int)
                    or num_character_classes <= 0
                ):
                    final_fallback_classes = 18384  # 使用PP-OCRv5的默认值
                    print(
                        f"严重错误: num_character_classes 无效 ({num_character_classes})。将默认设置为 {final_fallback_classes}。"
                    )
                    num_character_classes = final_fallback_classes

                out_channels_list_for_multihead = {
                    "CTCLabelDecode": 18385,  # PP-OCRv5固定18384+1
                    "NRTRLabelDecode": 18388,  # PP-OCRv5固定18384+4
                    "SARLabelDecode": 18387,  # PP-OCRv5固定18384+3
                }
                head_config["out_channels_list"] = out_channels_list_for_multihead

            self.head = build_head(
                head_config, **kwargs
            )  # **kwargs 是从BaseModel构造函数传来的额外参数

        self.return_all_feats = config.get("return_all_feats", False)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        初始化模型中不同类型层的权重。
        对 Conv2d 和 ConvTranspose2d 使用 Kaiming Normal 初始化，
        对 BatchNorm2d 权重使用全1初始化，偏置项使用全0初始化，
        对 Linear 层权重使用正态分布初始化。
        """
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
