from __future__ import absolute_import, division, print_function

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple


# --- SAREncoder 的 PyTorch 实现 ---
class SAREncoder_PT(nn.Module):
    """SAR 编码器的 PyTorch 实现。"""

    def __init__(
        self,
        enc_bi_rnn: bool = False,
        enc_drop_rnn: float = 0.1,
        enc_gru: bool = False,
        d_model: int = 512,  # 来自主干网络的特征图通道数
        d_enc: int = 512,  # 编码器 RNN 层的隐藏大小
        mask: bool = True,  # 是否在 RNN 序列中使用 mask (基于 valid_ratios)
        **kwargs: Any,
    ):
        super().__init__()
        assert isinstance(enc_bi_rnn, bool)
        assert isinstance(enc_drop_rnn, (int, float)) and 0 <= enc_drop_rnn < 1.0
        assert isinstance(enc_gru, bool)
        assert isinstance(d_model, int)
        assert isinstance(d_enc, int)
        assert isinstance(mask, bool)

        self.enc_bi_rnn = enc_bi_rnn
        self.mask = mask

        # RNN 输入维度是 d_model (经过全局池化和维度调整后)
        rnn_input_size = d_model
        num_directions = 2 if enc_bi_rnn else 1

        # LSTM/GRU 编码器
        # Paddle: time_major=False -> PyTorch: batch_first=True
        rnn_kwargs = dict(
            input_size=rnn_input_size,
            hidden_size=d_enc,
            num_layers=2,  # Paddle 固定为 2 层
            batch_first=True,
            dropout=enc_drop_rnn
            if enc_drop_rnn > 0 and 2 > 1
            else 0,  # Dropout 只在 num_layers > 1 时有效
            bidirectional=enc_bi_rnn,
        )
        if enc_gru:
            self.rnn_encoder = nn.GRU(**rnn_kwargs)
        else:
            self.rnn_encoder = nn.LSTM(**rnn_kwargs)

        # 全局特征变换
        encoder_rnn_out_size = d_enc * num_directions
        self.linear = nn.Linear(encoder_rnn_out_size, encoder_rnn_out_size)

    def forward(self, feat: Tensor, img_metas: Optional[List[Any]] = None) -> Tensor:
        # feat: (N, C, H, W) - 来自主干网络的特征图
        # img_metas: 在 Paddle 中是 targets 列表。
        #            如果 self.mask=True, img_metas[-1] 被用作 valid_ratios。
        #            PyTorch 中, 我们假设 img_metas (如果提供) 是一个列表，
        #            并且 valid_ratios (如果需要) 在 img_metas[1] (如果 img_metas[0] 是标签)。

        valid_ratios: Optional[Tensor] = None
        if self.mask and img_metas is not None:
            # 尝试从 img_metas 获取 valid_ratios
            # Paddle 是 img_metas[-1]。如果 img_metas = [label, valid_ratio]，则是 img_metas[1]
            if len(img_metas) > 1 and isinstance(img_metas[1], Tensor):
                valid_ratios = img_metas[1]
            elif len(img_metas) == 1 and isinstance(
                img_metas[0], Tensor
            ):  # 可能只传了 valid_ratios (推理时)
                valid_ratios = img_metas[0]

        N, C, H, W = feat.shape
        # 全局最大池化 (沿高度 H)
        # Paddle: F.max_pool2d(feat, kernel_size=(h_feat, 1), stride=1, padding=0)
        # PyTorch: F.max_pool2d(feat, kernel_size=(H, 1)) 默认 stride=kernel_size, 所以指定 stride=1
        feat_v = F.max_pool2d(
            feat, kernel_size=(H, 1), stride=1, padding=0
        )  # (N, C, 1, W)
        feat_v = feat_v.squeeze(2)  # (N, C, W)
        feat_v = feat_v.permute(
            0, 2, 1
        )  # (N, W, C) - 作为 RNN 的输入序列 (batch_first=True)

        # RNN 编码器
        # output: (N, W, num_directions * d_enc)
        # h_n / c_n: (num_layers * num_directions, N, d_enc)
        holistic_feat_sequence, _ = self.rnn_encoder(feat_v)  # _ 是最后一个隐藏状态

        if valid_ratios is not None:
            valid_hf_list = []
            T_seq_len = holistic_feat_sequence.size(1)  # 序列长度 W
            for i in range(valid_ratios.size(0)):  # 遍历 batch
                # Paddle: valid_step = paddle.minimum(T, paddle.ceil(valid_ratios[i] * T).astype(T.dtype)) - 1
                # PyTorch:
                valid_len_float = valid_ratios[i] * T_seq_len
                # 向上取整得到有效长度，然后减1得到最后一个有效时间步的索引
                valid_step_idx = torch.ceil(valid_len_float).long()
                # 确保索引在有效范围内 [0, T_seq_len - 1]
                valid_step_idx = torch.clamp(valid_step_idx, 1, T_seq_len) - 1
                valid_hf_list.append(holistic_feat_sequence[i, valid_step_idx, :])
            holistic_feat_last_valid = torch.stack(
                valid_hf_list, dim=0
            )  # (N, num_directions * d_enc)
        else:
            holistic_feat_last_valid = holistic_feat_sequence[
                :, -1, :
            ]  # (N, num_directions * d_enc)

        holistic_feat_transformed = self.linear(
            holistic_feat_last_valid
        )  # (N, num_directions * d_enc)
        return holistic_feat_transformed


# --- BaseDecoder 的 PyTorch 等效 (直接继承 nn.Module) ---
class BaseDecoder_PT(nn.Module):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.train_mode = True  # 由 SARHead 的 forward 设置

    def forward_train(
        self,
        feat: Tensor,
        out_enc: Tensor,
        targets: Tensor,
        img_metas: Optional[List[Any]] = None,
    ) -> Tensor:
        raise NotImplementedError

    def forward_test(
        self, feat: Tensor, out_enc: Tensor, img_metas: Optional[List[Any]] = None
    ) -> Tensor:
        raise NotImplementedError

    def forward(
        self,
        feat: Tensor,
        out_enc: Tensor,  # out_enc 是 holistic_feat_encoder
        label: Optional[Tensor] = None,  # 训练时的目标序列
        img_metas: Optional[List[Any]] = None,  # 包含 valid_ratios 等
        train_mode: bool = True,
    ) -> Tensor:
        self.train_mode = train_mode  # SARHead 会设置这个
        if self.train_mode:
            if label is None:
                raise ValueError(
                    "BaseDecoder_PT.forward: 训练模式下需要 'label' (目标序列)。"
                )
            return self.forward_train(feat, out_enc, label, img_metas)
        return self.forward_test(feat, out_enc, img_metas)


# --- ParallelSARDecoder 的 PyTorch 实现 ---
class ParallelSARDecoder_PT(BaseDecoder_PT):
    def __init__(
        self,
        out_channels: int,  # 总类别数, e.g., 字符数 + unknown + start + padding
        enc_bi_rnn: bool = False,  # 编码器是否双向 (影响 holistic_feat_encoder 的维度)
        dec_bi_rnn: bool = False,
        dec_drop_rnn: float = 0.0,
        dec_gru: bool = False,
        d_model: int = 512,  # 主干网络特征图通道数 (feat)
        d_enc: int = 512,  # SAREncoder 的 RNN 隐藏层大小 (影响 holistic_feat_encoder)
        d_k: int = 64,  # 注意力模块内部通道数
        pred_dropout: float = 0.1,
        max_text_length: int = 30,
        mask: bool = True,  # 是否在特征图上使用 mask (通过 valid_ratios)
        pred_concat: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)  # 调用 BaseDecoder_PT 的 __init__

        self.num_classes = out_channels
        self.d_k = d_k
        # 根据 Paddle 实现，out_channels 已经包含了特殊 token
        # START_TOKEN = num_classes - 2
        # PADDING_TOKEN = num_classes - 1
        self.start_idx = out_channels - 2
        self.padding_idx = out_channels - 1
        self.max_seq_len = max_text_length
        self.mask = mask
        self.pred_concat = pred_concat

        encoder_rnn_out_size_actual = d_enc * (
            2 if enc_bi_rnn else 1
        )  # holistic_feat_encoder 的实际维度

        # 解码器 RNN 的输入是词嵌入，其维度通常与 holistic_feat_encoder 维度一致
        decoder_rnn_input_size = encoder_rnn_out_size_actual
        decoder_rnn_hidden_size = encoder_rnn_out_size_actual
        decoder_rnn_num_directions = 2 if dec_bi_rnn else 1
        decoder_rnn_output_size_actual = (
            decoder_rnn_hidden_size * decoder_rnn_num_directions
        )

        # 2D 注意力层
        self.conv1x1_1_attn_query = nn.Linear(decoder_rnn_output_size_actual, d_k)
        self.conv3x3_1_attn_key = nn.Conv2d(
            d_model, d_k, kernel_size=3, stride=1, padding=1
        )
        self.conv1x1_2_attn_energy = nn.Linear(d_k, 1)

        # 解码器 RNN 层
        rnn_kwargs_decoder = dict(
            input_size=decoder_rnn_input_size,
            hidden_size=decoder_rnn_hidden_size,
            num_layers=2,  # Paddle 固定为 2
            batch_first=True,
            dropout=dec_drop_rnn if dec_drop_rnn > 0 and 2 > 1 else 0,
            bidirectional=dec_bi_rnn,
        )
        if dec_gru:
            self.rnn_decoder = nn.GRU(**rnn_kwargs_decoder)
        else:
            self.rnn_decoder = nn.LSTM(**rnn_kwargs_decoder)

        # 解码器输入嵌入
        self.embedding = nn.Embedding(
            self.num_classes, encoder_rnn_out_size_actual, padding_idx=self.padding_idx
        )

        # 预测层
        self.pred_dropout_layer = nn.Dropout(pred_dropout)
        pred_num_output_classes = self.num_classes - 1  # 不预测 padding token

        if pred_concat:
            # concat(decoder_rnn_output, attention_feature_from_d_model, holistic_feature_encoder)
            fc_in_channel = (
                decoder_rnn_output_size_actual + d_model + encoder_rnn_out_size_actual
            )
        else:
            # 仅 attention_feature (来自 d_model)
            fc_in_channel = d_model
        self.prediction_fc = nn.Linear(fc_in_channel, pred_num_output_classes)

    def _2d_attention(
        self,
        decoder_rnn_input: Tensor,
        feat: Tensor,
        holistic_feat_encoder: Tensor,
        valid_ratios: Optional[Tensor] = None,
    ) -> Tensor:
        # decoder_rnn_input: (N, S_dec_current, EmbDim) - [holistic_feat, emb(SOS), emb(c1)...]
        # feat: (N, C_feat, H_feat, W_feat)
        # holistic_feat_encoder: (N, EncDim_actual)

        # 1. 解码器 RNN
        y, _ = self.rnn_decoder(
            decoder_rnn_input
        )  # (N, S_dec_current, DecoderRNNOutDim)

        # 2. 计算注意力权重
        attn_query = self.conv1x1_1_attn_query(y)  # (N, S_dec_current, d_k)
        N_bs, S_dec_curr, _ = attn_query.shape
        attn_query_expanded = attn_query.unsqueeze(3).unsqueeze(
            4
        )  # (N, S_dec_curr, d_k, 1, 1)

        attn_key = self.conv3x3_1_attn_key(feat)  # (N, d_k, H_feat, W_feat)
        attn_key_expanded = attn_key.unsqueeze(1)  # (N, 1, d_k, H_feat, W_feat)

        # Paddle: attn_weight = paddle.tanh(paddle.add(attn_key, attn_query))
        # PyTorch: add 会广播。attn_key_expanded (N,1,dk,H,W), attn_query_expanded (N,S,dk,1,1)
        # 结果 (N,S,dk,H,W)
        attn_energy_map = torch.tanh(attn_key_expanded + attn_query_expanded)

        # (N, S, H, W, dk) -> (N, S, H, W, 1)
        attn_energy = self.conv1x1_2_attn_energy(attn_energy_map.permute(0, 1, 3, 4, 2))

        _N, _T_steps, _H, _W, _C_energy = attn_energy.shape
        assert _C_energy == 1

        if valid_ratios is not None and self.mask:
            attn_energy_masked = attn_energy.clone()  # 避免原地修改
            for i in range(valid_ratios.size(0)):
                valid_width = torch.ceil(valid_ratios[i] * _W).long()
                valid_width = torch.clamp(valid_width, 0, _W)  # 确保在 [0, W]
                if valid_width < _W:
                    attn_energy_masked[i, :, :, valid_width:, :] = float("-inf")
            attn_energy_to_softmax = attn_energy_masked
        else:
            attn_energy_to_softmax = attn_energy

        attn_weight_flat = attn_energy_to_softmax.reshape(
            N_bs, S_dec_curr, -1
        )  # (N, S_dec_curr, H*W)
        attn_weight_softmax = F.softmax(attn_weight_flat, dim=-1)
        attn_weight_reshaped = attn_weight_softmax.reshape(N_bs, S_dec_curr, _H, _W, 1)
        attn_weight_final = attn_weight_reshaped.permute(
            0, 1, 4, 2, 3
        )  # (N, S_dec_curr, 1, H, W)

        feat_expanded = feat.unsqueeze(1)  # (N, 1, C_feat, H, W)
        # (N, S_dec_curr, C_feat)
        attn_feat = torch.sum(
            feat_expanded * attn_weight_final, dim=(3, 4), keepdim=False
        )

        # 3. 预测
        if self.pred_concat:
            # holistic_feat_encoder: (N, EncDim_actual)
            holistic_feat_expanded = holistic_feat_encoder.unsqueeze(1).expand(
                -1, S_dec_curr, -1
            )
            # y: (N, S_dec_curr, DecoderRNNOutDim)
            # attn_feat: (N, S_dec_curr, C_feat) (C_feat is d_model)
            combined_features = torch.cat(
                (y, attn_feat.type_as(y), holistic_feat_expanded.type_as(y)), dim=2
            )
            prediction_logits = self.prediction_fc(combined_features)
        else:
            prediction_logits = self.prediction_fc(attn_feat)

        return (
            self.pred_dropout_layer(prediction_logits)
            if self.train_mode
            else prediction_logits
        )

    def forward_train(
        self,
        feat: Tensor,
        holistic_feat_encoder: Tensor,
        label_tokens: Tensor,  # (N, S_label), e.g. [SOS, c1, ..., cN]
        img_metas: Optional[List[Any]] = None,
    ) -> Tensor:
        valid_ratios = None
        if (
            self.mask
            and img_metas
            and len(img_metas) > 1
            and isinstance(img_metas[1], Tensor)
        ):
            valid_ratios = img_metas[1]

        label_embeddings = self.embedding(label_tokens)  # (N, S_label, EmbDim)
        holistic_feat_unsqueezed = holistic_feat_encoder.unsqueeze(1).type_as(
            label_embeddings
        )
        # 解码器输入: [holistic_feat_encoder_emb, SOS_emb, char1_emb, ..., charN_emb]
        decoder_rnn_input = torch.cat(
            (holistic_feat_unsqueezed, label_embeddings), dim=1
        )

        # out_dec_logits: (N, 1+S_label, NumClasses-1)
        out_dec_logits = self._2d_attention(
            decoder_rnn_input, feat, holistic_feat_encoder, valid_ratios
        )

        # 移除对应 holistic_feat_encoder 输入产生的第一个时间步的预测
        # 目标是预测 S_label 长度的序列 (对应 label_tokens)
        return out_dec_logits[:, 1:, :]  # (N, S_label, NumClasses-1)

    def forward_test(
        self,
        feat: Tensor,
        holistic_feat_encoder: Tensor,
        img_metas: Optional[List[Any]] = None,
    ) -> Tensor:
        valid_ratios = None
        if self.mask and img_metas and len(img_metas) > 0:
            # 推理时，img_metas 可能只包含 valid_ratios，或者在列表的第一个位置
            if isinstance(img_metas[0], Tensor):
                valid_ratios = img_metas[0]

        N_bs, device = feat.size(0), feat.device

        # 1. 初始化解码器 RNN 输入序列
        #    包含 [holistic_feat_encoder, embedding(START_TOKEN)]
        start_token_ids = torch.full(
            (N_bs,), self.start_idx, dtype=torch.long, device=device
        )
        start_token_emb = self.embedding(start_token_ids)  # (N, EmbDim)

        holistic_feat_unsqueezed = holistic_feat_encoder.unsqueeze(1).type_as(
            start_token_emb
        )
        # decoder_input_current_sequence: (N, 2, EmbDim)
        decoder_input_current_sequence = torch.cat(
            (holistic_feat_unsqueezed, start_token_emb.unsqueeze(1)), dim=1
        )

        output_probs_list = []

        for t in range(self.max_seq_len):  # 迭代生成 max_seq_len 个字符
            # 2. 通过注意力解码一步
            # logits_all_steps: (N, current_len_for_rnn_input, NumClasses-1)
            logits_all_steps = self._2d_attention(
                decoder_input_current_sequence,
                feat,
                holistic_feat_encoder,
                valid_ratios=valid_ratios,
            )

            # 3. 获取当前时间步的预测
            # 我们只关心对应最新输入 token 的那个输出时间步
            char_logits_current_step = logits_all_steps[:, -1, :]  # (N, NumClasses-1)
            char_probs_current_step = F.softmax(char_logits_current_step, dim=-1)
            output_probs_list.append(char_probs_current_step)

            # 4. 准备下一个时间步的输入 (如果未达到最大长度)
            if t < self.max_seq_len - 1:
                predicted_char_ids = torch.argmax(
                    char_probs_current_step, dim=-1
                )  # (N,)
                next_char_embedding = self.embedding(predicted_char_ids)  # (N, EmbDim)
                decoder_input_current_sequence = torch.cat(
                    (decoder_input_current_sequence, next_char_embedding.unsqueeze(1)),
                    dim=1,
                )
            # SAR 没有显式的 EOS token 来提前停止，总是解码到 max_seq_len

        output_probs_stacked = torch.stack(
            output_probs_list, dim=1
        )  # (N, max_seq_len, NumClasses-1)
        return output_probs_stacked


class SARHead(nn.Module):
    """SAR 预测头部的 PyTorch 实现。"""

    def __init__(
        self,
        in_channels: int,  # 来自主干网络的通道数
        out_channels: int,  # 总类别数 (字符数 + unknown + start + padding)
        enc_dim: int = 512,
        max_text_length: int = 30,
        enc_bi_rnn: bool = False,
        enc_drop_rnn: float = 0.1,
        enc_gru: bool = False,
        dec_bi_rnn: bool = False,
        dec_drop_rnn: float = 0.0,
        dec_gru: bool = False,
        d_k: int = 512,  # 注意力内部维度
        pred_dropout: float = 0.1,
        pred_concat: bool = True,
        **kwargs: Any,  # 吸收 MultiHead 传来的其他参数
    ):
        super().__init__()
        self.encoder = SAREncoder_PT(
            enc_bi_rnn=enc_bi_rnn,
            enc_drop_rnn=enc_drop_rnn,
            enc_gru=enc_gru,
            d_model=in_channels,
            d_enc=enc_dim,
        )
        self.decoder = ParallelSARDecoder_PT(
            out_channels=out_channels,
            enc_bi_rnn=enc_bi_rnn,
            dec_bi_rnn=dec_bi_rnn,
            dec_drop_rnn=dec_drop_rnn,
            dec_gru=dec_gru,
            d_model=in_channels,
            d_enc=enc_dim,
            d_k=d_k,
            pred_dropout=pred_dropout,
            max_text_length=max_text_length,
            pred_concat=pred_concat,
        )

    def forward(
        self, feat: Tensor, targets: Optional[List[Any]] = None
    ) -> Dict[str, Tensor]:
        # feat: (N, C, H, W)
        # targets: 列表。
        #   训练时: [label_tokens (N,S_label_for_input), valid_ratios (N,), ...]
        #   推理时: [valid_ratios (N,)] (可选) 或 None

        # Paddle 的 SARHead.forward(feat, targets) 中，targets 同时用于 encoder 和 decoder
        # encoder 用 targets 获取 valid_ratios (targets[-1] 或 targets[1])
        # decoder (训练时) 用 targets 获取 label (targets[0]) 和 valid_ratios (targets[-1] 或 targets[1])

        holistic_feat = self.encoder(feat, img_metas=targets)  # (N, EncDim_actual)

        if self.training:
            if not targets or len(targets) < 1 or targets[0] is None:
                raise ValueError("SARHead 训练时 targets[0] (label_tokens) 是必需的。")
            # label_tokens for decoder input: e.g., [SOS, c1, ..., cN]
            # Paddle: label = targets[0]
            label_tokens_for_decoder = targets[0]

            # final_out 是 logits
            final_out = self.decoder(
                feat,
                holistic_feat,
                label_tokens_for_decoder,
                img_metas=targets,
                train_mode=True,
            )
            return {"predict": final_out}
        else:
            # final_out 是概率
            final_out = self.decoder(
                feat,
                holistic_feat,
                label_tokens=None,
                img_metas=targets,
                train_mode=False,
            )
            return {"predict": final_out}
