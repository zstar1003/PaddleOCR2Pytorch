from __future__ import absolute_import, division, print_function

import math
import torch
from torch import nn, Tensor  # 确保 Tensor 被导入
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple  # 增加了 Tuple

# 从 ppocr.modeling.backbones.rec_svtrnet import Mlp, zeros_
# 我们需要 Mlp 的 PyTorch 版本，这里直接定义
# zeros_ 和 xavier_normal_ 也是辅助函数


# --- 辅助初始化函数 和 Mlp ---
def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    """截断正态初始化，用于权重。"""

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


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Xavier Normal 初始化。"""
    with torch.no_grad():
        return nn.init.xavier_normal_(tensor, gain=gain)


class Mlp(nn.Module):
    """多层感知机 (MLP)，用于 TransformerBlock。
    与 ppocr.modeling.backbones.rec_svtrnet.Mlp 对应。
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # Paddle TransformerBlock 中的 Mlp 使用 nn.ReLU
        self.act = nn.ReLU() if act_layer == nn.ReLU else act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)  # Paddle Mlp 在 act 后有 drop
        x = self.fc2(x)
        x = self.drop(x)  # Paddle Mlp 在 fc2 后有 drop
        return x


# --- NRTR Head 的 PyTorch 实现 ---


class Embeddings(nn.Module):  # 重命名以匹配 Paddle 中的类名
    """NRTR 词嵌入层。对应 Paddle 的 Embeddings 类。"""

    def __init__(
        self,
        d_model: int,
        vocab: int,
        padding_idx: Optional[int] = None,
        scale_embedding: bool = True,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        # Paddle 初始化: w0 = np.random.normal(0.0, d_model**-0.5, (vocab, d_model))
        nn.init.normal_(self.embedding.weight, mean=0.0, std=d_model**-0.5)
        if padding_idx is not None:  # 确保 padding_idx 的权重为 0
            with torch.no_grad():
                self.embedding.weight[padding_idx].fill_(0)
        self.d_model = d_model
        self.scale_embedding = scale_embedding
        self.scale_factor = math.sqrt(d_model) if scale_embedding else 1.0

    def forward(self, x: Tensor) -> Tensor:
        emb = self.embedding(x)
        return emb * self.scale_factor


class PositionalEncoding(nn.Module):  # 重命名以匹配 Paddle 中的类名
    """NRTR 位置编码层。对应 Paddle 的 PositionalEncoding 类。
    Paddle 的实现期望输入是 (S, N, E)，输出也是 (S, N, E)。
    PyTorch Transformer 层如果 batch_first=False，也是 (S, N, E)。
    如果 batch_first=True，则是 (N, S, E)。这里我们按照 Paddle 的原始 S,N,E 格式。
    """

    def __init__(self, dropout: float, dim: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)  # (S, E)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (S, 1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )  # (E/2)

        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, : dim // 2]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        # Paddle: pe = paddle.unsqueeze(pe, 0) -> (1, max_len, dim)
        #         pe = paddle.transpose(pe, [1, 0, 2]) -> (max_len, 1, dim)
        pe = pe.unsqueeze(1)  # (max_len, 1, dim) 以便广播到 (max_len, N, dim)
        self.register_buffer("pe", pe)  # (max_len, 1, dim)

    def forward(self, x: Tensor) -> Tensor:
        # x 形状: (S, N, E) - 对应 Paddle 的输入格式
        # self.pe 形状: (max_len, 1, E)
        x = x + self.pe[: x.size(0), :]  # x.size(0) is S (sequence length)
        return self.dropout(x)


class MultiheadAttention(nn.Module):  # 重命名以匹配 Paddle 中的类名
    """NRTR 多头注意力机制。对应 Paddle 的 MultiheadAttention 类。
    输入 query, key, value 期望形状 (N, S, E) (batch_first=True 风格)。
    但 Paddle 的 TransformerBlock.forward 传递给它的是 (S, N, E) 转置后的。
    这里我们假设输入是 (S, N, E)，并进行相应调整。
    或者，我们让输入是 (N, S, E)，并在内部处理转置，或者让调用者处理。
    Paddle 的 `MultiheadAttention` forward 接收的 query shape 是 `(N, qN, C)`
    经过 reshape 和 transpose 后变成 `(N, num_heads, qN, head_dim)`
    这里我们直接采用 `batch_first=True` 的 PyTorch 风格 `(N, S, E)` 作为输入。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        self_attn: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim 必须能被 num_heads 整除"
        )
        self.scale = self.head_dim**-0.5
        self.self_attn = self_attn

        # Paddle 的 Linear 层在其定义中未指定 bias_attr 时，是有偏置的。
        if self_attn:
            self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)  # Paddle: self.qkv
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.kv_proj = nn.Linear(embed_dim, embed_dim * 2)  # Paddle: self.kv

        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)  # Paddle: self.out_proj

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        # 假设输入是 (N, S, E) (batch_first=True)
        N, S_q, E = query.shape

        if self.self_attn:
            if key is not None or value is not None:
                raise ValueError("自注意力模式下，key 和 value 应该为 None。")
            # (N, S_q, 3*E) -> (N, S_q, 3, num_heads, head_dim) -> (3, N, num_heads, S_q, head_dim)
            qkv = (
                self.qkv_proj(query)
                .reshape(N, S_q, 3, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]  # 各自是 (N, num_heads, S_q, head_dim)
        else:
            if (
                key is None or value is None
            ):  # 在交叉注意力中，key 和 value (通常来自 memory) 是必需的
                # Paddle 的实现中，当 self_attn=False 时，是交叉注意力，key=memory
                # 如果是解码器自注意力，但 with_cross_attn=False 的情况，则 key=query, value=query
                # 这里我们严格按照 Paddle MultiheadAttention 的用法：self_attn=False 时 key 不会是 None
                raise ValueError("交叉注意力模式下，key 和 value 不能为空。")

            S_k = key.shape[1]

            q = (
                self.q_proj(query)
                .reshape(N, S_q, self.num_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )  # (N, num_h, S_q, head_d)

            # Paddle: self.kv = nn.Linear(embed_dim, embed_dim * 2)
            # kv = self.kv(key).reshape((0, kN, 2, num_heads, head_dim)).transpose((2, 0, 3, 1, 4))
            # k, v = kv[0], kv[1]
            # 这意味着 key 和 value 是从同一个输入 `key`（即 memory）经过同一个 `kv_proj` 得到的
            kv_projected_from_key = self.kv_proj(key)  # (N, S_k, 2*E)
            kv_split = kv_projected_from_key.reshape(
                N, S_k, 2, self.num_heads, self.head_dim
            ).permute(2, 0, 3, 1, 4)
            k, v = kv_split[0], kv_split[1]  # (N, num_h, S_k, head_d)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            # attn_mask (S_q, S_k) for causal or (N, S_k) for padding, needs broadcasting
            # PyTorch nn.MultiheadAttention expects attn_mask where True means "don't attend"
            # Paddle adds -inf. So, if attn_mask is float (-inf for masked), direct add is fine.
            # If attn_mask is (S_q, S_k), it should be broadcastable to (N, num_heads, S_q, S_k)
            # Or if (N, S_k), expand to (N, 1, 1, S_k) for padding mask
            attn_scores = attn_scores + attn_mask

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        context = (attn_probs @ v).transpose(1, 2).reshape(N, S_q, self.embed_dim)
        output = self.out_proj(context)
        return output


class TransformerBlock(nn.Module):  # 重命名以匹配 Paddle 中的类名
    """NRTR Transformer 块的 PyTorch 实现。对应 Paddle 的 TransformerBlock 类。"""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        attention_dropout_rate: float = 0.0,
        residual_dropout_rate: float = 0.1,
        with_self_attn: bool = True,
        with_cross_attn: bool = False,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.with_self_attn = with_self_attn
        if with_self_attn:
            self.self_attn = (
                MultiheadAttention(  # 使用上面定义的 PyTorch 版 MultiheadAttention
                    d_model, nhead, dropout=attention_dropout_rate, self_attn=True
                )
            )
            self.norm1 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout1 = nn.Dropout(residual_dropout_rate)

        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = (
                MultiheadAttention(  # 使用上面定义的 PyTorch 版 MultiheadAttention
                    d_model,
                    nhead,
                    dropout=attention_dropout_rate,
                    self_attn=False,  # self_attn=False for cross-attn
                )
            )
            self.norm2 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout2 = nn.Dropout(residual_dropout_rate)

        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=dim_feedforward,
            act_layer=nn.ReLU,
            drop=residual_dropout_rate,  # Paddle 用 ReLU
        )
        self.norm3 = nn.LayerNorm(d_model, eps=epsilon)
        self.dropout3 = nn.Dropout(residual_dropout_rate)

    def forward(
        self,
        tgt: Tensor,
        memory: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,  # 用于自注意力的掩码
        cross_attn_mask: Optional[
            Tensor
        ] = None,  # 用于交叉注意力的掩码 (通常是 memory_padding_mask)
    ) -> Tensor:
        # 输入 tgt, memory 期望形状 (N, S, E) (batch_first=True)
        # Paddle 的 TransformerBlock.forward 接收的 tgt 和 memory 都是 (N,S,C)
        # 然后在 MultiheadAttention 内部或者之前进行转置
        # 我们的 MultiheadAttention_PT 假设输入是 (N,S,E)

        if self.with_self_attn:
            sa_out = self.self_attn(tgt, attn_mask=self_attn_mask)
            tgt = self.norm1(tgt + self.dropout1(sa_out))

        if self.with_cross_attn:
            if memory is None:
                raise ValueError("交叉注意力需要 memory 输入。")
            # cross_attn(query, key, value, attn_mask)
            # query=tgt, key=memory, value=memory
            ca_out = self.cross_attn(
                tgt, key=memory, value=memory, attn_mask=cross_attn_mask
            )
            tgt = self.norm2(tgt + self.dropout2(ca_out))

        mlp_out = self.mlp(tgt)
        tgt = self.norm3(tgt + self.dropout3(mlp_out))
        return tgt


class Transformer(nn.Module):  # 主 Transformer 类，与 Paddle 文件中的类名一致
    """NRTR Transformer 模型的 PyTorch 实现。"""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        beam_size: int = 0,
        num_decoder_layers: int = 6,
        max_len: int = 25,
        dim_feedforward: int = 1024,
        attention_dropout_rate: float = 0.0,
        residual_dropout_rate: float = 0.1,
        in_channels: int = 0,  # 此参数在 Paddle Transformer 中存在但未使用，因为特征已转换
        out_channels: int = 0,  # 纯字符类别数
        scale_embedding: bool = True,
    ):
        super().__init__()
        # Paddle: self.out_channels = out_channels + 1
        # 这个 self.out_channels 用于 Embedding 的词表大小和 tgt_word_prj 的输出维度
        self.decoder_vocab_size = out_channels + 1

        self.max_len = max_len
        self.d_model = d_model
        self.nhead = nhead
        self.beam_size = beam_size  # 0 表示贪婪解码

        self.embedding = Embeddings(  # 使用上面定义的 PyTorch 版 Embeddings
            d_model=d_model,
            vocab=self.decoder_vocab_size,
            padding_idx=0,
            scale_embedding=scale_embedding,  # 假设 PAD_IDX = 0
        )
        self.positional_encoding = (
            PositionalEncoding(  # 使用上面定义的 PyTorch 版 PositionalEncoding
                dropout=residual_dropout_rate,
                dim=d_model,
                max_len=max_len + 50,  # 增加 buffer for max_len
            )
        )

        if num_encoder_layers > 0:
            self.encoder = nn.ModuleList(  # Paddle: self.encoder
                [
                    TransformerBlock(
                        d_model,
                        nhead,
                        dim_feedforward,
                        attention_dropout_rate,
                        residual_dropout_rate,
                        with_self_attn=True,
                        with_cross_attn=False,
                    )
                    for _ in range(num_encoder_layers)
                ]
            )
        else:
            self.encoder = None  # 表示 src 直接作为 memory

        self.decoder = nn.ModuleList(  # Paddle: self.decoder
            [
                TransformerBlock(
                    d_model,
                    nhead,
                    dim_feedforward,
                    attention_dropout_rate,
                    residual_dropout_rate,
                    with_self_attn=True,
                    with_cross_attn=True,
                )
                for _ in range(num_decoder_layers)
            ]
        )

        self.tgt_word_prj = nn.Linear(
            d_model, self.decoder_vocab_size, bias=False
        )  # Paddle bias_attr=False
        # Paddle 初始化: w0 = np.random.normal(0.0, d_model**-0.5, (d_model, self.out_channels)).astype(np.float32)
        # self.tgt_word_prj.weight.set_value(w0)
        # PyTorch Linear weight is (out_features, in_features)
        # Paddle Linear weight is (in_features, out_features)
        # 如果直接用 paddle 的 numpy weight，需要转置
        nn.init.normal_(self.tgt_word_prj.weight, mean=0.0, std=d_model**-0.5)

        self.apply(self._init_weights_module_level)

    def _init_weights_module_level(self, m: nn.Module):
        """应用于整个模块的权重初始化。"""
        if isinstance(m, nn.Linear):
            # 避免重复初始化已经特定初始化的层（如 tgt_word_prj 和 Embedding 内的）
            # 以及 Mlp 内部的 fc1, fc2 (它们可以用默认 Kaiming 或 Xavier)
            if m is not self.tgt_word_prj and not hasattr(m, "_is_embedding_related"):
                xavier_normal_(m.weight)  # Paddle 使用 XavierNormal
                if m.bias is not None:
                    zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            # Embeddings 类内部已经处理了初始化
            pass
        elif isinstance(m, nn.LayerNorm):
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> Tensor:
        """为序列生成一个上三角的注意力掩码 (causal mask)。
        被掩码的位置为 -inf，未被掩码的为 0.0。
        输出形状: (sz, sz)
        Paddle 的 mask.unsqueeze([0,1]) 会变成 (1,1,sz,sz)
        PyTorch nn.MultiheadAttention attn_mask: (L,S) or (N*num_heads, L,S) where L=tgt_len, S=src_len
        对于 decoder自注意力 L=S=tgt_len, mask (tgt_len, tgt_len)
        对于 decoder交叉注意力 L=tgt_len, S=src_len, mask (tgt_len, src_len)
        我们自定义的 MultiheadAttention_PT 的 attn_mask 会直接加到 scores 上，所以 (L,S) 的形状是合适的。
        """
        mask = torch.triu(
            torch.full((sz, sz), float("-inf"), device=device), diagonal=1
        )
        return mask

    def forward_train(self, src: Tensor, tgt_tokens_input: Tensor) -> Tensor:
        # src: (N, S_src, E_src) - 编码器输出或图像特征 (memory)
        # tgt_tokens_input: (N, S_tgt_in) - 目标序列输入 (例如, [SOS, c1, c2, ..., cN])
        # 假设输入都已经是 batch_first=True (N, S, E)

        tgt_emb = self.embedding(tgt_tokens_input)  # (N, S_tgt_in, d_model)
        tgt_pe = self.positional_encoding(tgt_emb)  # (N, S_tgt_in, d_model)

        tgt_self_attn_mask = self._generate_square_subsequent_mask(
            tgt_tokens_input.size(1), src.device
        )

        memory = src
        if self.encoder is not None:
            # Paddle: src = self.positional_encoding(src)
            # 假设 src 已经是 (N,S,E) 格式
            memory = self.positional_encoding(src)  # 直接对 (N,S,E) 的 src 加位置编码
            for encoder_layer in self.encoder:
                # TransformerBlock.forward(tgt, memory, self_attn_mask, cross_attn_mask)
                # 编码器块只有自注意力
                memory = encoder_layer(
                    memory, self_attn_mask=None
                )  # 编码器自注意力通常不需要 mask，除非有 padding

        decoder_output = tgt_pe
        for decoder_layer in self.decoder:
            decoder_output = decoder_layer(
                decoder_output,
                memory,
                self_attn_mask=tgt_self_attn_mask,
                cross_attn_mask=None,
            )

        logits = self.tgt_word_prj(decoder_output)
        return logits

    def forward_test(self, src: Tensor) -> List[Tensor]:
        """贪婪解码。"""
        N, device = src.size(0), src.device

        memory = src
        if self.encoder is not None:
            memory = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                memory = encoder_layer(memory)

        SOS_TOKEN_IDX = 2  # 与 Paddle 实现一致
        EOS_TOKEN_IDX = 3

        decoded_ids = torch.full((N, 1), SOS_TOKEN_IDX, dtype=torch.long, device=device)

        for _ in range(self.max_len - 1):  # 最多生成 max_len-1 个 token (因为SOS已存在)
            current_seq_len = decoded_ids.size(1)

            tgt_emb = self.embedding(decoded_ids)
            tgt_pe = self.positional_encoding(tgt_emb)

            tgt_self_attn_mask = self._generate_square_subsequent_mask(
                current_seq_len, device
            )

            decoder_output = tgt_pe
            for decoder_layer in self.decoder:
                decoder_output = decoder_layer(
                    decoder_output, memory, self_attn_mask=tgt_self_attn_mask
                )

            last_step_logits = self.tgt_word_prj(decoder_output[:, -1, :])
            next_token_ids = torch.argmax(last_step_logits, dim=-1, keepdim=True)

            # 检查是否所有批次都生成了 EOS
            # 如果是，并且我们想在 EOS 后停止（不包括 EOS 本身），则在此处 break
            # Paddle 的逻辑似乎是：如果下一个预测是 EOS，就停止，最终结果不包含这个 EOS
            # 但它的循环条件是 `range(1, paddle.to_tensor(self.max_len))`，且 EOS 检查在 concat 之前
            # 这里为了简单，我们先 concat，如果需要，可以在后处理移除 EOS 之后的内容
            all_eos = (next_token_ids == EOS_TOKEN_IDX).all()
            decoded_ids = torch.cat([decoded_ids, next_token_ids], dim=1)
            if all_eos:
                break
            if decoded_ids.size(1) >= self.max_len:  # 达到最大长度 (包含SOS)
                break

        # Paddle forward_test 返回 [dec_seq, dec_prob]
        # PyTorch MultiHead 的期望是字典，例如 {"predict": dec_seq, "predict_probs": dec_prob}
        # 我们只返回序列，概率的精确跟踪对于贪婪解码不是必需的，除非 Beam Search
        dummy_probs = torch.zeros_like(
            decoded_ids, dtype=torch.float, device=device
        )  # 占位符概率
        return [decoded_ids, dummy_probs]

    def forward_beam(self, src: Tensor) -> List[Tensor]:
        """Beam Search (占位符)。"""
        print(
            "警告: Beam search 未在 PyTorch Transformer (NRTR) 中完全实现，将回退到贪婪解码。"
        )
        return self.forward_test(src)

    def forward(
        self, src: Tensor, targets: Optional[List[Tensor]] = None
    ) -> Dict[str, Tensor]:
        # src: (N, S_src, E_src) - 来自 before_gtc 的特征 (batch_first=True)
        # targets: 训练时为 [tgt_tokens_for_input (N, S_tgt_in), tgt_lengths (N,)] (Paddle 风格)
        #          或者简单地 [tgt_tokens_for_input_and_loss (N, S_full)]
        #          推理时为 None

        if self.training:
            if targets is None or len(targets) < 1 or targets[0] is None:
                raise ValueError("训练模式下，targets[0] (目标 token 序列) 不能为空。")

            # Paddle 实现:
            # max_len_dynamic = targets[1].max() # targets[1] 是不含SOS/EOS的长度
            # tgt_padded_to_max_with_sos_eos = targets[0][:, : 2 + max_len_dynamic]
            # decoder_input_tokens = tgt_padded_to_max_with_sos_eos[:, :-1] # [SOS, c1, ..., cN]
            # 我们假设 PyTorch 的 targets[0] 已经是解码器输入形式: [SOS, c1, ..., cN]
            decoder_input_tokens = targets[0]

            logits = self.forward_train(src, decoder_input_tokens)
            # logits 用于与 targets[0] 右移一位并加上 EOS 的序列计算损失
            return {"predict": logits}
        else:  # 推理模式
            if self.beam_size > 0:
                # 需要完整的 Beam 类和解码逻辑
                pred_output = self.forward_beam(src)
            else:
                pred_output = self.forward_test(
                    src
                )  # 返回 [decoded_ids, decoded_probs]

            return {"predict": pred_output[0], "predict_probs": pred_output[1]}
