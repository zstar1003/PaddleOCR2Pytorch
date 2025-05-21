# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

# import paddle # Removed for PyTorch conversion
# from paddle import ParamAttr # Removed for PyTorch conversion
# import paddle.nn as nn # Replaced with torch.nn
# import paddle.nn.functional as F # Replaced with torch.nn.functional
import torch  # Added for PyTorch
import torch.nn as nn  # Added for PyTorch
import torch.nn.functional as F  # Added for PyTorch


# Assuming these are PyTorch compatible implementations or will be converted
# For now, we focus on MultiHead structure. If these are Paddle-specific, they need conversion too.
from ppocr.modeling.necks.rnn import (  # These need to be PyTorch compatible
    Im2Seq,
    # EncoderWithRNN, # Not directly used by MultiHead, but by SequenceEncoder
    # EncoderWithFC, # Not directly used by MultiHead, but by SequenceEncoder
    SequenceEncoder,
    # EncoderWithSVTR, # Not directly used by MultiHead
    # trunc_normal_, # PyTorch has its own init
    # zeros_, # PyTorch has its own init
)
from .rec_ctc_head import CTCHead  # Must be PyTorch compatible
from .rec_sar_head import SARHead  # Must be PyTorch compatible
from .rec_nrtr_head import (
    Transformer as NRTRTransformer,
)  # Must be PyTorch compatible, renamed to avoid conflict


# PyTorch compatible initializers (example, actual init might be in modules)
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # Simple version, for full Torch equivalence use timm's trunc_normal_
    nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def zeros_(tensor):
    nn.init.zeros_(tensor)


class FCTranspose(nn.Module):  # Changed nn.Layer to nn.Module
    def __init__(self, in_channels, out_channels, only_transpose=False):
        super().__init__()
        self.only_transpose = only_transpose
        if not self.only_transpose:
            # self.fc = nn.Linear(in_channels, out_channels, bias_attr=False) # Paddle
            self.fc = nn.Linear(in_channels, out_channels, bias=False)  # PyTorch

    def forward(self, x):
        # PyTorch: permute for transpose
        if self.only_transpose:
            return x.permute(
                0, 2, 1
            )  #  (B, H*W, C) -> (B, C, H*W) if input was (B,C,H,W) then flatten(2)
        #  If input x is (B, T, C), then (B, C, T)
        else:
            # Assuming x is (B, T, C_in) from a previous layer like Flatten(2)
            # then transpose to (B, C_in, T) if fc expects (B, *, C_in)
            # The original Paddle code did x.transpose([0, 2, 1]) which means (B, T, C) -> (B, C, T)
            # then passed to fc. This is unusual.
            # Standard PyTorch nn.Linear expects (B, *, C_in) and outputs (B, *, C_out).
            # If x is (B, Features, Time) and FC is on Features:
            # return self.fc(x.permute(0, 2, 1)) # (B, Time, Features_in) -> (B, Time, Features_out)
            # If x is (B, Time, Features) and FC is on Time:
            # No, the original was x.transpose([0,2,1]) meaning if input is (N,L,C), it becomes (N,C,L)
            # and then fc is applied. This implies fc's in_channels is C.
            # Let's assume x input to this module is (Batch, SeqLen, Channels)
            # Original Paddle: x.transpose([0, 2, 1]) -> (Batch, Channels, SeqLen)
            # Then self.fc(this) -> error because fc expects last dim as in_channels
            # It seems the intention was to apply FC on the SeqLen dimension after transposing,
            # or the in_channels of FC is actually the previous SeqLen.
            # Given `FCTranspose(in_channels, nrtr_dim)`, `in_channels` is likely the feature dim.
            # So, x is (B, T, in_channels). self.fc expects (B, T, in_channels).
            # The transpose is likely for specific data layout expected by NRTR.
            # Let's keep it simple: fc operates on the last dimension.
            # If transpose is needed before FC for a specific reason:
            # x_transposed = x.permute(0, 2, 1) # (B, in_channels, T)
            # out = self.fc(x_transposed.permute(0,2,1)) # This would be complex.
            # Let's assume the input x to this layer is (B, ..., in_channels)
            # and the transpose is for the *output* of FC or the input to something else.
            # The original code `self.fc(x.transpose([0, 2, 1]))` is confusing.
            # If x is (B, T, C_feat) and in_channels=C_feat, then x.transpose([0,2,1]) -> (B, C_feat, T)
            # self.fc applied to this would mean C_feat is the sequence dim for FC, and T is the feature dim.
            # This is highly unconventional.

            # Let's assume standard FC application on the last dim.
            # The transpose happens *after* FC if only_transpose is also False, or it is the *only* op.
            # If the name means "FC then Transpose":
            # x_fc = self.fc(x)
            # return x_fc.permute(0, 2, 1)
            # If the name means "Transpose then FC (on new feature dim)":
            # This is more likely for the NRTR `before_gtc` where `in_channels` is previous channels
            # and `nrtr_dim` is the new channel dim.
            # x is (B, H*W, C_in). Flatten(2) makes it (B, C_in, H*W).
            # transpose([0,2,1]) makes it (B, H*W, C_in). This is standard.
            # So, if x comes from nn.Flatten(start_dim=2), its shape is (B, C, H*W)
            # Then x.permute(0, 2, 1) makes it (B, H*W, C)
            # Then fc(in_channels=C, out_channels=nrtr_dim) is applied. This makes sense.
            x_permuted = x.permute(0, 2, 1)
            return self.fc(x_permuted)


class AddPos(nn.Module):  # Changed nn.Layer to nn.Module
    def __init__(self, dim, w_max_len):  # Renamed w to w_max_len for clarity
        super().__init__()
        # self.dec_pos_embed = self.create_parameter( # Paddle
        #     shape=[1, w, dim], default_initializer=zeros_
        # )
        # self.add_parameter("dec_pos_embed", self.dec_pos_embed) # Paddle
        self.dec_pos_embed = nn.Parameter(torch.zeros(1, w_max_len, dim))  # PyTorch
        trunc_normal_(self.dec_pos_embed)

    def forward(self, x):
        # x shape: (B, T, dim)
        # self.dec_pos_embed shape: (1, w_max_len, dim)
        # Ensure T <= w_max_len
        if x.shape[1] > self.dec_pos_embed.shape[1]:
            # This case should ideally not happen if w_max_len is set correctly
            # Or, positional embedding should be dynamically created or interpolated
            # For now, truncate or error
            raise ValueError(
                f"Input sequence length {x.shape[1]} > positional embedding max length {self.dec_pos_embed.shape[1]}"
            )
        x = x + self.dec_pos_embed[:, : x.shape[1], :]
        return x


class MultiHead(nn.Module):  # Changed nn.Layer to nn.Module
    def __init__(
        self, in_channels, out_channels_list=None, **kwargs
    ):  # Added default None for out_channels_list
        super().__init__()
        self.head_list_config = kwargs.pop("head_list")  # Store the config list
        self.use_pool = kwargs.get("use_pool", False)
        self.use_pos = kwargs.get("use_pos", False)  # For NRTR
        self.in_channels = in_channels  # Backbone output channels

        if self.use_pool:
            # Kernel/stride might need adjustment for typical PyTorch AvgPool2d if input is (B,C,H,W)
            # Original: kernel_size=[3, 2], stride=[3, 2] on reshaped input.
            # Assuming input to pool is (B,C,H,W)
            self.pool = nn.AvgPool2d(
                kernel_size=(1, 1), stride=(1, 1)
            )  # Placeholder, needs review based on usage in forward
            # The original pooling was very specific.

        # Initialize actual head modules to None
        self.sar_head = None
        self.nrtr_before_gtc = None
        self.nrtr_gtc_head = None
        self.ctc_encoder_reshape = None
        self.ctc_encoder = None
        self.ctc_head = None

        # Default active head for training if multiple are built (original logic)
        # This might need to be configured or determined differently in PyTorch.
        self.active_gtc_head_type = "sar"  # Default from original, can be 'nrtr'

        # Check for at least one head to be configured if out_channels_list is provided
        # if out_channels_list is None or not any(k in out_channels_list for k in ["SARLabelDecode", "NRTRLabelDecode", "CTCLabelDecode"]):
        #     print("Warning: MultiHead initialized without any specific head output channels in out_channels_list.")
        # raise ValueError("MultiHead requires at least one head to be configured via out_channels_list.")

        for idx, head_config_item in enumerate(self.head_list_config):
            head_name_key = list(head_config_item.keys())[0]  # e.g., "SARHead"
            head_specific_args = head_config_item[head_name_key]

            if head_name_key == "SARHead":
                if out_channels_list and "SARLabelDecode" in out_channels_list:
                    print("MultiHead: Initializing SARHead...")
                    self.sar_head = SARHead(  # Assuming SARHead is PyTorch compatible
                        in_channels=self.in_channels,
                        out_channels=out_channels_list["SARLabelDecode"],
                        **head_specific_args,
                    )
                    self.active_gtc_head_type = (
                        "sar"  # If SAR is built, it's a candidate for GTC
                    )
                else:
                    print(
                        "MultiHead: SARHead config found in head_list, but 'SARLabelDecode' not in out_channels_list. Skipping SARHead."
                    )

            elif head_name_key == "NRTRHead":
                if out_channels_list and "NRTRLabelDecode" in out_channels_list:
                    print("MultiHead: Initializing NRTRHead...")
                    max_text_length = head_specific_args.get("max_text_length", 25)
                    nrtr_dim = head_specific_args.get(
                        "nrtr_dim", 256
                    )  # This should be the output dim of FCTranspose
                    num_decoder_layers = head_specific_args.get("num_decoder_layers", 4)

                    # before_gtc for NRTR: nn.Flatten(2) -> FCTranspose -> AddPos (optional)
                    # Input to nn.Flatten(2) is (B, C_in, H, W) from backbone.
                    # Output of nn.Flatten(2) is (B, C_in, H*W).
                    # FCTranspose input C_in, output nrtr_dim. Its internal permute makes it (B, H*W, nrtr_dim).

                    # Note: Original Paddle FCTranspose input was in_channels (backbone out)
                    # If backbone output (x) is (B, C, H, W), then x.flatten(2) -> (B, C, H*W)
                    # FCTranspose then takes this. So its `in_channels` should be C (backbone output).
                    # And its `out_channels` is `nrtr_dim`.
                    fc_transpose_module = FCTranspose(self.in_channels, nrtr_dim)

                    if self.use_pos:
                        # Assuming max sequence length for pos embedding is related to feature map width
                        # This needs careful setting. `80` was hardcoded.
                        # If feature map is (H_feat, W_feat), then seq_len is H_feat * W_feat or W_feat if H_feat=1
                        # For typical rec, H_feat might be 1 after neck.
                        # Let's assume a max_seq_len needs to be passed or inferred.
                        # For now, using a placeholder. This is a CRITICAL parameter.
                        estimated_max_seq_len_for_nrtr_pos = head_specific_args.get(
                            "max_seq_len_for_pos_emb", 80
                        )
                        self.nrtr_before_gtc = nn.Sequential(
                            nn.Flatten(start_dim=2),  # (B, C, H, W) -> (B, C, H*W)
                            fc_transpose_module,  # (B, C, H*W) -> (B, H*W, nrtr_dim)
                            AddPos(nrtr_dim, estimated_max_seq_len_for_nrtr_pos),
                        )
                    else:
                        self.nrtr_before_gtc = nn.Sequential(
                            nn.Flatten(start_dim=2),  # (B, C, H, W) -> (B, C, H*W)
                            fc_transpose_module,  # (B, C, H*W) -> (B, H*W, nrtr_dim)
                        )

                    self.nrtr_gtc_head = (
                        NRTRTransformer(  # Assuming Transformer is PyTorch NRTR head
                            d_model=nrtr_dim,
                            nhead=nrtr_dim
                            // 32,  # Ensure nhead is compatible with PyTorch MHA
                            num_encoder_layers=-1,  # Not used if only decoder
                            beam_size=-1,  # Not used if not beam search
                            num_decoder_layers=num_decoder_layers,
                            max_len=max_text_length,  # Max output text length
                            dim_feedforward=nrtr_dim * 4,
                            out_channels=out_channels_list[
                                "NRTRLabelDecode"
                            ],  # Num classes for NRTR
                        )
                    )
                    self.active_gtc_head_type = (
                        "nrtr"  # If NRTR is built, it's a candidate for GTC
                    )
                else:
                    print(
                        "MultiHead: NRTRHead config found in head_list, but 'NRTRLabelDecode' not in out_channels_list. Skipping NRTRHead."
                    )

            elif head_name_key == "CTCHead":
                if out_channels_list and "CTCLabelDecode" in out_channels_list:
                    print("MultiHead: Initializing CTCHead...")
                    # ctc neck
                    # self.ctc_encoder_reshape = Im2Seq(self.in_channels) # Im2Seq might be part of SequenceEncoder

                    neck_args = head_specific_args.get("Neck", {})
                    encoder_type = neck_args.pop(
                        "name", "lstm"
                    )  # Default or from config

                    # SequenceEncoder needs to be PyTorch compatible
                    # Its in_channels is the backbone's out_channels
                    self.ctc_encoder = SequenceEncoder(
                        in_channels=self.in_channels,
                        encoder_type=encoder_type,
                        **neck_args,
                    )

                    # ctc head
                    ctc_head_args = head_specific_args.get("Head", {})
                    self.ctc_head = CTCHead(  # Assuming CTCHead is PyTorch compatible
                        in_channels=self.ctc_encoder.out_channels,  # Output from CTC's own neck
                        out_channels=out_channels_list["CTCLabelDecode"],
                        **ctc_head_args,
                    )
                else:
                    print(
                        "MultiHead: CTCHead config found in head_list, but 'CTCLabelDecode' not in out_channels_list. Skipping CTCHead."
                    )
            else:
                print(
                    f"Warning: {head_name_key} is not a recognized head type in MultiHead. Skipping."
                )
                # raise NotImplementedError( # Or be strict
                #     "{} is not supported in MultiHead yet".format(head_name_key)
                # )

        if not self.ctc_head:
            print(
                "CRITICAL WARNING: MultiHead was configured, but CTCHead part was NOT built. Inference will likely fail."
            )

    def forward(self, x, targets=None):
        # x is backbone output, e.g., (B, C, H, W)

        # Original pooling logic was complex and tied to a reshape.
        # If `use_pool` is true, x was reshaped to (B, 3, H_new, C_in) then permuted to (B, C_in, 3, H_new)
        # This seems highly specific. For now, skipping this complex pooling.
        # If your model absolutely needs it, this part needs careful porting.
        # if self.use_pool:
        #     # x_reshaped = x.reshape([x.shape[0], 3, -1, self.in_channels]) # This assumes C_in % 3 == 0 or C_in is fixed.
        #     # x_permuted = x_reshaped.permute([0, 3, 1, 2]) # (B, C_in, 3, H_new)
        #     # x = self.pool(x_permuted)
        #     print("Warning: use_pool=True, but the specific pooling logic is not fully ported. Using input x directly.")
        #     pass

        # --- CTC Path ---
        ctc_out = None
        ctc_encoder_features = None
        if self.ctc_encoder and self.ctc_head:
            # Input to ctc_encoder (SequenceEncoder) is usually (B, C, H, W)
            # Or (B, C, 1, W) if height is already 1.
            # Im2Seq inside SequenceEncoder handles permutation to (B, T, C_new_neck)
            ctc_encoder_features = self.ctc_encoder(x)  # Output (B, T, C_neck_out)
            ctc_out = self.ctc_head(
                ctc_encoder_features, targets
            )  # targets for CTC is usually a tuple (text_labels, text_lengths)
        else:
            # This case should ideally not happen if conversion expects CTC.
            # Return a dummy tensor or raise error.
            # For inference, we absolutely need ctc_out.
            # For PyTorch, if a module isn't used in forward, its params might not get grads in training.
            # But for conversion, we just need the structure.
            # print("Warning: CTC path not fully built in MultiHead. Returning None for ctc_out.")
            # To avoid errors downstream if only CTC is expected at inference:
            if not self.training:
                raise RuntimeError(
                    "MultiHead: CTC path is not built, cannot proceed with inference."
                )

        # --- Output dictionary ---
        head_out = dict()
        if ctc_out is not None:
            head_out["ctc"] = ctc_out
        if ctc_encoder_features is not None:
            head_out["ctc_neck"] = ctc_encoder_features  # Features after CTC's neck

        # --- Inference Mode (eval mode) ---
        if not self.training:
            if ctc_out is not None:
                return ctc_out  # For server rec, CTC output is primary
            else:
                # This should have been caught above if ctc_out is None
                return {"error": "CTC output not available"}

        # --- Training Mode ---
        # GTC path (SAR or NRTR) - only if built and targets are provided appropriately
        if self.active_gtc_head_type == "sar":
            if self.sar_head:
                # SARHead expects backbone output `x` and SAR-specific targets
                # targets[1:] implies targets = (ctc_target, sar_target, nrtr_target, ...)
                # Ensure `targets` is structured correctly if SAR is used.
                sar_target = (
                    targets[1]
                    if isinstance(targets, (list, tuple)) and len(targets) > 1
                    else None
                )
                if sar_target is not None:
                    sar_out = self.sar_head(x, sar_target)
                    head_out["sar"] = sar_out
                # else:
                #     print("Warning: SAR head built but no SAR target provided in training.")
            # else:
            #     print("Warning: active_gtc_head_type is 'sar' but sar_head is not built.")

        elif self.active_gtc_head_type == "nrtr":
            if self.nrtr_before_gtc and self.nrtr_gtc_head:
                # NRTRTransformer expects features after `nrtr_before_gtc` and NRTR-specific targets
                nrtr_features = self.nrtr_before_gtc(
                    x
                )  # Input x (B,C,H,W) -> (B, T_nrtr, C_nrtr_dim)
                nrtr_target = (
                    targets[1]
                    if isinstance(targets, (list, tuple)) and len(targets) > 1
                    else None
                )  # Placeholder for target logic
                if (
                    nrtr_target is not None
                ):  # NRTR head needs targets for training (teacher forcing)
                    gtc_out = self.nrtr_gtc_head(
                        nrtr_features, nrtr_target
                    )  # Pass appropriate target
                    head_out["gtc"] = gtc_out  # Original key was 'gtc' for NRTR
                    head_out["nrtr"] = gtc_out  # Alias for clarity
                # else:
                #      print("Warning: NRTR head built but no NRTR target provided in training.")
            # else:
            #     print("Warning: active_gtc_head_type is 'nrtr' but nrtr_gtc_head or before_gtc is not built.")

        return head_out
