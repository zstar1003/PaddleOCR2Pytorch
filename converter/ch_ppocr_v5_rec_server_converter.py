import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import OrderedDict
import numpy as np
import torch
import yaml

from pytorchocr.base_ocr_v20 import BaseOCRV20


class PPOCRv5RecConverter(BaseOCRV20):
    def __init__(self, config, paddle_model_params_path, **kwargs_from_main):
        # 1. Read Paddle weights.
        para_state_dict, _ = self.read_paddle_weights(paddle_model_params_path)

        # 2. Modify state dict (e.g., remove or rename layers)
        para_state_dict = self.del_invalid_state_dict(para_state_dict)

        # 3. Determine out_channels for Pytorch model construction
        if not para_state_dict:
            raise ValueError(
                "Paddle parameter state dictionary is empty after loading and processing."
            )

        out_channels_for_ctc = -1  # Initialize with an invalid value
        try:
            if not para_state_dict:
                raise ValueError(
                    "Paddle parameter state dictionary is empty AFTER del_invalid_state_dict."
                )

            ctc_head_fc_weight_key = None
            ctc_head_fc_bias_key = None

            # Prioritize specific CTC head weight key from filtered para_state_dict
            target_key_patterns = [
                "head.ctc_head.fc.weight",  # Most specific for typical CTCHead under a 'head' module
                "head.fc.weight",  # If CTC head's fc is directly under 'head'
                "fc.weight",  # If it's a very simple model with a top-level fc
            ]
            # Search from the end of the filtered dict as head params are usually last
            reversed_keys = list(reversed(list(para_state_dict.keys())))

            for pattern in target_key_patterns:
                for k in reversed_keys:
                    if (
                        pattern in k and "ctc" in k.lower()
                    ):  # Ensure 'ctc' is part of the name for safety
                        ctc_head_fc_weight_key = k
                        break
                if ctc_head_fc_weight_key:
                    break

            if ctc_head_fc_weight_key:
                last_value_tensor = para_state_dict[ctc_head_fc_weight_key]
                # Paddle FC W: [in_features, out_features] -> shape[1] is num_classes
                out_channels_for_ctc = last_value_tensor.shape[1]
                print(
                    f"Determined out_channels for CTC from WEIGHT key '{ctc_head_fc_weight_key}' (shape {last_value_tensor.shape}) as shape[1]: {out_channels_for_ctc}"
                )
            else:
                # Fallback to bias key if weight not found
                target_bias_patterns = [
                    "head.ctc_head.fc.bias",
                    "head.fc.bias",
                    "fc.bias",
                ]
                for pattern in target_bias_patterns:
                    for k in reversed_keys:
                        if pattern in k and "ctc" in k.lower():
                            ctc_head_fc_bias_key = k
                            break
                    if ctc_head_fc_bias_key:
                        break

                if ctc_head_fc_bias_key:
                    last_value_tensor = para_state_dict[ctc_head_fc_bias_key]
                    # Bias shape is [out_features] -> shape[0] is num_classes
                    out_channels_for_ctc = last_value_tensor.shape[0]
                    print(
                        f"Determined out_channels for CTC from BIAS key '{ctc_head_fc_bias_key}' (shape {last_value_tensor.shape}) as shape[0]: {out_channels_for_ctc}"
                    )
                else:
                    # Absolute fallback: use the last key in the filtered dict (less reliable)
                    if para_state_dict:  # Ensure dict is not empty
                        last_key_after_filter = list(para_state_dict.keys())[-1]
                        last_value_tensor = para_state_dict[last_key_after_filter]
                        if (
                            len(last_value_tensor.shape) == 2
                        ):  # Assume weight [in, out] or [out, in]
                            # This is a guess; depends on Pytorch vs Paddle convention for the *absolute last* layer
                            # If Paddle's last layer (non-CTC) was FC: shape[1]
                            # If PyTorch's model's last layer (if it determined order) was FC: shape[0]
                            # Given we are reading Paddle, assume Paddle's [in, out] if it's a weight.
                            out_channels_for_ctc = (
                                last_value_tensor.shape[1]
                                if last_key_after_filter.endswith(".weight")
                                else last_value_tensor.shape[0]
                            )

                        elif len(last_value_tensor.shape) == 1:  # Assume bias
                            out_channels_for_ctc = last_value_tensor.shape[0]
                        print(
                            f"Warning: Falling back to determine out_channels for CTC from LAST key '{last_key_after_filter}' (shape {last_value_tensor.shape}), derived as: {out_channels_for_ctc}"
                        )
                    else:
                        raise ValueError(
                            "Cannot determine out_channels, filtered para_state_dict is empty."
                        )

            if out_channels_for_ctc <= 0:
                raise ValueError(
                    f"Failed to determine a valid positive out_channels for CTC. Got: {out_channels_for_ctc}"
                )

        except Exception as e:
            print(f"Critical error determining out_channels_for_CTC: {e}")
            raise

        # Update the config dictionary that will be passed to BaseModel
        if "Head" not in config:
            config["Head"] = {}

        head_name_from_yaml = config.get("Head", {}).get("name", "UnknownHead")
        print(f"DEBUG: Head name from YAML config is: {head_name_from_yaml}")

        if head_name_from_yaml == "MultiHead":
            # For MultiHead, it expects 'out_channels_list'
            # Assuming 'CTCLabelDecode' is the key for the CTC branch within MultiHead.
            # This key needs to match the one used in your PyTorch MultiHead implementation.
            config["Head"]["out_channels_list"] = {
                "CTCLabelDecode": out_channels_for_ctc
            }
            # If MultiHead might use other heads, you'd need to define their out_channels too,
            # or ensure the PyTorch MultiHead can handle missing entries gracefully if only CTC is used.
            if "out_channels" in config["Head"]:
                del config["Head"]["out_channels"]
            if "num_classes" in config["Head"]:
                del config["Head"]["num_classes"]
            print(
                f"Updated config['Head']['out_channels_list'] for MultiHead: {config['Head']['out_channels_list']}"
            )

        elif head_name_from_yaml == "CTCHead":
            config["Head"]["out_channels"] = out_channels_for_ctc

        # --- Start of new block ---
        # 为 BaseModel 提供 num_classes (纯字符数) 作为后备或主要来源
        # BaseModel 期望 num_classes 是不包含特殊字符 (如CTC blank) 的数量

        # 首先检查 config 中是否已经有 character_dict_path (由 main 函数从 YAML 传入)
        # 如果有，BaseModel 会优先尝试使用它。
        # 我们在这里设置 num_classes 主要是为了：
        # 1. 作为 character_dict_path 加载失败时的后备。
        # 2. 如果 YAML 中根本没有 character_dict_path，则 num_classes 成为主要来源。

        num_pure_characters = -1
        # 确保 out_channels_for_ctc 已成功确定
        if out_channels_for_ctc > 0:
            # out_channels_for_ctc 是从模型权重推断的 Head 输出维度，
            # 通常包含特殊字符 (例如，CTC blank)。
            # 对于 CTC-based heads (CTCHead, or CTCLabelDecode in MultiHead),
            # 纯字符数 = 输出维度 - 1 (blank token).
            # head_name_from_yaml 变量是在此代码块之前定义的
            # if head_name_from_yaml == "MultiHead" or head_name_from_yaml == "CTCHead":
            # 强制设置num_classes为18384（字典大小+1）
            num_pure_characters = 18384
            print(
                f"INFO: Setting/Updating top-level 'num_classes' in config to: {num_pure_characters} (based on dictionary size)."
            )
            config["num_classes"] = num_pure_characters
        elif "character_dict_path" not in config and "num_classes" not in config:
            # 如果无法从权重推断，并且YAML也没提供，则打印一个提示。
            print(
                f"WARNING: Could not derive 'num_classes' from weights, and 'character_dict_path' / 'num_classes' not found in top-level config. BaseModel may issue a warning if it cannot determine class count."
            )
        # --- End of new block ---

        # 4. Initialize the Pytorch model structure
        # 强制设置decoder层数为4
        if "Head" in config and "num_decoder_layers" in config["Head"]:
            config["Head"]["num_decoder_layers"] = 4
            print("强制设置decoder层数为4以匹配Paddle模型")

        super(PPOCRv5RecConverter, self).__init__(config, **kwargs_from_main)

        # 修正调试信息（正确缩进）
        if hasattr(self.net, "head"):
            if hasattr(self.net.head, "nrtr_head_transformer"):
                decoder = self.net.head.nrtr_head_transformer.decoder
                if isinstance(decoder, torch.nn.ModuleList):
                    print(f"实际decoder层数: {len(decoder)}")
                else:
                    print("无法确定decoder层数")

        # 5. Load the (processed) Paddle weights into the Pytorch model
        self.load_paddle_weights([para_state_dict, None])
        print(
            f"PyTorch model constructed and Paddle weights loaded from: {paddle_model_params_path}"
        )
        self.net.eval()

    def del_invalid_state_dict(self, para_state_dict):
        new_state_dict = OrderedDict()
        print("Filtering Paddle state_dict keys (GTC/NRTR/SAR weights will be KEPT)...")
        kept_count = 0

        # 保留所有权重，不再过滤GTC相关权重
        for k, v_tensor in para_state_dict.items():
            new_state_dict[k] = v_tensor
            kept_count += 1

        print(
            f"  Kept {kept_count} keys from original Paddle state_dict to attempt loading (GTC weights are now included)."
        )
        if not new_state_dict:
            print(
                "Warning: del_invalid_state_dict resulted in an empty state dictionary."
            )
        return new_state_dict

    def load_paddle_weights(self, paddle_weights_tuple):
        para_state_dict, _ = paddle_weights_tuple
        if not hasattr(self, "net") or self.net is None:
            raise RuntimeError(
                "PyTorch model (self.net) is not initialized before loading weights."
            )

        pytorch_state_dict = self.net.state_dict()
        loaded_keys_pytorch = set()  # 存储成功加载到 PyTorch 模型中的 PyTorch 键名
        unmatched_paddle_keys_info = []  # 存储未成功加载的 Paddle 键及其原因
        print("Loading Paddle weights into PyTorch model...")

        # 1. 尝试将 Paddle 权重加载到 PyTorch 模型
        for paddle_key, paddle_tensor_val in para_state_dict.items():
            pytorch_key = paddle_key  # 初始假设 PyTorch 键名与 Paddle 相同

            # --- 必要的键名替换规则 ---
            pytorch_key = pytorch_key.replace("._mean", ".running_mean")
            pytorch_key = pytorch_key.replace("._variance", ".running_var")

            # --- 特定于您模型的、更复杂的重命名规则可能需要在这里添加 ---
            # 例如，如果 Paddle 的 LayerNorm 是 'layer.norm.weight' 而 PyTorch 是 'layer.norm_layer.weight'
            # pytorch_key = pytorch_key.replace("some_paddle_pattern", "corresponding_pytorch_pattern")

            # --- NRTR/GTC Head Mappings ---
            # 规则1: 处理backbone的last_conv和fc层
            if pytorch_key == "backbone.last_conv.weight":
                # 跳过这些参数，因为PyTorch模型中可能没有对应的参数
                continue
            elif pytorch_key == "backbone.fc.weight":
                # 跳过这些参数，因为PyTorch模型中可能没有对应的参数
                continue
            elif pytorch_key == "backbone.fc.bias":
                # 跳过这些参数，因为PyTorch模型中可能没有对应的参数
                continue

            # 规则2: Map 'head.before_gtc.X' to 'head.nrtr_before_gtc.X'
            if pytorch_key.startswith("head.before_gtc."):
                pytorch_key = pytorch_key.replace(
                    "head.before_gtc.", "head.nrtr_before_gtc.", 1
                )

            # 规则3: 处理embedding、positional encoding和词投影层
            if pytorch_key == "head.gtc_head.embedding.embedding.weight":
                pytorch_key = "head.nrtr_head_transformer.embedding.embedding.weight"
            elif pytorch_key == "head.gtc_head.positional_encoding.pe":
                # 跳过positional encoding参数，因为形状不匹配
                continue
            elif pytorch_key == "head.gtc_head.tgt_word_prj.weight":
                # 跳过这个参数，因为维度不匹配（18389 vs 18385）
                # 后续可以通过其他方式初始化这个层
                continue

            # 规则4: Map 'head.gtc_head.decoder.layers.X' (Paddle) to 'head.nrtr_head_transformer.decoder.X' (PyTorch)
            if pytorch_key.startswith("head.gtc_head.decoder."):
                # 处理decoder层的参数
                if "layers" in pytorch_key:
                    # Base replacement for the decoder layer path
                    new_pytorch_key = pytorch_key.replace(
                        "head.gtc_head.decoder.layers.",
                        "head.nrtr_head_transformer.decoder.",
                        1,
                    )
                else:
                    # 处理非layers部分的decoder参数
                    new_pytorch_key = pytorch_key.replace(
                        "head.gtc_head.decoder.",
                        "head.nrtr_head_transformer.decoder.",
                        1,
                    )

                # MLP/FFN layers: 处理各种可能的命名方式
                # 不需要替换，因为PyTorch模型中已经使用了mlp.fc1和mlp.fc2
                pass

                # Attention layers: More specific mappings based on PyTorch error log
                # PyTorch uses:
                #   self_attn.qkv_proj, self_attn.out_proj
                #   cross_attn.q_proj, cross_attn.kv_proj, cross_attn.out_proj
                # Common Paddle Naming (can vary based on specific GTC/NRTR implementation):
                #   self_attn.qkv_fc (if fused) or q_fc, k_fc, v_fc (if separate)
                #   self_attn.out_linear or self_attn.fc
                #   cross_attn.q_fc, cross_attn.k_fc, cross_attn.v_fc (or kv_fc if k and v are fused in Paddle)
                #   cross_attn.out_linear or cross_attn.fc

                # 直接映射decoder层的参数
                # Self-Attention Mappings
                if ".self_attn.qkv." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(
                        ".self_attn.qkv.", ".self_attn.qkv_proj."
                    )
                elif ".self_attn.qkv_linear." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(
                        ".self_attn.qkv_linear.", ".self_attn.qkv_proj."
                    )
                elif ".self_attn.qkv_fc." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(
                        ".self_attn.qkv_fc.", ".self_attn.qkv_proj."
                    )

                if ".self_attn.out_proj." in new_pytorch_key:
                    # 已经是正确的格式，不需要替换
                    pass
                elif ".self_attn.out_linear." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(
                        ".self_attn.out_linear.", ".self_attn.out_proj."
                    )
                elif (
                    ".self_attn.fc." in new_pytorch_key
                    and not ".mlp.fc" in new_pytorch_key
                ):
                    new_pytorch_key = new_pytorch_key.replace(
                        ".self_attn.fc.", ".self_attn.out_proj."
                    )

                # Cross-Attention Mappings
                if ".cross_attn.q." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(
                        ".cross_attn.q.", ".cross_attn.q_proj."
                    )
                elif ".cross_attn.q_linear." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(
                        ".cross_attn.q_linear.", ".cross_attn.q_proj."
                    )
                elif ".cross_attn.q_fc." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(
                        ".cross_attn.q_fc.", ".cross_attn.q_proj."
                    )

                # 处理kv参数
                if ".cross_attn.kv." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(
                        ".cross_attn.kv.", ".cross_attn.kv_proj."
                    )
                    # 如果是权重或偏置，需要特殊处理
                    if ".weight" in new_pytorch_key or ".bias" in new_pytorch_key:
                        pytorch_state_dict[new_pytorch_key] = paddle_tensor_val
                        loaded_keys_pytorch.add(new_pytorch_key)
                        continue
                elif ".cross_attn.kv_linear." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(
                        ".cross_attn.kv_linear.", ".cross_attn.kv_proj."
                    )
                elif ".cross_attn.kv_fc." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(
                        ".cross_attn.kv_fc.", ".cross_attn.kv_proj."
                    )

                # 处理单独的k和v参数
                if (
                    ".cross_attn.k." in new_pytorch_key
                    or ".cross_attn.v." in new_pytorch_key
                ):
                    # 跳过单独的k和v参数，因为我们使用合并的kv_proj
                    continue

                if ".cross_attn.out_proj." in new_pytorch_key:
                    # 已经是正确的格式，不需要替换
                    pass
                elif ".cross_attn.out_linear." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(
                        ".cross_attn.out_linear.", ".cross_attn.out_proj."
                    )
                elif (
                    ".cross_attn.fc." in new_pytorch_key
                    and not ".mlp.fc" in new_pytorch_key
                ):  # Avoid mlp.fc, map general cross_attn.fc
                    new_pytorch_key = new_pytorch_key.replace(
                        ".cross_attn.fc.", ".cross_attn.out_proj."
                    )

                # 处理norm层参数
                if ".norm1." in new_pytorch_key:
                    # 已经是正确的格式，不需要替换
                    pass
                elif ".pre_norm." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(".pre_norm.", ".norm1.")

                if ".norm2." in new_pytorch_key:
                    # 已经是正确的格式，不需要替换
                    pass
                elif ".post_norm." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(".post_norm.", ".norm2.")

                if ".norm3." in new_pytorch_key:
                    # 已经是正确的格式，不需要替换
                    pass
                elif ".ffn_norm." in new_pytorch_key:
                    new_pytorch_key = new_pytorch_key.replace(".ffn_norm.", ".norm3.")

                pytorch_key = (
                    new_pytorch_key  # Update the key after all sub-replacements
                )

            # 规则3: SVTR Neck in CTCHead (if applicable, from previous attempts)
            # CTCHead fc layer issues are most likely due to incorrect num_classes in config, not key names here.
            if pytorch_key.startswith("head.ctc_head.head.encoder_svtr."):
                pytorch_key = pytorch_key.replace(
                    "head.ctc_head.head.encoder_svtr.", "head.ctc_head.neck.svtr.", 1
                )
            elif pytorch_key.startswith("head.ctc_head.head.encoder_svtr_fusion."):
                pytorch_key = pytorch_key.replace(
                    "head.ctc_head.head.encoder_svtr_fusion.",
                    "head.ctc_head.neck.svtr_fusion.",
                    1,
                )

            # ------------------------------------------------------------------

            if pytorch_key not in pytorch_state_dict:
                unmatched_paddle_keys_info.append(
                    f"{paddle_key} (Pytorch key '{pytorch_key}' not found in model)"
                )
                continue

            try:
                v_numpy = paddle_tensor_val.numpy()
                source_torch_tensor = torch.from_numpy(v_numpy)
                target_pytorch_tensor = pytorch_state_dict[pytorch_key]

                if not isinstance(target_pytorch_tensor, torch.Tensor):
                    unmatched_paddle_keys_info.append(
                        f"{paddle_key} (PyTorch value for '{pytorch_key}' is not a Tensor: {type(target_pytorch_tensor)})"
                    )
                    continue

                # --- 权重转置和形状调整逻辑 ---
                transpose_this_key = False
                reshape_needed = False
                reshape_dims = None

                # 处理卷积层的形状不匹配
                if "head.ctc_encoder.encoder.conv4.conv.weight" in pytorch_key:
                    # 特殊处理conv4卷积层的形状不匹配
                    if (
                        source_torch_tensor.shape[2] == 1
                        and target_pytorch_tensor.shape[2] == 3
                    ):
                        reshape_needed = True
                        # 将[256, 4096, 1, 3]调整为[256, 4096, 3, 3]
                        reshape_dims = (
                            source_torch_tensor.shape[0],
                            source_torch_tensor.shape[1],
                            target_pytorch_tensor.shape[2],
                            source_torch_tensor.shape[3],
                        )
                # 通用的权重转置逻辑
                elif pytorch_key.endswith(".weight"):  # 普遍检查所有 .weight 后缀的键
                    if (
                        len(source_torch_tensor.shape) == 2
                        and len(target_pytorch_tensor.shape) == 2
                    ):
                        # 如果是2D权重（如全连接层），通常 Paddle [in, out] vs PyTorch [out, in]
                        if (
                            source_torch_tensor.shape[::-1]
                            == target_pytorch_tensor.shape
                        ):
                            transpose_this_key = True

                # 应用转置和形状调整
                if reshape_needed and reshape_dims is not None:
                    # 创建新的形状调整后的张量
                    reshaped_tensor = torch.zeros(
                        reshape_dims, dtype=source_torch_tensor.dtype
                    )
                    # 复制原始数据到新张量
                    if reshape_dims[2] > source_torch_tensor.shape[2]:
                        # 如果目标高度大于源高度，则在每个位置复制源数据
                        for i in range(reshape_dims[2]):
                            reshaped_tensor[:, :, i, :] = source_torch_tensor[
                                :, :, 0, :
                            ]
                    source_torch_tensor_final = reshaped_tensor
                elif transpose_this_key:
                    source_torch_tensor_final = source_torch_tensor.T
                else:
                    source_torch_tensor_final = source_torch_tensor
                # -------------------------------------------------------

                if source_torch_tensor_final.shape != target_pytorch_tensor.shape:
                    unmatched_paddle_keys_info.append(
                        f"{paddle_key} (Shape Mismatch: PyTorch {target_pytorch_tensor.shape} vs Paddle (processed) {source_torch_tensor_final.shape})"
                    )
                    continue

                target_pytorch_tensor.copy_(
                    source_torch_tensor_final.to(target_pytorch_tensor.dtype)
                )
                loaded_keys_pytorch.add(pytorch_key)  # 添加成功加载的 PyTorch 键名

            except Exception as e:
                unmatched_paddle_keys_info.append(
                    f"{paddle_key} (PyTorch key: '{pytorch_key}', Exception: {e})"
                )

        # 2. 打印加载总结和未匹配的 Paddle 键
        if unmatched_paddle_keys_info:
            print(
                f"  重要警告: {len(unmatched_paddle_keys_info)} 个 Paddle 键未能加载到 PyTorch 模型中:"
            )
            for (
                k_unmatched_info
            ) in unmatched_paddle_keys_info:  # 打印所有未加载的 Paddle 键和原因
                print(f"    - {k_unmatched_info}")
        else:
            print("  所有来自 Paddle (过滤后) 的键都已尝试加载。")

        print(
            f"Paddle 权重加载尝试完成。在 PyTorch 模型中，{len(loaded_keys_pytorch)} 个键被成功加载。"
            f"原始 Paddle (过滤后) 权重数量为 {len(para_state_dict)}。"
        )

        # 3. 找出 PyTorch 模型中哪些键没有从 Paddle 权重初始化
        all_pytorch_model_keys = set(pytorch_state_dict.keys())
        uninitialized_pytorch_keys = all_pytorch_model_keys - loaded_keys_pytorch

        if uninitialized_pytorch_keys:
            print(
                f"  重要警告: PyTorch 模型中有 {len(uninitialized_pytorch_keys)} 个键没有从 Paddle 权重中获得初始化:"
            )
            # 过滤掉通常不需要加载的 num_batches_tracked
            uninitialized_pytorch_keys_filtered = [
                k
                for k in sorted(list(uninitialized_pytorch_keys))
                if not k.endswith(".num_batches_tracked")
            ]
            if uninitialized_pytorch_keys_filtered:
                print(
                    f"  (已过滤掉 '.num_batches_tracked' 后，仍有 {len(uninitialized_pytorch_keys_filtered)} 个未初始化键):"
                )
                for k_unloaded in (
                    uninitialized_pytorch_keys_filtered
                ):  # 打印所有未初始化的 PyTorch 键
                    print(
                        f"    - {k_unloaded} (形状: {pytorch_state_dict[k_unloaded].shape})"
                    )
            else:
                print(
                    "  (过滤掉 '.num_batches_tracked' 后，所有 PyTorch 键都已初始化或尝试过加载。)"
                )
        else:
            print(
                "  所有 PyTorch 模型中的键 (除了 num_batches_tracked) 都已从 Paddle 权重中获得或尝试过初始化。"
            )


def read_network_config_from_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML configuration file not found: {yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        res = yaml.safe_load(f)

    if res.get("Architecture") is None:
        raise ValueError(f"{yaml_path} has no 'Architecture' key.")

    global_config = res.get("Global", {})
    if "character_dict_path" in global_config:
        char_dict_path_from_yaml = global_config["character_dict_path"]
        # Try to resolve path
        paths_to_try = [
            char_dict_path_from_yaml,
            os.path.join(
                os.path.dirname(yaml_path), char_dict_path_from_yaml
            ),  # Relative to YAML
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                char_dict_path_from_yaml.replace("./", "", 1),
            ),  # Relative to project root if starts with ./
        ]
        resolved_char_dict_path = None
        for p_try in paths_to_try:
            if os.path.exists(p_try):
                resolved_char_dict_path = os.path.abspath(p_try)
                break

        if resolved_char_dict_path:
            print(f"Using character dictionary: {resolved_char_dict_path}")
            global_config["character_dict_path"] = resolved_char_dict_path
            res["Global"] = global_config
        else:
            print(
                f"Warning: Character dictionary file not found. Checked paths based on: '{char_dict_path_from_yaml}'"
            )

    return res["Architecture"]


if __name__ == "__main__":
    import argparse
    import traceback

    parser = argparse.ArgumentParser(
        description="Convert PaddleOCR PP-OCRv5 Rec Server Model to PyTorch"
    )
    parser.add_argument(
        "--yaml_path",
        type=str,
        required=True,
        help="Path to the YAML network configuration file",
    )
    parser.add_argument(
        "--src_model_path",
        type=str,
        required=True,
        help="Path to the directory containing the PaddleOCR trained model files",
    )
    parser.add_argument(
        "--paddle_params_filename",
        type=str,
        default="PP-OCRv5_server_rec_pretrained.pdparams",
        help="Filename of the PaddlePaddle parameters file",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the converted PyTorch model weights (.pth).",
    )
    args = parser.parse_args()

    cfg = read_network_config_from_yaml(args.yaml_path)
    paddle_pdiparams_file_path = os.path.join(
        os.path.abspath(args.src_model_path), args.paddle_params_filename
    )

    print(f"Using Paddle model parameters file: {paddle_pdiparams_file_path}")
    if not os.path.exists(paddle_pdiparams_file_path):
        raise FileNotFoundError(
            f"PaddlePaddle model parameters file not found: {paddle_pdiparams_file_path}"
        )

    print("Initializing model converter...")
    converter = PPOCRv5RecConverter(cfg, paddle_pdiparams_file_path)
    print("Model conversion process completed.")

    print("Performing a test inference with the converted PyTorch model...")
    np.random.seed(666)

    image_height = 48
    test_image_width = 320
    try:
        with open(args.yaml_path, encoding="utf-8") as f_full_config:
            full_config_data = yaml.safe_load(f_full_config)
            if (
                "Global" in full_config_data
                and "image_shape" in full_config_data["Global"]
            ):
                g_image_shape = full_config_data["Global"]["image_shape"]
                if len(g_image_shape) == 3:
                    image_height = g_image_shape[1]
                    print(
                        f"Using image_height from Global.image_shape in YAML: {image_height}"
                    )
    except Exception as e_yaml_global:
        print(
            f"Note: Could not read Global.image_shape from YAML for test input size. Error: {e_yaml_global}"
        )

    print(f"Using test input shape: (1, 3, {image_height}, {test_image_width})")
    dummy_input_np = np.random.randn(1, 3, image_height, test_image_width).astype(
        np.float32
    )
    dummy_input_torch = torch.from_numpy(dummy_input_np)

    try:
        raw_output = converter.inference(dummy_input_torch)
        main_output = None

        if isinstance(raw_output, torch.Tensor):
            main_output = raw_output
        elif isinstance(raw_output, (list, tuple)):
            if raw_output:
                if isinstance(raw_output[0], torch.Tensor):
                    main_output = raw_output[0]
                    print(
                        f"Model returned a sequence, using the first tensor element (shape: {main_output.shape})."
                    )
                else:
                    # If the first element isn't a tensor, but it's a dict (like from MultiHead with named outputs)
                    if (
                        isinstance(raw_output[0], dict) and "ctc" in raw_output[0]
                    ):  # Common key for CTC output from a dict
                        main_output = raw_output[0]["ctc"]
                        print(
                            f"Model returned a sequence with a dict, using 'ctc' entry (shape: {main_output.shape})."
                        )
                    elif (
                        isinstance(raw_output[0], dict) and "head_out" in raw_output[0]
                    ):  # Another common key
                        main_output = raw_output[0]["head_out"]
                        print(
                            f"Model returned a sequence with a dict, using 'head_out' entry (shape: {main_output.shape})."
                        )
                    else:
                        raise ValueError(
                            f"Model returned a sequence, first element is not a tensor (type: {type(raw_output[0])}) and not a recognized dict structure."
                        )
            else:
                raise ValueError("Model returned an empty list/tuple.")
        else:
            raise TypeError(f"Model returned an unexpected type: {type(raw_output)}")

        if main_output is not None and isinstance(main_output, torch.Tensor):
            main_output_np = main_output.data.cpu().numpy()
            print(f"Test inference (main output) tensor shape: {main_output_np.shape}")
            print(
                f"Test inference (main output) summary: sum={np.sum(main_output_np):.4f}, mean={np.mean(main_output_np):.4f}, "
                f"max={np.max(main_output_np):.4f}, min={np.min(main_output_np):.4f}"
            )
        elif main_output is None:
            print(
                "No main output tensor was identified from model's return value for summary."
            )
        else:
            print(
                f"Main output identified but is not a tensor (type: {type(main_output)}). Cannot summarize."
            )

    except Exception as e:
        print(f"ERROR during test inference with converted model: {e}")
        traceback.print_exc()

    if args.save_path:
        save_name = args.save_path
    else:
        base_yaml_name = os.path.splitext(os.path.basename(args.yaml_path))[0]
        save_name = f"converted_{base_yaml_name}.pth"

    if os.path.dirname(save_name) and not os.path.exists(os.path.dirname(save_name)):
        os.makedirs(os.path.dirname(save_name), exist_ok=True)

    converter.save_pytorch_weights(save_name)
    print("Conversion script finished.")
