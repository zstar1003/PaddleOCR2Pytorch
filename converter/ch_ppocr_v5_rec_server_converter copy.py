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
            # CTCHead in PaddleOCR often takes 'n_class' or 'num_classes'.
            # Ensure your PyTorch CTCHead implementation uses one of these or 'out_channels'.
            config["Head"]["n_class"] = (
                out_channels_for_ctc  # Common in some PaddleOCR heads
            )
            config["Head"]["num_classes"] = out_channels_for_ctc  # Common in PyTorch
            print(
                f"Updated config['Head'] for {head_name_from_yaml} with out_channels/n_class: {out_channels_for_ctc}"
            )

        else:  # For other or unknown head types
            print(
                f"Warning: Unknown or unhandled Head type '{head_name_from_yaml}'. Setting 'out_channels' by default."
            )
            config["Head"]["out_channels"] = out_channels_for_ctc

        # 4. Initialize the Pytorch model structure
        super(PPOCRv5RecConverter, self).__init__(config, **kwargs_from_main)

        # 5. Load the (processed) Paddle weights into the Pytorch model
        self.load_paddle_weights([para_state_dict, None])
        print(
            f"PyTorch model constructed and Paddle weights loaded from: {paddle_model_params_path}"
        )
        self.net.eval()

    def del_invalid_state_dict(self, para_state_dict):
        new_state_dict = OrderedDict()
        print("Filtering Paddle state_dict keys...")
        skipped_count = 0
        for k, v_tensor in para_state_dict.items():
            # More robust skipping for GTC components
            if "gtc_head" in k:  # Skip any key containing 'gtc_head'
                # print(f"  Skipping Paddle GTC layer: {k}")
                skipped_count += 1
                continue
            # Keep the 'before_gtc' skip if it's still relevant for your specific model version
            # This was in original code. If PP-OCRv5 server doesn't have this, it can be removed.
            elif k.startswith("head.before_gtc"):
                # print(f"  Skipping Paddle 'before_gtc' layer: {k}")
                skipped_count += 1
                continue
            else:
                new_state_dict[k] = v_tensor
        if skipped_count > 0:
            print(f"  Skipped {skipped_count} GTC-related or 'before_gtc' keys.")
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
        loaded_keys = set()
        unmatched_paddle_keys = []
        print("Loading Paddle weights into PyTorch model...")

        for paddle_key, paddle_tensor_val in para_state_dict.items():
            pytorch_key = paddle_key
            pytorch_key = pytorch_key.replace("._mean", ".running_mean")
            pytorch_key = pytorch_key.replace("._variance", ".running_var")
            # Add more specific renaming rules if needed for PP-OCRv5 architecture,
            # e.g., for different batchnorm or layernorm naming conventions.
            # Example: some Paddle models use '.norm.weight' for LayerNorm where PyTorch uses '.weight'
            # pytorch_key = pytorch_key.replace(".norm.weight", ".weight") # If LN is just 'norm'
            # pytorch_key = pytorch_key.replace(".norm.bias", ".bias")

            if pytorch_key not in pytorch_state_dict:
                unmatched_paddle_keys.append(paddle_key)
                continue

            try:
                v_numpy = paddle_tensor_val.numpy()
                source_torch_tensor = torch.from_numpy(v_numpy)

                target_pytorch_tensor = pytorch_state_dict[pytorch_key]

                if not isinstance(target_pytorch_tensor, torch.Tensor):
                    print(
                        f"  CRITICAL WARNING: PyTorch state_dict value for key '{pytorch_key}' is not a Tensor! Type: {type(target_pytorch_tensor)}. Skipping load for this key."
                    )
                    continue

                transpose_this_key = False
                if (
                    pytorch_key.endswith(
                        (
                            ".fc.weight",
                            ".fc1.weight",
                            ".fc2.weight",
                            ".qkv.weight",
                            ".proj.weight",
                            ".query.weight",
                            ".key.weight",
                            ".value.weight",
                            ".tgt_word_prj.weight",
                        )
                    )  # Added from your log
                ):
                    if (
                        len(source_torch_tensor.shape) == 2
                        and len(target_pytorch_tensor.shape) == 2
                    ):
                        if (
                            source_torch_tensor.shape[::-1]
                            == target_pytorch_tensor.shape
                        ):
                            transpose_this_key = True

                if transpose_this_key:
                    source_torch_tensor = source_torch_tensor.T

                if source_torch_tensor.shape != target_pytorch_tensor.shape:
                    # print(f"  ERROR: Shape mismatch for key '{pytorch_key}'. " # Can be verbose
                    #       f"Paddle (source after T): {source_torch_tensor.shape}, "
                    #       f"PyTorch model: {target_pytorch_tensor.shape}. CANNOT LOAD.")
                    unmatched_paddle_keys.append(
                        f"{paddle_key} (Shape Mismatch: Pytorch {target_pytorch_tensor.shape} vs Paddle {source_torch_tensor.shape})"
                    )
                    continue

                target_pytorch_tensor.copy_(
                    source_torch_tensor.to(target_pytorch_tensor.dtype)
                )
                loaded_keys.add(pytorch_key)

            except Exception as e:
                print(
                    f"  EXCEPTION during weight loading for Paddle key: '{paddle_key}' (PyTorch key: '{pytorch_key}') - Error: {e}"
                )
                unmatched_paddle_keys.append(f"{paddle_key} (Exception: {e})")

        if unmatched_paddle_keys:
            print(
                f"  Warning: {len(unmatched_paddle_keys)} Paddle keys were not loaded into PyTorch model (either not found, shape mismatch, or exception):"
            )
            # for k_unmatched in unmatched_paddle_keys[:10]: # Print first 10
            #     print(f"    - {k_unmatched}")
            # if len(unmatched_paddle_keys) > 10: print("    ... and more.")

        print(
            f"Paddle weights loaded into PyTorch model. {len(loaded_keys)} keys matched and loaded out of {len(para_state_dict)} Paddle keys (after filter)."
        )

        all_pytorch_keys = set(pytorch_state_dict.keys())
        unloaded_pytorch_keys = all_pytorch_keys - loaded_keys
        if unloaded_pytorch_keys:
            print(
                f"Warning: {len(unloaded_pytorch_keys)} keys in PyTorch model were not initialized from Paddle weights:"
            )
            # for k_unloaded in sorted(list(unloaded_pytorch_keys))[:10]: # Print first 10
            #     if not k_unloaded.endswith("num_batches_tracked"):
            #         print(f"  - {k_unloaded} (Shape: {pytorch_state_dict[k_unloaded].shape})")
            # if len(unloaded_pytorch_keys) > 10 : print("    ... and more.")


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
