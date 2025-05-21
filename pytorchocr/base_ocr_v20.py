import os
import sys
from collections import OrderedDict
import numpy as np

# import cv2 # Not used in this file's provided snippet
import torch
import paddle

# 确保你的项目结构中 pytorchocr.modeling.architectures.base_model 路径是正确的
# 如果 BaseModel 在其他地方，需要调整导入
try:
    # 假设你的 PyTorch 模型结构定义在 ppocr.modeling.architectures.base_model
    # 这通常意味着你正在使用 PaddleOCR 原始模型结构定义，并尝试将 Paddle 权重加载进去
    # 如果你的 PyTorch 模型是独立实现的，请修改此导入路径
    from pytorchocr.modeling.architectures.base_model import BaseModel
except ImportError:
    print(
        "Error: Could not import BaseModel from ppocr.modeling.architectures.base_model. "
        "Please check the import path in pytorchocr/base_ocr_v20.py. "
        "This path should point to your PyTorch model's base class definition."
    )
    raise


class BaseOCRV20:
    def __init__(self, config, **kwargs):
        self.config = config
        # PPOCRv5RecConverter 会在调用 super().__init__ 之前确定 out_channels
        # 并将其放入 config['Head']['out_channels']。
        # kwargs 可能会包含其他从 main 传来的参数。
        self.build_net(**kwargs)  # kwargs 主要用于传递除 config 之外的参数
        if hasattr(self, "net") and self.net is not None:  # 确保网络已构建
            self.net.eval()
        else:
            print("Warning: self.net was not initialized in BaseOCRV20.__init__.")

    def build_net(self, **kwargs):
        # config 应该是从 YAML 加载并可能被修改过的架构配置
        # BaseModel 的 __init__ 现在应该从 self.config 中获取所有必要的配置，
        # 包括 self.config['Head']['out_channels']
        # kwargs 可以传递其他 BaseModel 可能需要的参数（如果有的话）
        self.net = BaseModel(self.config, **kwargs)

    def read_pytorch_weights(self, weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError("{} is not existed.".format(weights_path))
        weights = torch.load(weights_path, map_location=torch.device("cpu"))
        return weights

    def get_out_channels(self, weights_dict):  # weights_dict is a state_dict
        if not weights_dict:
            raise ValueError(
                "Weights dictionary is empty, cannot determine out_channels."
            )

        # Try to find a key that clearly belongs to the final classification layer,
        # e.g., often ends with 'fc.weight' or 'classifier.weight' or similar.
        # Using the absolute last key can be risky if the order is not guaranteed
        # or if auxiliary parameters are last.

        # Heuristic: look for keys typically associated with a final linear layer's weights or bias
        potential_last_layer_keys = [
            k
            for k in weights_dict.keys()
            if "fc.weight" in k
            or "classifier.weight" in k
            or "head.weight" in k
            or k.endswith(".bias")
        ]
        if (
            not potential_last_layer_keys
        ):  # Fallback to absolute last key if no heuristic match
            last_key = list(weights_dict.keys())[-1]
        else:
            # Prefer weight keys over bias keys, and among those, the "longest" or most specific one
            # This is still a heuristic.
            last_key = sorted(potential_last_layer_keys, key=len, reverse=True)[0]
            print(
                f"DEBUG get_out_channels: Chose '{last_key}' as potential last layer key."
            )

        last_value = weights_dict[last_key]

        if hasattr(last_value, "numpy"):
            last_value_np = (
                last_value.cpu().numpy()
                if hasattr(last_value, "cpu")
                else last_value.numpy()
            )
        elif isinstance(last_value, np.ndarray):
            last_value_np = last_value
        else:
            raise TypeError(
                f"Unsupported weight type for get_out_channels (key: {last_key}): {type(last_value)}"
            )

        # For an FC layer's weight (PyTorch: [out_features, in_features]), shape[0] is out_channels.
        # For an FC layer's bias (PyTorch: [out_features]), shape[0] is out_channels.
        if len(last_value_np.shape) > 0:
            out_channels = last_value_np.shape[0]
        else:
            raise ValueError(
                f"Cannot determine out_channels from layer '{last_key}' with shape {last_value_np.shape}"
            )
        return out_channels

    def load_state_dict(self, weights):
        self.net.load_state_dict(weights)
        print("PyTorch state_dict is loaded into the model.")

    def load_pytorch_weights(self, weights_path):
        print(f"Loading PyTorch weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=torch.device("cpu"))
        self.net.load_state_dict(state_dict)
        print("PyTorch model weights loaded successfully.")

    def save_pytorch_weights(self, weights_path):
        print(f"Saving PyTorch model weights to: {weights_path}")

        state_dict_to_save = OrderedDict()
        all_tensors_valid = True

        # Create a clean state_dict, ensuring all values are torch.Tensor
        original_state_dict = self.net.state_dict()
        print("DEBUG: Verifying contents of state_dict before saving:")
        for k, v in original_state_dict.items():
            if isinstance(v, torch.Tensor):
                state_dict_to_save[k] = v.cpu()  # Move to CPU before saving
                # print(f"  Key: {k}, Value Type: {type(v)}, Shape: {v.shape}, Is Tensor: True")
            else:
                print(
                    f"  WARNING - Key: {k}, Value Type: {type(v)}. This item is NOT a torch.Tensor and will be SKIPPED for saving."
                )
                all_tensors_valid = False

        if not all_tensors_valid:
            print(
                "DEBUG: Not all values in the original state_dict were torch.Tensors. Non-tensor items were skipped."
            )
        else:
            print(
                "DEBUG: All values in state_dict are (or were converted to) torch.Tensors and moved to CPU."
            )

        if not state_dict_to_save:
            print("ERROR: The state_dict to save is empty. Nothing will be saved.")
            return

        try:
            torch.save(state_dict_to_save, weights_path)
            print(f"PyTorch model weights saved successfully to: {weights_path}")
        except Exception as e:
            print(f"ERROR saving PyTorch weights: {e}")
            print(
                "This likely means some values in the state_dict are still not serializable by torch.save, even after filtering."
            )
            print(
                "Problematic state_dict keys (if any were logged above with WARNING):"
            )
            for k, v in original_state_dict.items():
                if not isinstance(v, torch.Tensor):
                    print(f"  - Key '{k}' had non-Tensor type: {type(v)}")

    def print_pytorch_state_dict(self):
        if not hasattr(self, "net") or self.net is None:
            print("PyTorch model (self.net) is not initialized.")
            return
        print("PyTorch model state_dict:")
        for k, v in self.net.state_dict().items():
            print(f"{k} ---- {v.shape} ---- {v.dtype} ---- {type(v)}")

    def read_paddle_weights(self, weights_path):
        print(
            f"Attempting to load Paddle weights from: {weights_path} using paddle.load()"
        )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Paddle weights file not found: {weights_path}")

        try:
            para_state_dict = paddle.load(weights_path)

            if not isinstance(para_state_dict, dict):
                raise RuntimeError(
                    f"paddle.load('{weights_path}') did not return a dictionary (state_dict). "
                    f"Got type: {type(para_state_dict)}. This converter expects a state dictionary."
                )

            for k, v in para_state_dict.items():
                if not isinstance(v, paddle.Tensor):
                    print(
                        f"Warning: Value for key '{k}' in loaded Paddle state_dict is not a paddle.Tensor (type: {type(v)})."
                    )
            print(
                f"Successfully loaded Paddle weights from {weights_path} using paddle.load()."
            )
            return para_state_dict, None
        except Exception as e:
            print(f"Error using paddle.load() for '{weights_path}': {e}")
            raise

    def print_paddle_state_dict(self, weights_path):
        print(
            f"Attempting to read Paddle state_dict from: {weights_path} for printing (using paddle.load())."
        )
        try:
            para_state_dict, _ = self.read_paddle_weights(weights_path)
            print("Paddle state_dict (loaded via paddle.load()):")
            for k, v in para_state_dict.items():
                print(f"{k} ---- {v.shape} ---- {v.dtype} ---- {type(v)}")
        except Exception as e:
            print(
                f"Error reading Paddle state_dict for printing from {weights_path}: {e}"
            )

    def inference(self, inputs):
        if not hasattr(self, "net") or self.net is None:
            raise RuntimeError(
                "Model (self.net) not initialized. Cannot perform inference."
            )
        with torch.no_grad():
            device = next(self.net.parameters()).device
            inputs = inputs.to(device)

            print(
                f"DEBUG: Input tensor to model: shape={inputs.shape}, dtype={inputs.dtype}, device={inputs.device}"
            )
            infer_out = self.net(inputs)

            # ---- DEBUGGING model output ----
            print(
                f"DEBUG inference: Type of output from self.net(inputs): {type(infer_out)}"
            )
            if isinstance(infer_out, (list, tuple)):
                print(
                    f"DEBUG inference: Output is a sequence of length {len(infer_out)}"
                )
                for i, item in enumerate(infer_out):
                    if isinstance(item, torch.Tensor):
                        print(
                            f"  Item {i}: type={type(item)}, shape={item.shape}, dtype={item.dtype}, device={item.device}"
                        )
                    else:
                        print(
                            f"  Item {i}: type={type(item)}, value={item}"
                        )  # Print non-tensor items directly
            elif isinstance(infer_out, torch.Tensor):
                print(
                    f"DEBUG inference: Output is a Tensor with shape {infer_out.shape}, dtype={infer_out.dtype}, device={infer_out.device}"
                )
            else:
                print(f"DEBUG inference: Output is something else: {infer_out}")
            # ---- END DEBUGGING ----
        return infer_out
