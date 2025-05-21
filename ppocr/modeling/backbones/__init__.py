# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is adapted for a PyTorch-based OCR project.
# It assumes that the corresponding PyTorch implementations of these backbones
# exist in the same directory (e.g., rec_svtrnet.py for SVTRNet).

__all__ = ["build_backbone"]


def build_backbone(config, model_type):
    """
    Build a backbone module for PyTorch.

    Args:
        config (dict): Configuration dictionary for the backbone.
                       It must contain a 'name' key specifying the backbone type.
        model_type (str): Type of the model, e.g., 'det', 'rec', 'cls'.

    Returns:
        torch.nn.Module: An instance of the specified backbone.
    """
    support_dict = []

    # Recognition and Classification Backbones
    if model_type == "rec" or model_type == "cls":
        from .rec_mobilenet_v3 import MobileNetV3
        from .rec_resnet_vd import ResNet  # Assuming this is ResNet_vd for rec
        from .rec_resnet_fpn import ResNetFPN
        from .rec_mv1_enhance import MobileNetV1Enhance
        from .rec_nrtr_mtb import MTB  # Note: MTB is often part of NRTR neck/head
        from .rec_resnet_31 import ResNet31

        # from .rec_resnet_32 import ResNet32 # If you have it
        # from .rec_resnet_45 import ResNet45 # If you have it
        # from .rec_resnet_aster import ResNet_ASTER # If you have it
        # from .rec_micronet import MicroNet # If you have it
        # from .rec_efficientb3_pren import EfficientNetb3_PREN # If you have it
        from .rec_svtrnet import SVTRNet
        from .rec_vitstr import ViTSTR

        # from .rec_resnet_rfl import ResNetRFL # If you have it
        from .rec_densenet import DenseNet

        # from .rec_resnetv2 import ResNetV2 # If you have it
        # from .rec_hybridvit import HybridTransformer # If you have it
        # from .rec_donut_swin import DonutSwinModel # If you have it
        # from .rec_shallow_cnn import ShallowCNN # If you have it
        from .rec_lcnetv3 import PPLCNetV3
        from .rec_hgnet import (
            PPHGNet_small,
        )  # Assuming you have this PyTorch implementation
        # from .rec_vit_parseq import ViTParseQ # If you have it
        # from .rec_repvit import RepSVTR # If you have it
        # from .rec_svtrv2 import SVTRv2 # If you have it
        # from .rec_vary_vit import Vary_VIT_B, Vary_VIT_B_Formula # If you have these
        # from .rec_pphgnetv2 import PPHGNetV2_B4, PPHGNetV2_B4_Formula, PPHGNetV2_B6_Formula # If you have these

        support_dict = [
            "MobileNetV1Enhance",
            "MobileNetV3",
            "ResNet",
            "ResNetFPN",
            "MTB",
            "ResNet31",
            # 'ResNet32', 'ResNet45', 'ResNet_ASTER', 'MicroNet', 'EfficientNetb3_PREN',
            "SVTRNet",
            "ViTSTR",
            # 'ResNetRFL',
            "DenseNet",
            # 'ShallowCNN',
            "PPLCNetV3",
            "PPHGNet_small",
            # 'ViTParseQ', 'RepSVTR', 'SVTRv2', 'ResNetV2', 'HybridTransformer', 'DonutSwinModel',
            # 'Vary_VIT_B', 'PPHGNetV2_B4',
            # 'PPHGNetV2_B4_Formula', 'PPHGNetV2_B6_Formula', 'Vary_VIT_B_Formula',
            # Add other recognition backbones your project supports and has PyTorch implementations for.
            # For example, if your YAML uses "YourCustomRecNet":
            # from .rec_your_custom_net import YourCustomRecNet
            # support_dict.append('YourCustomRecNet')
        ]

    # Detection Backbones
    elif model_type == "det":
        from .det_mobilenet_v3 import MobileNetV3
        from .det_resnet import ResNet  # Assuming this is a specific Det ResNet
        from .det_resnet_vd import ResNet_vd
        from .det_resnet_vd_sast import ResNet_SAST

        # from .det_pp_lcnet import PPLCNet # If you have it
        from .rec_lcnetv3 import PPLCNetV3  # Often PPLCNetV3 is also used for det
        from .rec_hgnet import PPHGNet_small  # PPHGNet_small might also be used for det
        # from .det_pp_lcnet_v2 import PPLCNetV2_base # If you have it
        # from .rec_repvit import RepSVTR_det # If you have it (specific det version)
        # from .rec_vary_vit import Vary_VIT_B # If you have it
        # from .rec_pphgnetv2 import PPHGNetV2_B4 # If you have it

        support_dict = [
            "MobileNetV3",
            "ResNet",
            "ResNet_vd",
            "ResNet_SAST",
            # 'PPLCNet',
            "PPLCNetV3",
            "PPHGNet_small",
            # 'PPLCNetV2_base', 'RepSVTR_det', 'Vary_VIT_B', 'PPHGNetV2_B4',
            # Add other detection backbones
        ]

    # End-to-End Backbones (if needed)
    elif model_type == "e2e":
        # from .e2e_resnet_vd_pg import ResNet # Example
        # support_dict = ['ResNet']
        pass  # Add if your project supports e2e models

    # Table Recognition Backbones (if needed)
    elif model_type == "table":
        # from .table_resnet_vd import ResNet # Example
        # from .table_mobilenet_v3 import MobileNetV3 # Example
        # support_dict = ["ResNet", "MobileNetV3"]
        pass  # Add if your project supports table models

    # KIE Backbones (if needed)
    elif model_type == "kie":
        # from .kie_unet_sdmgr import Kie_backbone # Example
        # from .vqa_layoutlm import LayoutLMForSer, LayoutLMv2ForSer # etc.
        # support_dict = ['Kie_backbone', 'LayoutLMForSer', ...]
        pass  # Add if your project supports KIE models

    else:
        raise NotImplementedError(
            f"model_type='{model_type}' is not supported by this build_backbone function."
        )

    if not support_dict:
        raise NotImplementedError(
            f"No backbones defined for model_type='{model_type}'. Please check __init__.py."
        )

    module_name = config.pop("name")  # Get backbone name from config and remove it

    if module_name not in support_dict:
        error_msg = (
            f"Backbone '{module_name}' is not supported for model_type '{model_type}'.\n"
            f"Supported backbones for '{model_type}': {support_dict}"
        )
        raise AssertionError(error_msg)

    # Dynamically get the class from the current module's scope
    # This relies on the import statements above being correct.
    try:
        module_class = eval(module_name)
    except NameError:
        raise NameError(
            f"Backbone class '{module_name}' is not defined or not imported correctly "
            f"in pytorchocr/modeling/backbones/__init__.py for model_type '{model_type}'."
        )

    # Instantiate the backbone with the rest of the config
    # Ensure the PyTorch backbone's __init__ method matches the config parameters
    print(f"Building PyTorch backbone: {module_name} with config: {config}")
    instance = module_class(**config)
    return instance
