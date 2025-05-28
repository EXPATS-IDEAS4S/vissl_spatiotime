# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

###### code adapted from https://github.com/kenshohara/3d-resnets-pytorch/blob/master/models/resnet2p1d.py/

# %%
import logging
from enum import Enum
from typing import List
import sys
import torch
import torch.nn as nn
# import torchvision.models as models
# from torchvision.models.resnet import Bottleneck
from vissl.config import AttrDict
from vissl.data.collators.collator_helper import MultiDimensionalTensor
from vissl.models.model_helpers import (
    Flatten,
    _get_norm,
    get_trunk_forward_outputs,
    get_tunk_forward_interpolated_outputs,
    transform_model_input_data_type,
)
from vissl.models.trunks import register_model_trunk


import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_inplanes():
    return [64, 128, 256, 512]

def conv1x3x3(in_planes, mid_planes, stride=1):
    return nn.Conv3d(in_planes,
                     mid_planes,
                     kernel_size=(1, 3, 3),
                     stride=(1, stride, stride),
                     padding=(0, 1, 1),
                     bias=False)

def conv3x1x1(mid_planes, planes, stride=1):
    return nn.Conv3d(mid_planes,
                     planes,
                     kernel_size=(3, 1, 1),
                     stride=(stride, 1, 1),
                     padding=(1, 0, 0),
                     bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock2plus1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        n_3d_parameters1 = in_planes * planes * 3 * 3 * 3
        n_2p1d_parameters1 = in_planes * 3 * 3 + 3 * planes
        mid_planes1 = n_3d_parameters1 // n_2p1d_parameters1
        self.conv1_s = conv1x3x3(in_planes, mid_planes1, stride)
        self.bn1_s = nn.BatchNorm3d(mid_planes1)
        self.conv1_t = conv3x1x1(mid_planes1, planes, stride)
        self.bn1_t = nn.BatchNorm3d(planes)

        n_3d_parameters2 = planes * planes * 3 * 3 * 3
        n_2p1d_parameters2 = planes * 3 * 3 + 3 * planes
        mid_planes2 = n_3d_parameters2 // n_2p1d_parameters2
        self.conv2_s = conv1x3x3(planes, mid_planes2)
        self.bn2_s = nn.BatchNorm3d(mid_planes2)
        self.conv2_t = conv3x1x1(mid_planes2, planes)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1_s(x)
        out = self.bn1_s(out)
        out = self.relu(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck2plus1D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)

        n_3d_parameters = planes * planes * 3 * 3 * 3
        n_2p1d_parameters = planes * 3 * 3 + 3 * planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv2_s = conv1x3x3(planes, mid_planes, stride)
        self.bn2_s = nn.BatchNorm3d(mid_planes)
        self.conv2_t = conv3x1x1(mid_planes, planes, stride)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    

BLOCK_CONFIG = {
    10: {"layer": (1, 1, 1, 1), "block": BasicBlock2plus1D},
    18: {"layer": (2, 2, 2, 2), "block": BasicBlock2plus1D},
    34: {"layer": (3, 4, 6, 3), "block": BasicBlock2plus1D},
    50: {"layer": (3, 4, 6, 3), "block": Bottleneck2plus1D},
    101: {"layer": (3, 4, 23, 3), "block": Bottleneck2plus1D},
    152: {"layer": (3, 8, 36, 3), "block": Bottleneck2plus1D},
    200: {"layer": (3, 24, 36, 3), "block": Bottleneck2plus1D},
}


class SUPPORTED_DEPTHS(int, Enum):
    RN50 = 50
    RN101 = 101
    RN152 = 152
    RN200 = 200


class INPUT_CHANNEL(int, Enum):
    lab = 1
    bgr = 3
    rgb = 3


class SUPPORTED_L4_STRIDE(int, Enum):
    one = 1
    two = 2

class ResNet2plus1D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_s_size=7,
                 conv1_s_stride=2,
                 conv1_t_size=3,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        n_3d_parameters = 3 * self.in_planes * conv1_t_size * 7 * 7
        n_2p1d_parameters = 3 * 7 * 7 + conv1_t_size * self.in_planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv1_s = nn.Conv3d(n_input_channels,
                                 mid_planes,
                                 kernel_size=(1, conv1_s_size, conv1_s_size),
                                 stride=(1, 2, conv1_s_stride),
                                 padding=(0, conv1_s_size // 2, conv1_s_size // 2),
                                 bias=False)
        self.bn1_s = nn.BatchNorm3d(mid_planes)
        self.conv1_t = nn.Conv3d(mid_planes,
                                 self.in_planes,
                                 kernel_size=(conv1_t_size, 1, 1),
                                 stride=(conv1_t_stride, 1, 1),
                                 padding=(conv1_t_size // 2, 0, 0),
                                 bias=False)
        self.bn1_t = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_s(x)
        x = self.bn1_s(x)
        x = self.relu(x)
        x = self.conv1_t(x)
        x = self.bn1_t(x)
        x = self.relu(x)

        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



@register_model_trunk("r2plus1d")
class R2plus1D(nn.Module):
    """
    Wrapper for above defined ResNet 2plus1D Model to support different depth and
    width_multiplier. We provide flexibility with LAB input, stride in last
    ResNet block and type of norm (BatchNorm, LayerNorm)
    """

    def __init__(self, model_config: AttrDict, model_name: str):
        super(R2plus1D, self).__init__()
        self.model_config = model_config
        logging.info(
            "ResNeXT trunk, supports activation checkpointing. {}".format(
                "Activated"
                if self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
                else "Deactivated"
            )
        )

        self.input_channels = INPUT_CHANNEL[self.model_config.INPUT_TYPE]
        self.trunk_config = self.model_config.TRUNK.R2PLUS1D
        self.depth = SUPPORTED_DEPTHS(self.trunk_config.DEPTH)
        self.width_multiplier = self.trunk_config.WIDTH_MULTIPLIER
        self.use_checkpointing = (
            self.model_config.ACTIVATION_CHECKPOINTING.USE_ACTIVATION_CHECKPOINTING
        )
        self.num_checkpointing_splits = (
            self.model_config.ACTIVATION_CHECKPOINTING.NUM_ACTIVATION_CHECKPOINTING_SPLITS
        )
        self.kernel_t_size = self.trunk_config.KERNEL_T_SIZE
        self.kernel_s_size = self.trunk_config.KERNEL_S_SIZE

        (n1, n2, n3, n4) = BLOCK_CONFIG[self.depth]["layer"]
        block = BLOCK_CONFIG[self.depth]["block"]
        logging.info(
            f"Building model: ResNet 2plus1D"
            f"depth {self.depth}"
            # f"w{self.width_multiplier}-{self._norm_layer.__name__}"
        )
        # get input channels from config

        model = ResNet2plus1D(
            block=block,
            layers=(n1, n2, n3, n4),
            block_inplanes=get_inplanes(),
            n_input_channels=self.input_channels.value,  # Assuming RGB input
            conv1_s_size=self.kernel_s_size,
            conv1_s_stride=2,
            conv1_t_size=self.kernel_t_size,
            conv1_t_stride=2,
            no_max_pool=False,
            shortcut_type='B',  # Assuming 'B' for bottleneck
            widen_factor=self.width_multiplier,
            n_classes=1000  # Assuming 1000 classes for ImageNet
        )

        # we mapped the layers of resnet model into feature blocks to facilitate
        # feature extraction at various layers of the model. The layers for which
        # to extract features is controlled by requested_feat_keys argument in the
        # forward() call.
        self._feature_blocks = nn.ModuleDict(
            [
                ("conv1_s", model.conv1_s),
                ("bn1_s", model.bn1_s),
                ("conv1_t", model.conv1_t),
                ("bn1_t", model.bn1_t),
                ("conv1_relu", model.relu),
                ("maxpool", model.maxpool),
                ("layer1", model.layer1),
                ("layer2", model.layer2),
                ("layer3", model.layer3),
                ("layer4", model.layer4),
                ("avgpool", model.avgpool),
                ("flatten", Flatten(1)),
            ]
        )

        # give a name mapping to the layers so we can use a common terminology
        # across models for feature evaluation purposes.
        self.feat_eval_mapping = {
            "conv1": "conv1_relu",
            "res1": "maxpool",
            "res2": "layer1",
            "res3": "layer2",
            "res4": "layer3",
            "res5": "layer4",
            "res5avg": "avgpool",
            "flatten": "flatten",
        }

    def forward(
        self, x: torch.Tensor, out_feat_keys: List[str] = None
    ) -> List[torch.Tensor]:
        if isinstance(x, MultiDimensionalTensor):
            out = get_tunk_forward_interpolated_outputs(
                input_type=self.model_config.INPUT_TYPE,
                interpolate_out_feat_key_name="res5",
                remove_padding_before_feat_key_name="avgpool",
                feat=x,
                feature_blocks=self._feature_blocks,
                feature_mapping=self.feat_eval_mapping,
                use_checkpointing=self.use_checkpointing,
                checkpointing_splits=self.num_checkpointing_splits,
            )
        else:
            model_input = transform_model_input_data_type(
                x, self.model_config.INPUT_TYPE
            )
            out = get_trunk_forward_outputs(
                feat=model_input,
                out_feat_keys=out_feat_keys,
                feature_blocks=self._feature_blocks,
                feature_mapping=self.feat_eval_mapping,
                use_checkpointing=self.use_checkpointing,
                checkpointing_splits=self.num_checkpointing_splits,
            )
        return out


# # %%
# if __name__ == "__main__":
#     # Example usage
#     model_config = AttrDict({
#         "TRUNK": {
#             "R2PLUS1D": {
#                 "DEPTH": 50,
#                 "WIDTH_MULTIPLIER": 1.0,
#                 "KERNEL_T_SIZE": 3,
#                 "KERNEL_S_SIZE": 7,
#                 "STANDARDIZE_CONVOLUTIONS": False,
#             }
#         },
#         "INPUT_TYPE": "rgb",
#         "ACTIVATION_CHECKPOINTING": {
#             "USE_ACTIVATION_CHECKPOINTING": False,
#             "NUM_ACTIVATION_CHECKPOINTING_SPLITS": 1
#         }
#     })
#     model_name = "r2plus1d50"
#     model = R2plus1D(model_config, model_name)
#     # for feature_block_name, feature_block in model._feature_blocks.items():
#     #     print(f"{feature_block_name}: {feature_block}")
#     #     print("layer shape:", feature_block)

#     # Create a dummy input tensor
#     input_tensor = torch.randn(1, 1, 8, 100, 100)  # Batch size of 1, RGB image
#     # Forward pass
#     # output = model(input_tensor, out_feat_keys=["res5"])
#     # print("Output shape:", output.shape)  # Should be [1, 1000] for ImageNet

# %%
