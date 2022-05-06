from turtle import back
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork 

from typing import OrderedDict, Tuple, List

from ssd.modeling.backbones.mobilenet import MobileNet
from .resnet import ResNet


class FPN(nn.Module):
    def __init__(self,
                out_channels: List[int],
                out_channels_backbone: List[int],
                out_channels_fpn: int,
                resnet_type: str):

        super().__init__()

        self.out_channels = out_channels
        self.resnet = ResNet(out_channels_backbone, resnet_type)

        self.pyramid = FeaturePyramidNetwork(
            in_channels_list=out_channels_backbone, 
            out_channels=out_channels_fpn
        )

    def forward(self, x):
        
        (x0, x1, x2, x3, x4, x5) = self.resnet.forward(x)

        fpn_input_features = {}

        fpn_input_features["0"] = x0
        fpn_input_features["1"] = x1
        fpn_input_features["2"] = x2
        fpn_input_features["3"] = x3
        fpn_input_features["4"] = x4
        fpn_input_features["5"] = x5
        
        fpn_output = self.pyramid.forward(fpn_input_features)
        
        out_features = OrderedDict([
            ("0", fpn_output["0"]), 
            ("1", fpn_output["1"]), 
            ("2", fpn_output["2"]), 
            ("3", fpn_output["3"]), 
            ("4", fpn_output["4"]),
            ("pool", fpn_output["5"])
        ])

        return out_features
