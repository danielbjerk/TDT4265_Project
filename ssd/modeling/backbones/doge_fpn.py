import torch.nn as nn
import torchvision
from typing import Tuple, List
from ssd.utils import load_config




class DogeModelFPN(nn.Module):
    """
    This is a basic backbone for fpn.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels_bottom_up: List[int],
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels_bottom_up = output_channels_bottom_up
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.pyramid = torchvision.ops.FeaturePyramidNetwork(in_channels_list=output_channels_bottom_up, out_channels=image_channels)
        
    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """

        out_features = []

        x0 = self.backbone.conv1(x)
        x0 = self.backbone.bn1(x0)
        x0 = self.backbone.relu(x0)
        x0 = self.backbone.maxpool(x0)

        x1 = self.backbone.layer1(x0)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        out_features = [x0, x1, x2, x3, x4]

        ligma_feet = {}
        ligma_feet["feet0"] = x0
        ligma_feet["feet1"] = x1
        ligma_feet["feet2"] = x2
        ligma_feet["feet3"] = x3
        ligma_feet["feet4"] = x4
        ligma = self.pyramid.forward(ligma_feet)
        
        out_features = [ligma["feet0"], ligma["feet1"], ligma["feet2"], ligma["feet3"], ligma["feet4"]]
        
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"

        out_features.reverse()
        return out_features
