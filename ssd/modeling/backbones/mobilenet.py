import torch
import torch.nn as nn
import torchvision

from torchvision.models.resnet import BasicBlock

class MobileNet(torch.nn.Module):
    def __init__(self, out_channels):

        super().__init__()
        self.out_channels = out_channels
        self.backbone = torchvision.models.mobilenet_v3_large(pretrained=True)

    def forward(self, x):

        x0 = self.backbone.features[0](x)
        x0 = self.backbone.features[1](x0)
        x0 = self.backbone.features[2](x0)

        x1 = self.backbone.features[3](x0)
        x1 = self.backbone.features[4](x1)
        x1 = self.backbone.features[5](x1)

        x2 = self.backbone.features[6](x1)
        x2 = self.backbone.features[7](x2)

        x3 = self.backbone.features[8](x2)
        x3 = self.backbone.features[9](x3)
        x3 = self.backbone.features[10](x3)

        x4 = self.backbone.features[11](x3)
        x4 = self.backbone.features[12](x4)

        x5 = self.backbone.features[13](x4)
        x5 = self.backbone.features[14](x5)
        x5 = self.backbone.features[15](x5)

        return [x0, x1, x2, x3, x4, x5]
        
