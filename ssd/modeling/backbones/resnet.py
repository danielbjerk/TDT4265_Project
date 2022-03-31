import torch
import torchvision


class ResNet(torch.nn.Module):
    def __init__(self, out_channels, resnet_type: str):

        super().__init__()
        self.out_channels = out_channels

        if resnet_type == "resnet34":
            self.backbone = torchvision.models.resnet34(pretrained=True)
        elif resnet_type == "resnet50":
            self.backone = torchvision.models.resnet50(pretrained=True)
        else:
            self.backone = torchvision.models.resnet18(pretrained=True)

    def forward(self, x):
        x0 = self.backbone.conv1(x)
        x0 = self.backbone.bn1(x0)
        x0 = self.backbone.relu(x0)

        x1 = self.backbone.maxpool(x0)

        x2 = self.backbone.layer1(x1)
        x3 = self.backbone.layer2(x2)
        x4 = self.backbone.layer3(x3)
        x5 = self.backbone.layer4(x4)

        return [x0, x1, x2, x3, x4, x5]
