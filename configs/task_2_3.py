import torch
from .task_2_2 import (
    train,
    backbone,
    anchors,
    loss_objective,
    model,
    optimizer,
    schedulers,
    train_cpu_transform,
    val_cpu_transform,
    data_train,
    data_val,
    label_map
)

from ssd.modeling import AnchorBoxes
from ssd.modeling.backbones import resnet
from tops.config import LazyCall as L

# Testing:  
# img = torch.zeros(1, 3, 128, 1024)
# 
# import torchvision
# model = torchvision.models.resnet34(pretrained=True)
# 
# x0 = img.conv1(x)
# x0 = img.bn1(x0)
# x0 = img.relu(x0)
# 
# x1 = img.maxpool(x0)
# 
# x2 = img.layer1(x1)
# x3 = img.layer2(x2)
# x4 = img.layer3(x3)
# x5 = img.layer4(x4)
# 
# print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)
# 
# exit()


anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

model.feature_extractor = L(resnet.ResNet)(
    out_channels = [64, 64, 128, 256, 512],
    resnet_type = "resnet34"
)