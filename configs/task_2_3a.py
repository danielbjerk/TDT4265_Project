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

from ssd.modeling.backbones import resnet
from tops.config import LazyCall as L

backbone = L(resnet.ResNet)(
    out_channels = [64, 128, 256, 512, 1024, 2048],
    resnet_type = "resnet34"
)