import torch
from .task_2_3a import (
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

from ssd.modeling.backbones import fpn 
from tops.config import LazyCall as L
from ssd.modeling import SSD300

backbone = L(fpn.FPN)(
    out_channels = [16, 16, 16, 16, 16, 16],
    out_channels_backbone = [24, 40, 80, 80, 112, 160],
    backbone_type = "mobilenet",
    out_channels_fpn = 16 
)

model = L(SSD300)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes="${train.num_classes}",
    init_better_last=True
)