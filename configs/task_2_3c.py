import torch

from ssd.modeling.focal_loss import FocalLoss
from .task_2_3b import (
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

loss_objective = L(FocalLoss)(
    anchors="${anchors}",
    alphas=[0.01, *[1 for _ in range(model.num_classes - 1)]],
    gamma=2
)