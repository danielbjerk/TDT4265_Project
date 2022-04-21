# With deeper convolutional nets for regression and classification output heads


from .task_2_3c import (
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
from ssd.modeling import RetinaNet

model = L(RetinaNet)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes="${train.num_classes}",
    init_better_last=True
)
