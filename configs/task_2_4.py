# Changing anchor boxes 

from .task_2_3d import (
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
from ssd.modeling import RetinaNet, AnchorBoxes

# anchors = L(AnchorBoxes)(
#     feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
#     strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
#     min_sizes=[[16, 8], [32, 16], [48, 24], [64, 32], [86, 43], [128, 64], [128, 400]],
#     aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
#     image_shape="${train.imshape}",
#     scale_center_variance=0.1,
#     scale_size_variance=0.2
# )