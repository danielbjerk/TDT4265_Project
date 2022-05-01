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
from ssd.modeling import RetinaNet, AnchorBoxes, FocalLoss

# Mennesker er mer avlange en firkanta
anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[24, 24], [40, 40], [48, 48], [128, 128], [172, 172], [256, 256], [256, 800]],
    aspect_ratios=[[1, 0.25], [1, 0.8, 0.25], [0.8, 0.25], [0.7, 0.25, 2], [3, 2, 4], [2, 3, 1.3, 4]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

# bakgrunn, bil, lastebil, buss, motorsykkel, sykkel, sparkesykkel, person, syklist]
# Disse tallene kommer fra å telle relativ andel observasjoner, også vekte motsatt for klasser man ser mye av
# 0.0, 0.52293979, 0.00672609, 0.02433423, 0.0, 0.05703505, 0.03363045, 0.26849675, 0.08683764 
observed_rarity = [1-0.99, 1-0.52293979, 1-0.00672609, 1-0.433423, 1, 1-0.05703505, 1 - 0.03363045, 1 - 0.26849675, 1 - 0.08683764]

loss_objective = L(FocalLoss)(
    anchors="${anchors}",
    alphas=observed_rarity, #[0.01, *[1 for _ in range(train.num_classes - 1)]],
    gamma=2,
    num_classes=train.num_classes
)
