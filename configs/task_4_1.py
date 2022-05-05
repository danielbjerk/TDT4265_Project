raise Exception("Hakke valgt om vi skal bruke task_2_4 eller 5")
from .task_2_4 import (
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

from .utils import get_dataset_dir

from tops.config import LazyCall as L
from ssd.modeling.backbones import bifpn, AnchorBoxes

train.batch_size = 32
# data_train.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
# data_train.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/train_annotations.json")
# data_val.dataset.img_folder = get_dataset_dir("tdt4265_2022_updated")
# data_val.dataset.annotation_file = get_dataset_dir("tdt4265_2022_updated/val_annotations.json")

backbone = L(bifpn.BIFPN)(
    size = [64, 128, 256, 512, 1024, 2048],
    feature_size = 128,
)

model.feature_extractor = backbone

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16]],
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64]],
    min_sizes=[[24, 24], [40, 40], [48, 48], [128, 128], [172, 172], [256, 256], [256, 800]],
    aspect_ratios=[[1, 0.25], [1, 0.8, 0.25], [0.8, 0.25], [0.7, 0.25, 2], [3, 2, 4]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)