import torchvision
from .task_2_1 import (
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
from tops.config import LazyCall as L
from ssd.data.transforms import ToTensor, Resize, GroundTruthBoxesToAnchors, RandomHorizontalFlip, RandomSampleCrop 

train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(RandomSampleCrop)(),
    L(ToTensor)(),
    L(RandomHorizontalFlip)(),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])