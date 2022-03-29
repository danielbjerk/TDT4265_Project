# Inherit configs from the default ssd300
import torchvision
from ssd.data import TDT4265Dataset
from tops.config import LazyCall as L
from ssd.data.transforms import (
    ToTensor, Normalize, Resize,
    GroundTruthBoxesToAnchors, RandomSampleCrop, RandomHorizontalFlip)

from .doge_fpn import train, anchors, optimizer, schedulers, backbone, model, data_train, data_val, loss_objective
from .utils import get_dataset_dir

# Keep the model, except change the backbone and number of classes
label_map = {idx: cls_name for idx, cls_name in enumerate(TDT4265Dataset.class_names)}
