import torch
from torch.optim.lr_scheduler import MultiStepLR, LinearLR
import torchvision
from ssd.data import TDT4265Dataset
from tops.config import LazyCall as L
from ssd.data.transforms import ToTensor, Normalize, Resize, GroundTruthBoxesToAnchors
from ssd.modeling import SSD300, SSDMultiboxLoss, backbones, AnchorBoxes
from .utils import get_dataset_dir
from ssd import utils
from ssd.modeling import AnchorBoxes
from tops.config import LazyCall as L
from .utils import get_dataset_dir, get_output_dir


## ------------------------------ MODEL ----------------------------------

train = dict(
    batch_size=32,
    amp=True,  
    log_interval=20,
    seed=0,
    epochs=50,
    _output_dir=get_output_dir(),
    imshape=(128, 1024),
    image_channels=3,
    num_classes = 8 + 1
)

backbone = L(backbones.DogeModel)(
    output_channels=[512, 1024, 512, 512, 256, 256],
    image_channels="${train.image_channels}",
    output_feature_sizes="${anchors.feature_sizes}"
)

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    strides=[[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes=[[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)

loss_objective = L(SSDMultiboxLoss)(anchors="${anchors}")

model = L(SSD300)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes="${train.num_classes}"
)

optimizer = L(torch.optim.SGD)(
    lr=5e-3, momentum=0.9, weight_decay=0.0005
)

schedulers = dict(
    linear=L(LinearLR)(start_factor=0.1, end_factor=1, total_iters=500),
    multistep=L(MultiStepLR)(milestones=[], gamma=0.1)
)

## ------------------------------ DATA ----------------------------------


train_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
    L(GroundTruthBoxesToAnchors)(anchors="${anchors}", iou_threshold=0.5),
])

val_cpu_transform = L(torchvision.transforms.Compose)(transforms=[
    L(ToTensor)(),
    L(Resize)(imshape="${train.imshape}"),
])

data_train = dict(
    dataset = L(TDT4265Dataset)(
        img_folder=get_dataset_dir("tdt4265_2022"),
        transform="${train_cpu_transform}",
        annotation_file=get_dataset_dir("tdt4265_2022/train_annotations.json")

        # When running on cluster
        #annotation_file=get_dataset_dir("/work/datasets/tdt4265_2022/train_annotations.json")

    ),
    dataloader=L(torch.utils.data.DataLoader)(
        dataset="${..dataset}", 
        num_workers=4, 
        pin_memory=True, 
        shuffle=True, 
        batch_size="${...train.batch_size}", 
        collate_fn=utils.batch_collate,
        drop_last=True
    ),
    gpu_transform=L(torchvision.transforms.Compose)(transforms=[
        L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
    ])
)
data_val = dict(
    dataset = L(TDT4265Dataset)(
        img_folder=get_dataset_dir("tdt4265_2022"),
        transform="${val_cpu_transform}",
        annotation_file=get_dataset_dir("tdt4265_2022/val_annotations.json")

        # When running on cluster
        #annotation_file=get_dataset_dir("/work/datasets/tdt4265_2022/val_annotations.json")
    ),
    dataloader=L(torch.utils.data.DataLoader)(
        dataset="${..dataset}", 
        num_workers=4, 
        pin_memory=True, 
        shuffle=False, 
        batch_size="${...train.batch_size}", 
        collate_fn=utils.batch_collate_val
    ),
    gpu_transform=L(torchvision.transforms.Compose)(transforms=[
        L(Normalize)(mean=[0.4765, 0.4774, 0.2259], std=[0.2951, 0.2864, 0.2878])
    ])
)

label_map = {idx: cls_name for idx, cls_name in enumerate(TDT4265Dataset.class_names)}
