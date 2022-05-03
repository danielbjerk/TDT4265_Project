import sys
assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import functools
import time
import click
import torch
import pprint
import tops
import tqdm
from pathlib import Path
from ssd.evaluate import evaluate
from ssd import utils
from tops.config import instantiate
from tops import logger, checkpointer
from torch.optim.lr_scheduler import ChainedScheduler
from omegaconf import OmegaConf
torch.backends.cudnn.benchmark = True

from pyexpat import model
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
import requests
import torchvision
from PIL import Image

coco_names = ['bakgrunn', 'bil', 'lastebil', 'buss', 'motorsykkel', 'sykkel', 'sparkesykkel', 'person', 'syklist']
# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    print(outputs[0][0])

    pred_classes = [coco_names[i] for i in outputs[1].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    
    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices

def draw_boxes(boxes, labels, classes, image):
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image

@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option("--evaluate-only", default=False, is_flag=True, help="Only run evaluation, no training.")
def train(config_path: Path, evaluate_only: bool):
    logger.logger.DEFAULT_SCALAR_LEVEL = logger.logger.DEBUG
    cfg = utils.load_config(config_path)


    model_path = "outputs/task_2_5/checkpoints/77400.ckpt"
    image_path = "data/tdt4265_2022/images/train/trip007_glos_Video00000_0.png"
    
    
    model = instantiate(cfg.model)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    #print(checkpoint)
    #model.load_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(checkpoint["model"])


    image = np.array(Image.open(image_path))
    image_float_np = np.float32(image) / 255
    # define the torchvision image transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    input_tensor = transform(image)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = input_tensor.to(device)
    # Add a batch dimension:
    input_tensor = input_tensor.unsqueeze(0)

    model.eval().to(device)

    # Run the model and display the detections
    boxes, classes, labels, indices = predict(input_tensor, model, device, 0.9)
    image = draw_boxes(boxes, labels, classes, image)

    # Show the image:
    Image.fromarray(image)

    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
    target_layers = [model.backbone]
    cam = AblationCAM(model,
                    target_layers, 
                    use_cuda=torch.nn.cuda.is_available(), 
                    reshape_transform=fasterrcnn_reshape_transform,
                    ablation_layer=AblationLayerFasterRCNN(),
                    ratio_channels_to_ablate=1.0)

    # or a very fast alternative:

    cam = EigenCAM(model,
                target_layers, 
                use_cuda=torch.nn.cuda.is_available(), 
                reshape_transform=fasterrcnn_reshape_transform) #Endre transform
                
    grayscale_cam = cam(input_tensor, targets=targets)
    # Take the first image in the batch:

    
    grayscale_cam = grayscale_cam[0, :]
    cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    # And lets draw the boxes again:
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
    Image.fromarray(image_with_bounding_boxes)

if __name__ == "__main__":
    train()
