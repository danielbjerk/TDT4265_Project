import sys
from tabnanny import check
assert sys.version_info >= (3, 7), "This code requires python version >= 3.7"
import click
import torch
from pathlib import Path
from ssd import utils
from tops.config import instantiate
from tops import logger
import warnings
import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision
from PIL import Image
import os

torch.backends.cudnn.benchmark = True
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

dataset_classes = ['bakgrunn', 'bil', 'lastebil', 'buss', 'motorsykkel', 'sykkel', 'sparkesykkel', 'person', 'syklist']

# This will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(dataset_classes), 3))

def predict(input_tensor, model, detection_threshold):
    outputs = model(input_tensor)

    pred_classes = [dataset_classes[i] for i in outputs[0][1].cpu().numpy()]
    pred_labels = outputs[0][1].cpu().numpy()
    pred_scores = outputs[0][2].detach().cpu().numpy()
    pred_bboxes = outputs[0][0].detach().cpu().numpy()

    
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    
    
    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:

            box = np.array([
                pred_bboxes[index][0] * 1024, 
                pred_bboxes[index][1] * 128, 
                pred_bboxes[index][2] * 1024, 
                pred_bboxes[index][3] * 128]).astype(np.int32)

            boxes.append(box)
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
@click.argument("image_path", default="data/tdt4265_2022/images/train/trip007_glos_Video00000_37.png", type=click.Path(exists=True, dir_okay=False, path_type=str))
def train(config_path: Path, image_path: Path):

    logger.logger.DEFAULT_SCALAR_LEVEL = logger.logger.DEBUG
    cfg = utils.load_config(config_path)

    checkpoint_dir = f"{cfg.output_dir}/checkpoints"
    checkpoint_name = sorted([f for f in os.listdir(checkpoint_dir) if os.path.isfile(os.path.join(checkpoint_dir, f))])[-1]
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Instantiate and load model from checkpoint
    model = instantiate(cfg.model)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"])

    # Load image, convert to floating point and compose transform
    image = np.array(Image.open(image_path))
    image_float_np = np.float32(image) / 255
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
    boxes, classes, labels, _ = predict(input_tensor, model, 0.5)
    image = draw_boxes(boxes, labels, classes, image)

    # Show the image:
    Image.fromarray(image).show()
    return

    target_layers = [model.feature_extractor]
    targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]

    cam = AblationCAM(model,
                    target_layers, 
                    use_cuda=torch.cuda.is_available(), 
                    reshape_transform=fasterrcnn_reshape_transform,
                    ablation_layer=AblationLayerFasterRCNN(),
                    ratio_channels_to_ablate=1.0)

    """
    # or a very fast alternative:
    cam = EigenCAM(model,
                target_layers, 
                use_cuda=torch.cuda.is_available(), 
                reshape_transform=fasterrcnn_reshape_transform) #Endre transform
    """

    grayscale_cam = cam(input_tensor, targets=targets)

    return

    # Take the first image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
    # And lets draw the boxes again:
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, cam_image)
    Image.fromarray(image_with_bounding_boxes)

if __name__ == "__main__":
    train()
