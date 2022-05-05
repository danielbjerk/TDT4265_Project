from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

from multiprocessing.dummy import freeze_support

import os
import cv2
import random
import time
from tqdm import tqdm

register_coco_instances("tdt4265_train", {}, "data/tdt4265_2022/train_annotations.json", "data/tdt4265_2022/")
register_coco_instances("tdt4265_val", {}, "data/tdt4265_2022/val_annotations.json", "data/tdt4265_2022/")


if __name__ == '__main__':

	freeze_support()

	cfg = get_cfg()
	cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

	cfg.DATASETS.TRAIN = ("tdt4265_train",)  
	cfg.MODEL.MASK_ON = False
	cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  
	cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6

	metadata_catalog = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
	metadata_catalog.set(**{"thing_classes": ["bil", "lastebil", "buss", "motorsykkel", "sykkel", "sparkesykkel", "person", "syklist"]})

	predictor = DefaultPredictor(cfg)

	evaluator = COCOEvaluator("tdt4265_val", cfg, False, output_dir="./output/")
	val_loader = build_detection_test_loader(cfg, "tdt4265_val")
	inference_on_dataset(predictor.model, val_loader, evaluator)

	validation_directory = "./data/tdt4265_2022_updated/images/val/"

	for file in tqdm(os.listdir(validation_directory)):
		image = cv2.imread(os.path.join(validation_directory, file))
		outputs = predictor(image)
		v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
		v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
		cv2.imwrite("./output/images/" + file, v.get_image()[:, :, ::-1])


	num_images_to_test = 10
	images = []
	
	for file in random.shuffle(os.listdir(validation_directory)):
		image = cv2.imread(os.path.join(validation_directory, file))
		images.append(image)

		if len(images) == num_images_to_test:
			break

	
	num_times_to_test = 100
	start_time = time.time()

	for i in range(num_images_to_test):
		_ = predictor(images[i % len(images)])
	
	total_time = time.time() - start_time

	print(f"FPS: {num_times_to_test / total_time}")
		
