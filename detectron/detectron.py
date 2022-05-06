import os

from multiprocessing.dummy import freeze_support

from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

setup_logger()
register_coco_instances("tdt4265_train", {}, "data/tdt4265_2022_updated/train_annotations.json", "data/tdt4265_2022_updated/")
register_coco_instances("tdt4265_val", {}, "data/tdt4265_2022_updated/val_annotations.json", "data/tdt4265_2022_updated/")

if __name__ == '__main__':
    freeze_support()

    cfg = get_cfg()     
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("tdt4265_train",)  
    cfg.DATASETS.TEST = ()  
    
    cfg.MODEL.MASK_ON = False
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  
    cfg.SOLVER.MAX_ITER = 10000  
    cfg.SOLVER.STEPS = []        
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()