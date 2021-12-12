import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

extensions=[".jpg",".JPEG",".jpg",".jpeg",".png"]
def get_cuc_dicts(img_dir):
    json_files = [json_file for json_file in os.listdir(img_dir) if json_file.endswith(".json")] 
    dataset_dicts = []
    for idx,json_file in enumerate(json_files):
        #if idx==100:
        #    return dataset_dicts
        for ext in extensions:
            filename=json_file.split(".")[0]+ext
            c_fname=os.path.join(img_dir,filename)
            img=cv2.imread(c_fname)
            if img is not None:
                break
        if img is None:
            raise (f"Image Not Found for {json_file}")
        with open(os.path.join(img_dir,json_file)) as f:
            imgs_anns = json.load(f)

        record = {}      
        height, width = img.shape[:2]
        
        record["file_name"] = c_fname
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        #cv2.imshow("Image",img)
        #cv2.waitKey(0)
        annos = imgs_anns["shapes"]
        objs = []
        for anno in annos:
            #assert not anno["region_attributes"]
            px = [x for x,y in anno]
            py = [y for x,y in anno]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        print(f"Processed images {idx}")
        dataset_dicts.append(record)
    return dataset_dicts


if __name__=="__main__":

    for d in ["train", "valid"]:
        #DatasetCatalog.register("cuc_" + d, lambda d=d: get_cuc_dicts("/media/asad/ADAS_CV/cucumber_data/modified_cuc_data/" + d))
        DatasetCatalog.register("cuc_" + d, lambda d=d: get_cuc_dicts("/media/asad/ADAS_CV/cuc/data/" + d))
        MetadataCatalog.get("cuc_" + d).set(thing_classes=["cuc"])
    cuc_metadata = MetadataCatalog.get("cuc_train")

    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    print(cfg)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cuc_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 10
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml")  # Let training initialize from model zoo
    #cfg.MODEL.WEIGHTS = "/home/asad/dev/detectron2/output_resnet101_config_up1/model_0017999.pth"
    #cfg.MODEL.WEIGHTS="/media/asad/ADAS_CV/cuc/output_resnet101_axelspring/model_final.pth"
    cfg.MODEL.WEIGHTS="/media/asad/ADAS_CV/cuc/output_axels2/model_final.pth"
    #cfg.MODEL.WEIGHTS="/media/asad/ADAS_CV/cuc/output_axelMay/model_0002499.pth"

    cfg["MODEL"]["ANCHOR_GENERATOR"]["ASPECT_RATIOS"][0]=[0.03,1.0,6.0]
    #cfg["INPUT"]["MASK_FORMAT"]='bitmask'
    cfg["INPUT"]["RANDOM_FLIP"]="horizontal"
    #cfg["BASE_LR"]=1e-5
    cfg.SOLVER.IMS_PER_BATCH = 3
    cfg.SOLVER.BASE_LR = 1e-5  # pick a good LR
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES=1
    cfg.MODEL.RETINANET.NUM_CLASSES=1
    cfg.OUTPUT_DIR="/media/asad/ADAS_CV/cuc/output_axelMay" 
    print(cfg.OUTPUT_DIR)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()

    