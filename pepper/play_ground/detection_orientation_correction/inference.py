from utils import *

# Detectron2 config put in a seperate module
class detectroninference:
    def __init__(self,model_path,num_cls=1,name_classes=["pepp"]):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"))
        self.cfg.DATASETS.TRAIN = ("cuc_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 10
        self.cfg.MODEL.WEIGHTS = model_path  # Let training initialize from model zoo
        self.cfg["MODEL"]["ANCHOR_GENERATOR"]["ASPECT_RATIOS"][0]=[0.5,1.0,1.5]
        #cfg["INPUT"]["MASK_FORMAT"]='bitmask'
        self.cfg["INPUT"]["RANDOM_FLIP"]="horizontal"
        #cfg["INPUT"]["RANDOM_FLIP"]="horizontal"
        self.cfg["INPUT"]["ROTATE"]=[-2.0,2.0]
        self.cfg["INPUT"]["LIGHT_SCALE"]=2
        self.cfg["INPUT"]["Brightness_SCALE"]=[0.5,1.5]
        self.cfg["INPUT"]["Contrast_SCALE"]=[0.5,2]
        self.cfg["INPUT"]["Saturation_SCALE"]=[0.5,2]
        #cfg["BASE_LR"]=1e-5
        self.cfg.SOLVER.IMS_PER_BATCH = 1
        self.cfg.SOLVER.BASE_LR = 1e-3  # pick a good LR
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 500
        self.cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
        self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES=1
        self.cfg.MODEL.RETINANET.NUM_CLASSES=1
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        self.predictor = DefaultPredictor(self.cfg)
        self.cuc_metadata = MetadataCatalog.get("pepp").set(thing_classes=name_classes)

    def apply_mask(self,mask,img):
        all_masks=np.zeros(mask.shape,dtype=np.uint8)
        all_patches=np.zeros((*mask.shape,3),dtype=np.uint8)
        """Apply the given mask to the image."""
        for i in range(all_masks.shape[0]):
                all_masks[i][:, :] = np.where(mask[i] == True,255,0)
                for j in range(3):
                    all_patches[i][:, :,j] = np.where(mask[i] == True,img[:,:,j],0)
        return all_masks,all_patches

    def pred(self,img):
        orig_img=img.copy()
        height,width=img.shape[:2]
        outputs = self.predictor(img) 
        v = Visualizer(img[:, :, ::-1],
                        metadata=self.cuc_metadata, 
                        scale=0.3, 
                        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
            )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
        masks,patches=self.apply_mask(masks,orig_img)
        classes=outputs["instances"].pred_classes.to("cpu").numpy()
        boxes=(outputs["instances"].pred_boxes.to("cpu").tensor.numpy())
        return out.get_image()[:, :, ::-1],masks,patches,boxes,classes,outputs["instances"].scores.to("cpu").numpy()


def fast_AILearner(model):
    learn_inf = load_learner(model)
    return learn_inf

