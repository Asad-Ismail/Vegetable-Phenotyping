from utils import *



class detectroninference:
    """Inference if detectron2 detection and segmentation. The config of inference should be the same as training. 
    ## TODO get config from the training
    """
    def __init__(self,model_path,num_cls=1,name_classes=["pepp"]):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"))
        cfg.DATASETS.TRAIN = ("pep_train",)
        cfg.DATASETS.TEST = ()
        cfg.MODEL.WEIGHTS=model_path
        cfg["MODEL"]["ANCHOR_GENERATOR"]["ASPECT_RATIOS"][0]=[0.5,1.0,1.5]
        cfg["INPUT"]["RANDOM_FLIP"]="horizontal"
        cfg["INPUT"]["ROTATE"]=[-2.0,2.0]
        cfg["INPUT"]["LIGHT_SCALE"]=2
        cfg["INPUT"]["Brightness_SCALE"]=[0.5,1.5]
        cfg["INPUT"]["Contrast_SCALE"]=[0.5,2]
        cfg["INPUT"]["Saturation_SCALE"]=[0.5,2]
        cfg["MODEL"]["KEYPOINT_ON"]=True
        cfg.MODEL['ROI_KEYPOINT_HEAD']["NUM_KEYPOINTS"]=2
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.SOLVER.BASE_LR = 1e-5  # pick a good LR
        cfg.SOLVER.CHECKPOINT_PERIOD = 500
        cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES=1
        cfg.MODEL.RETINANET.NUM_CLASSES=1
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
        print(f"Test Image sizes {cfg.INPUT.MIN_SIZE_TEST},{cfg.INPUT.MAX_SIZE_TEST}")
        self.predictor = DefaultPredictor(cfg)
        self.pepp_metadata = MetadataCatalog.get("pepp").set(thing_classes=name_classes)
        self.pepp_metadata = MetadataCatalog.get("pep").set(keypoint_names=["head","tail"])
        self.pepp_metadata = MetadataCatalog.get("pep").set(keypoint_flip_map=[("head","head"),("tail","tail")])
        self.pepp_metadata = MetadataCatalog.get("pep").set(keypoint_connection_rules=[("head","tail",(0,255,255))])
        
    
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
                        metadata=self.pepp_metadata, 
                        scale=0.3, 
                        instance_mode=ColorMode.IMAGE_BW   
            )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        #print(outputs["instances"].to("cpu"))
        #print(outputs["instances"].to("cpu").pred_keypoints)
        #print(outputs["instances"].to("cpu").pred_masks.shape)
        masks = np.asarray(outputs["instances"].to("cpu").pred_masks)
        masks,patches=self.apply_mask(masks,orig_img)
        classes=outputs["instances"].pred_classes.to("cpu").numpy()
        boxes=(outputs["instances"].pred_boxes.to("cpu").tensor.numpy())
        #print(c)
        return out.get_image()[:, :, ::-1],masks,patches,boxes,classes,outputs["instances"].scores.to("cpu").numpy()
    
    

if __name__=="__main__":
    model_path="/media/asad/ADAS_CV/vegs_results/models/pepper/keypoints/model_final.pth"
    img_path="/media/asad/ADAS_CV/datasets_Vegs/pepper/one_annotated/train"
    print(f"Intializing Model!!")
    pepp=detectroninference(model_path)
    count=0
    print(f"Running Inference!!")
    for file_index,filename in enumerate(os.listdir(img_path)):
        if filename.endswith(".json") or filename.endswith(".jpg.png"):
            continue 
        f_p=os.path.join(img_path,filename)
        img=cv2.imread(f_p)
        if img is not None:
            count+=1
            print(f"Processing image {f_p} Count {count}")
            detected_pepp,all_masks,all_patches,boxes,*_=pepp.pred(img)
            #fig=plt.figure(figsize=(10,10))
            #plt.imshow(detected_pepp[...,::-1])
            cv2.imshow("Detected",detected_pepp)
            cv2.waitKey(0)