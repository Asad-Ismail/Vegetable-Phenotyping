import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from scipy.spatial import distance
from detectron2.evaluation import inference_on_dataset
from detectron2.evaluation import DatasetEvaluator
import detectron2.utils.logger as logger
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
import torch



# valid extensions for input image file
extensions=[".jpg",".JPEG",".jpg",".jpeg",".png"]

def load_points_dataset(f_p):
    """load keypoints(head/tail)from the text file
    Args:
        f_p ([type]): File path containing the head and tail points (x,y) of each fruit in the image. Each Image can have multiple fruits

    Returns:
        [type]: Dictionary of file names as keys and corresponding fruit points as values
    """
    with open(f_p,"r") as f:
        all_lines=f.readlines()
    points={}
    i=0
    while(i<len(all_lines)):
        if(i>len(all_lines)):
            break
        line=all_lines[i].split(",")
        label=line[0]
        file=line[3]
        first_point=None
        second_point=None
        if label=="head":
            first_point=(int(line[1]),int(line[2]))
        elif label =="tail":
            second_point=(int(line[1]),int(line[2]))
        i+=1
        if(i<len(all_lines)):
            line2=all_lines[i].split(",")
            if line2[3]==file:
                if line2[0]=="head":
                    first_point=(int(line2[1]),int(line2[2]))
                elif line2[0] =="tail":
                    second_point=(int(line2[1]),int(line2[2]))
                i+=1
        if file in points:
            # file already in dictionary append the list
            #print(f"Appending the file to existing one {file}")
            points[file].append([first_point,second_point])
        else:
            points[file]=[[first_point,second_point]]
    return points


def get_imagePoints(imgPoints,nz_points):
    """Associate the points to right fruit

    Args:
        imgPoints ([type]): All keypoints in the image
        nz_points ([type]): Desired fruit non zero points to match

    Returns:
        [type]: tuple of keypoints(head/tail) belongig to the fruit
    """
    mindistances=[]
    for point in imgPoints:
        cpoint_head=None
        cpoint_tail=None
        # very high intitial distances
        dist_tail=12345
        dist_head=12345
        if point is not None and point[0] is not None:
            cpoint_head=min(nz_points,key=lambda c:distance.euclidean(c,point[0]))
            dist_head=distance.euclidean(cpoint_head,point[0])
        if point is not None and point[1] is not None:
            cpoint_tail=min(nz_points,key=lambda c:distance.euclidean(c,point[1]))
            dist_tail=distance.euclidean(cpoint_tail,point[1])
        mindistances.append(dist_tail+dist_head)
    index_min = min(range(len(mindistances)), key=mindistances.__getitem__)
    res=imgPoints[index_min]
    return res


def get_veg_dataset(img_dir,gt_points):
    """Get dataset for Points and Segmentation

    Args:
        img_dir ([str]): path to input images
        gt_points ([dict]): dictionary of filenames and keypoints

    Returns:
        [type]: Dictionary with fields to train instance segmentation and keypoint network
    """
    json_files = [json_file for json_file in os.listdir(img_dir) if json_file.endswith(".json")] 
    dataset_dicts = []
    for idx,json_file in enumerate(json_files):
        for ext in extensions:
            filename=json_file.split(".")[0]+ext
            c_fname=os.path.join(img_dir,filename)
            img=cv2.imread(c_fname)
            if img is not None:
                break
        if img is None:
            raise (f"Image Not Found for annotation {json_file}")
        with open(os.path.join(img_dir,json_file)) as f:
            imgs_anns = json.load(f)
        record = {}      
        height, width = img.shape[:2] 
        record["file_name"] = c_fname
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        annos = imgs_anns["shapes"]
        objs = []
        for anno in annos:
            #assert not anno["region_attributes"]
            px = [x for x,y in anno]
            py = [y for x,y in anno]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            #print("Getting point annotations!!")
            keypoints=get_imagePoints(gt_points[filename],poly)
            ## fix points which are none
            if keypoints[0] is not None:
                x1,y1=keypoints[0]
                c1=2
            else:
                x1,y1=0.0,0.0
                c1=1
            if keypoints[0] is not None:
                x2,y2=keypoints[1]
                c2=2
            else:
                x2,y2=0.0,0.0
                c2=1
            poly = [p for x in poly for p in x]
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "keypoints": [x1+0.5,y1+0.5,c1,x2+0.5,y2+0.5,c2],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        print(f"Processed Data # {idx}")
        #logger.logging.info(f"Processed Data # {idx}")
        dataset_dicts.append(record)
    return dataset_dicts



def draw_points(datalist,k=5):
    """Draw dataset keypoints on the input Image

    Args:
        datalist ([type]): Dictionary containing the 
        k (int, optional): number of images to display. Defaults to 5.
    """
    for i in range(k):
        data=datalist[i]
        ann=data["annotations"]
        fn=data["file_name"]
        img=cv2.imread(fn)
        scale_percent = 50
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dsize=(width,height)
        for d in ann:
            kp=d["keypoints"]
            x1,y1=int(kp[0]),int(kp[1])
            x2,y2=int(kp[3]),int(kp[4])
            cv2.circle(img,(x1,y1),14,(255,255,0),-1)
            cv2.circle(img,(x2,y2),14,(255,0,255),-1)
        cv2.imshow("Keypoints",cv2.resize(img,dsize=dsize))
        cv2.waitKey(0)



if __name__=="__main__":
    #Check the data loader
    kp_file="/media/asad/adas_cv_2/test_kp.csv"
    img_dir="/media/asad/ADAS_CV/datasets_Vegs/pepper/one_annotated/"
    gt_points=load_points_dataset(kp_file)
    for d in ["train","valid"]:
        DatasetCatalog.register("pep_" + d, lambda d=d: get_veg_dataset(img_dir + d,gt_points))
        MetadataCatalog.get("pep_" + d).set(thing_classes=["pep"])
        MetadataCatalog.get("pep_" + d).set(keypoint_names=["head","tail"])
        MetadataCatalog.get("pep_" + d).set(keypoint_flip_map=[("head","head"),("tail","tail")])
        MetadataCatalog.get("pep_" + d).set(keypoint_connection_rules=[("head","tail",(0,255,255))])
    dataset_dicts = DatasetCatalog.get("pep_train")
    vis=True
    if vis:
        draw_points(dataset_dicts)
        
    