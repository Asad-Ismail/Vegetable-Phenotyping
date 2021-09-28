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
        dist_tail=123456
        dist_head=123456
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


def get_angle(p0, p1, p2):
    """Get signed angle between two points

    Args:
        p0 ([type]): head point
        p1 ([type]): tail point
        p2 ([type]): reference point

    Returns:
        [type]: Angle in degrees
    """
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def rotate_image(mat, angle):
    """Rotates Image by angle

    Args:
        mat ([type]): Image 
        angle ([type]): Angle in Degrees

    Returns:
        [type]: Rotated padded Image
    """
    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def check_points(cropedPatches,points):
    """Visualize detected Keypoints from DNN

    Args:
        cropedPatches ([type]): Cropped Patches
        points ([type]): predicted Points to be drawn

    Returns:
        [type]: Matplotlib figure of detcted keypoints drawn on Image
    """
    fig = plt.figure(figsize=(10, 10))
    plt.title("Points ResNet34",loc='center')
    #plt.rcParams['figure.figsize'] = [45, 45]
    nrows=int(np.ceil(len(points.keys())/2))
    ncols=2
    
    for i,img in enumerate(cropedPatches):
        img=img.copy()
        pointsize=img.shape[0]//20
        assert img is not None, "Image not found!!"
        if points[i][0] is not None:
            cv2.circle(img,points[i][0],pointsize,(255,255,0),-1)
        if points[i][1] is not None:
            cv2.circle(img,points[i][1],pointsize,(255,0,255),-1)
        if points[i][0] is not None and points[i][1] is not None:
            #cv2.line(img,tuple(points[i][0]),tuple(points[i][1]),(255,255,255),2)
            ref=(img.shape[1],points[i][1][1])
            #cv2.line(img,tuple(points[i][1]),ref,(255,255,255),2)
        ax = fig.add_subplot(nrows,ncols,i+1)
        ax.imshow(img[...,::-1])
        #plt.tight_layout()
    return fig

def apply_rotation(cropedPatches,predPoints,th=5):
    """Correct Orientation of fruit by rotation based on two points

    Args:
        cropedPatches ([type]): Image patches to apply rotation correction
        predPoints ([type]): Pred keypoints used to calculate angles
        th (int, optional): Rotation correction does not work if the detected point is no the boundry fix threshold for those points. Defaults to 5.

    Returns:
        [type]: Rotation corrected Image Patches
    """
    corrected_imgs=[]
    for i,crop in enumerate(cropedPatches):
        img=crop.copy()
        h,w=img.shape[0:2]
        head,tail=None,None
        if predPoints[i][0] is not None and predPoints[i][0][0]>0 and predPoints[i][0][1]>0:
            head=list(predPoints[i][0])
        if predPoints[i][1] is not None and predPoints[i][1][0]>0 and predPoints[i][1][1]>0:
            tail=list(predPoints[i][1])
        #print(head,tail)
        if head is not None and tail is not None:
            if abs(head[0]-w)<th:
                head[0]-=th
            if abs(tail[0]-w)<th:
                tail[0]-=th
            ref=(img.shape[1],tail[1]) 
            angle=get_angle(head,tail,ref)
            rotated=rotate_image(img,-angle)
            corrected_imgs.append(rotated)
    return corrected_imgs

def save_results(detected_cucumber,cropedPatches,predPoints,corrected_imgs,fname,save_path):
    """Save figures of results to visualize

    Args:
        detected_cucumber ([type]): Detection result from detectron2
        cropedPatches ([type]): Croped patch of original Image
        predPoints ([type]): predicted keypoints adjusted according to cropped patches
        corrected_imgs ([type]): orientation corrected Image
        fname ([type]): Filename for saving postfix
        save_path ([type]): "Path to save"
    """
    fig=plt.figure(figsize=(10,10))
    plt.title("Detected Pepper MASKRCNN",loc='center')
    plt.imshow(detected_cucumber[...,::-1])
    fig.savefig(os.path.join(save_path,f"segmented_{fname}"))
    #show croped images
    fig = plt.figure(figsize=(10, 10))
    plt.title("Cropped",loc='center')
    nrows=np.ceil(len(cropedPatches)/2)
    ncols=2
    for i,patch in enumerate(cropedPatches):
        ax = fig.add_subplot(nrows,ncols,i+1)
        ax.imshow(patch[...,::-1])
    fig.savefig(os.path.join(save_path,f"Cropped_{fname}"))
    fig=check_points(cropedPatches,predPoints)
    fig.savefig(os.path.join(save_path,f"Points_{fname}"))
    fig = plt.figure(figsize=(10, 10))
    #plt.title("Corrected Orientation",loc='center')
    #plt.rcParams['figure.figsize'] = [45, 45]
    nrows=np.ceil(len(corrected_imgs)/2)
    ncols=2
    for i,correct_img in enumerate(corrected_imgs):
        ax = fig.add_subplot(nrows,ncols,i+1)
        ax.imshow(correct_img[...,::-1])
    fig.savefig(os.path.join(save_path,f"corrected_{fname}"))

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
        
    