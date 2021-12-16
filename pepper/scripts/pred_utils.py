import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
from detectron2.structures import BoxMode
from colormath.color_objects import LabColor, XYZColor, sRGBColor
from colormath.color_conversions import convert_color
from fastai.vision.all import *

#global points collection
points={}


# Detectron2 config put in a seperate module
class detectroninference:
    def __init__(self,model_path,num_cls=1,name_classes=["pepp"]):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"))
        #self.cfg.merge_from_file("/mnt/lib/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml")
        self.cfg.DATASETS.TRAIN = ("cuc_train",)
        self.cfg.DATASETS.TEST = ()
        self.cfg.DATALOADER.NUM_WORKERS = 4
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
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
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
        outputs = self.predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
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
        #print(c)
        return out.get_image()[:, :, ::-1],masks,patches,boxes,classes,outputs["instances"].scores.to("cpu").numpy()

def get_labels(filename):
    with open(filename,"r") as f:
        all_lines=f.readlines()
    i=0
    while(i<len(all_lines)):
        if(i>len(all_lines)):
            break
        line=all_lines[i].split(",")
        label=line[0]
        file=line[3]
        first_point=None
        second_point=None
        #print(i)
        #print(line)
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
                #print(line2)
        #print(first_point,second_point)
        points[file]=[first_point,second_point]




def get_y(fname):
        f=fname.name
        ps=points[f]
        im = PILImage.create(fname)
        out=[]
        # put out of bound in bound
        for x in ps:
            if x is None:
                x=(-1,-1)
            if x[0] > im.size[0]:
                x[0]=im.size[0]-3
            if x[1] > im.size[1] :
                x[1]=im.size[1]-3
            out.append(x)
        out=np.array(out,dtype=np.float)
        return tensor(out)

def fast_AILearner(model):
    learn_inf = load_learner(model)
    return learn_inf
    item_tfms = [Resize(224, method='squish',pad_mode="zeros")]
    #batch_tfms = [Flip(), Rotate(), Zoom(), Warp(),ClampBatch()]
    batch_tfms = [Flip(pad_mode="zeros"), Rotate(pad_mode="zeros"), Warp(pad_mode="zeros")]
    dblock = DataBlock(blocks=(ImageBlock, PointBlock),
                   get_items=get_image_files,
                   splitter=RandomSplitter(),
                   get_y=get_y,
                   item_tfms=item_tfms,
                   batch_tfms=batch_tfms)
    bs=4
    print(img_dir)
    dls = dblock.dataloaders(img_dir, bs=bs)
    dls.c = dls.train.after_item.c
    learn = cnn_learner(dls, models.resnet34, lin_ftrs=[100], ps=0.01,concat_pool=True,loss_func=MSELossFlat())
    learn.load(model)
    return learn

def scale_predictions(pred,imgShape,pred_size=224):
    scale_x=imgShape[1]/pred_size
    scale_y=imgShape[0]/pred_size
    pred=pred[0]
    pred[:,0]=pred[:,0]*scale_x
    pred[:,1]=pred[:,1]*scale_y
    return pred


def get_angle(p0, p1, p2):
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(angle)

def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
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
    """Return Fig with detected points on the images"""
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


def check_points_imgs(cropedPatches,points):
    """Draw points on image and return the full size images"""
    vis_imgs=[]
    for i,img in enumerate(cropedPatches):
        img=img.copy()
        pointsize=img.shape[0]//20
        assert img is not None, "Image not found!!"
        if points[i][0] is not None:
            cv2.circle(img,points[i][0],pointsize,(255,255,0),-1)
        if points[i][1] is not None:
            cv2.circle(img,points[i][1],pointsize,(255,0,255),-1)
        if points[i][0] is not None and points[i][1] is not None:
            cv2.line(img,tuple(points[i][0]),tuple(points[i][1]),(255,255,255),2)
            #ref=(img.shape[1],points[i][1][1])
            #cv2.line(img,tuple(points[i][1]),ref,(255,255,255),2)
        vis_imgs.append(img)
        #plt.tight_layout()
    return vis_imgs

def apply_rotation(cropedPatches,predPoints,th=5):
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
            #print(i)
            if abs(head[0]-w)<th:
                head[0]-=th
                #print(f"Lower threshold for head index {i}")
            if abs(tail[0]-w)<th:
                tail[0]-=th
            ref=(img.shape[1],tail[1]) 
            #cv2.circle(img,tuple(head),5,(255,0,0),-1)
            #cv2.circle(img,tuple(tail),5,(0,255,0),-1)
            #cv2.line(img,tuple(head),tuple(tail),(255,255,255),2)
            #cv2.line(img,tail,ref,(255,255,255),2)
            angle=get_angle(head,tail,ref)
            #rotated=rotate_image(vis_img,-angle2,tail)
            rotated=rotate_image(img,-angle)
            corrected_imgs.append(rotated)
    return corrected_imgs


def min_max_segment(ref, xs, ys):
    """Min Max of fruit"""
    segments = []
    m_p = np.argwhere(xs == ref)
    y_p = ys[m_p]
    ymax = np.max(y_p)
    ymin = np.min(y_p)
    segments.append([(ref, ymin), (ref, ymax), ymax - ymin])
    return segments

def find_max_height(xs, ys):
    """Max height of fruit"""
    left_index = np.argwhere(xs == xs[0])
    y_lefts = set(ys[left_index].flatten())
    right_index = np.argwhere(xs == xs[-1])
    y_rights = set(ys[right_index].flatten())
    y_lefts = list(y_lefts)
    # print(f"The length of lengths is {len(y_lefts)}")
    y = y_lefts[len(y_lefts) // 2]
    height_points = [(xs[0], y), (xs[-1], y)]
    height = xs[-1] - xs[0]
    return height_points, height


def find_max_height_revised(xs, ys):
    """Max height of fruit"""
    # Loop throgh all ys and find min and max x
    assert len(xs)==len(ys), "Lengths of xs and ys are not equal"
    unique_y=sorted(set(ys))
    height_points=[]
    maxheight=-1
    height_points=None
    for y in unique_y:
        selected_index=np.argwhere(ys==y)
        selected_xs=xs[selected_index]
        xmin=np.min(selected_xs)
        xmax=np.max(selected_xs)
        height=xmax-xmin
        if height>maxheight:
            height_points=[(xmin, y), (xmax, y)]      
            maxheight=height
    #print(f"k"*100)
    #print(height_points)
    #print(maxheight)
    return height_points, maxheight

def find_mid_height_width(tops, bottoms):
    """Mid height width of fruit"""
    mid_height = len(tops) // 2
    width_points = [tops[mid_height], bottoms[mid_height]]
    width = bottoms[mid_height][1] - tops[mid_height][1]
    return width_points, width

def find_mid_width_height(xs, ys):
    """Mid width height of fruit"""
    y_sorted = sorted(set(ys))
    mid_y = y_sorted[len(y_sorted) // 2]
    y_indexs = np.argwhere(ys == mid_y)
    matched_xs = xs[y_indexs]
    min_x = int(min(matched_xs))
    max_x = int(max(matched_xs))
    height_points = [(min_x, mid_y), (max_x, mid_y)]
    height = max_x - min_x
    # print(f"Height Points {height_points}")
    return height_points, height

def find_max_width(tops, bottoms):
    """Max width of fruit"""
    max_index, max_width = None, -1
    for i in range(len(tops)):
        width = bottoms[i][1] - tops[i][1]
        if width > max_width:
            max_width = width
            max_index = i
    width_points = [tops[max_index], bottoms[max_index]]
    return width_points, max_width


def find_curve_length(top, bottom, mask,ign_pct=15):
    """Curved backbone length of fruit"""
    c_length = []
    # curved length causes issues at the borders so ignoring length at borders
    ign_pct = ign_pct / 100
    for i in range(int(len(top) * ign_pct), int(len(top) * (1 - ign_pct))):
        c_length.append((top[i][0], round((top[i][1] + bottom[i][1]) / 2)))
    # Ray trace to extend the curve length on left
    segment_left = ray_trace_segment(mask, c_length[0], direction="x", to_origin=1)
    c_length = segment_left + c_length
    # Ray trace to extend the curve length on right
    segment_right = ray_trace_segment(mask, c_length[-1], direction="x", to_origin=0)
    c_length = c_length + segment_right
    return c_length


def ellipse_fitting(mask):
    """Ellipse fitting of fruit"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = max(contours, key = cv2.contourArea)
    #assert len(contours) == 1, "More than one contours in ellipse mask"
    #cont = contours[0]
    cont=contours
    (x, y), (MA, ma), angle = cv2.fitEllipse(cont)
    x, y = int(x), int(y)
    MA, ma = int(MA), int(ma)
    ellipse_img = np.zeros_like(mask)
    #draw complete ellipse form angle 0 to 360
    cv2.ellipse(ellipse_img, (x, y), (MA // 2, ma // 2), angle, 0, 360, 255)
    elipse_contours, _ = cv2.findContours(ellipse_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    elipse_contour = max(elipse_contours, key = cv2.contourArea)
    #assert len(elipse_contours) == 1, "More than one contours in ellipse"
    #elipse_contour = elipse_contours[0]
    error = cv2.matchShapes(cont, elipse_contour, 1, 0.0)
    return (x,y,MA,ma,angle), error

def blockiness(top, bottom,ign_pct=20):
    """Blockiness at different points of fruits"""
    ign_pct = ign_pct / 100
    bottomindex = int(len(top) * ign_pct)
    midindex = int(len(top) // 2)
    topindex = int(len(top) * (1 - ign_pct))
    bottom_block = [top[bottomindex], bottom[bottomindex]]
    mid_block = [top[midindex], bottom[midindex]]
    top_block = [top[topindex], bottom[topindex]]
    return bottom_block, mid_block, top_block

def box_fit(mask):
    """Find bounding box of the detected cropped mask"""
    x, y, w, h = cv2.boundingRect(mask)
    return (x,y,w,h),w/h

def get_color(patch,points):
    b=np.mean(patch[[*points,0]])
    g=np.mean(patch[[*points,1]])
    r=np.mean(patch[[*points,2]])
    rgb=sRGBColor(r,g,b,is_upscaled=True)
    lab=convert_color(rgb, LabColor)
    return [rgb.get_upscaled_value_tuple(),lab.get_value_tuple()]
    #print(f"Average value of blue green and red are {b}, {g}, {r}")
    #print(f"RGB upscaled {rgb.get_upscaled_value_tuple()}")
    #print(f"LAB upscaled {lab.get_value_tuple()}")


def ray_trace_segment(img, init, direction="y", to_origin=1):
    """Ray trace from the inital points to some end  like end of non zero pixels in image. Is useful for fruits that turns and make an S shape
    Inputs and outputs are different based on x or y. May be better to write a differnt function for both"""
    assert len(img.shape) == 2, "Should be Gray scale image"
    h, w = img.shape
    if direction == "y":
        # Go downwards
        increment = 1
        x = init
        y = 0
        segments = []
        # Go along the height of image to find segments
        while y < h:
            firstPoint = None
            secondPoint = None
            if img[y, x]:  # find first non zero point
                firstPoint = (x, y)
                while y < h and img[y, x]:
                    y += increment
                # find last non zero point for that x, Replace below 4 with some more dynamic threshold
                if y - firstPoint[1] > 4:
                    secondPoint = (x, y)
                if firstPoint and secondPoint:
                    segments.append(
                        [firstPoint, secondPoint, secondPoint[1] - firstPoint[1]]
                    )
            y += increment
    elif direction == "x":
        x, y = init
        if to_origin:
            # Go Left
            increment = -1
        else:
            # Go Right
            increment = 1
        segments = []
        x += increment
        while x < w and x >= 0:
            if img[y, x]:
                segments.append((x, y))
            x += increment
    return segments

def eucledian_distance(points,ppx=None,ppy=None):
    length=0
    if ppx and ppy:
        for i in range(1,len(points)):
            length+=np.sqrt((((points[i][0]-points[i-1][0])*ppx)**2) +  (((points[i][1]-points[i-1][1])*ppy)**2))
    else:
        for i in range(1,len(points)):
            length+=np.sqrt(((points[i][0]-points[i-1][0])**2) +  ((points[i][1]-points[i-1][1])**2))
    return length

def convert_pixels_to_measure(area,perimeter,mid_height_width_p,max_width_p,mid_width_height_p,max_height_p,c_length,ppx=None,ppy=None):
    """Convert pixels to real units cmm/mm if ppx and ppy are defined otherwise find lengths of curves"""
    if ppx and ppy:
        area=area*ppx*ppy
        perimeter=perimeter*ppx
    mid_height_width=eucledian_distance(mid_height_width_p,ppx,ppy)
    max_width=eucledian_distance(max_width_p,ppx,ppy)
    mid_width_height=eucledian_distance(mid_width_height_p,ppx,ppy)
    max_height=eucledian_distance(max_height_p,ppx,ppy)
    c_length=eucledian_distance(c_length,ppx,ppy)
    return (area,perimeter,mid_height_width,max_width,mid_width_height,max_height,c_length)


def phenotype_measurement_pepper(img, patch,skip_outliers=None, pct_width=0.4,vis=False,ppx=None,ppy=None):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Get the maximum contour if multiple contours exist per fruit
    contours = max(contours, key = cv2.contourArea)
    points=contours.squeeze()
    # points are perimeter
    points = np.array(sorted(points, key=lambda x: x[0]))
    # Perimeter fruit
    perimeter = cv2.arcLength(contours,True)
    # Area fruit
    area = cv2.contourArea(contours)
    area_points = np.where(img != 0)
    #color first tuple is rgb, second is LAB
    fruit_colors=get_color(patch.copy(),area_points)
    #contour points are reverse of image points x is in 0 and y is in 1 dimension
    xs = points[:, 0]
    ys = points[:, 1]
    # sort xs
    u_xs = sorted(set(xs))
    top = []
    bottom = []
    widths = []
    # lengths
    max_height_p, max_height_v = find_max_height_revised(xs, ys)
    mid_width_height_p, mid_width_height_v = find_mid_width_height(xs, ys)
    for i, x in enumerate(u_xs):
        # segments=ray_trace_segment(img,x)
        segments = min_max_segment(x, xs, ys)
        for segment in segments:
            top.append(segment[0])
            bottom.append(segment[1])
            widths.append(segment[2])

    assert (
        len(top) == len(bottom) == len(widths)
    ), "The top, bottom and width points are not equal"

    # widths and curve lengths
    max_width_p, max_midth_v = find_max_width(top, bottom)
    mid_height_width_p, mid_height_width_v = find_mid_height_width(top, bottom)
    c_length = find_curve_length(top, bottom, img)
    b_block, m_block, t_block = blockiness(top, bottom)
    
    # metric 14
    ellipse,ellipse_err=ellipse_fitting(img)
    #meric 15
    box,box_aspect=box_fit(img)
    if vis:
        fig,vis_images,labels = vis_phenotypes(
            points,
            area_points,
            max_height_p,
            mid_width_height_p,
            max_width_p,
            mid_height_width_p,
            c_length,
            b_block,
            m_block,
            t_block,
            box,
            ellipse,
            patch.copy(),
        )
    else:
        fig,vis_images,labels=None,None,None
    # median_width=statistics.median(widths)
    area,perimeter,mid_height_width,max_width,mid_width_height,max_height,c_length=\
    convert_pixels_to_measure(area,perimeter,mid_height_width_p,max_width_p,mid_width_height_p,max_height_p,c_length,ppx,ppy)
    # shape metrics
    #metric 8
    maxheight_to_maxwidth=max_height/max_width
    #metrix 9
    mid_width_height_to_mid_height_width=mid_width_height/mid_height_width
    # metric 10
    c_length_to_mid_height_width=c_length/mid_height_width
    # metric 11
    proximal_blockiness=eucledian_distance(t_block)/mid_height_width
    # metric 12
    distal_blockiness=eucledian_distance(b_block)/mid_height_width
    # metric 13
    fruit_triangle=eucledian_distance(t_block)/eucledian_distance(b_block)
    
    return [
        area,
        perimeter,
        mid_height_width,
        max_width,
        mid_width_height,
        max_height,
        c_length,
        maxheight_to_maxwidth,
        mid_width_height_to_mid_height_width,
        c_length_to_mid_height_width,
        proximal_blockiness,distal_blockiness,fruit_triangle,
        ellipse_err,box_aspect,
        fruit_colors,
        fig,vis_images,labels,
        # Added for validation and debugging
        max_height_p,
        mid_height_width_p
    ]
    
def draw_text(img,length,width):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org1 = (int(img.shape[1]*0.3), int(img.shape[0]*0.4))
    org2 = (int(img.shape[1]*0.3), int(img.shape[0]*0.7))
    # fontScale
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2
    img = cv2.putText(img, f"l: {length}", org1, font, fontScale, color, thickness, cv2.LINE_AA)
    img = cv2.putText(img, f"w: {width}", org2, font, fontScale, color, thickness, cv2.LINE_AA)
    return img

def run_inference(img,pepp,learn):
    if img is not None:
        detected_cucumber,all_masks,all_patches,boxes,*_=pepp.pred(img)
    else:
        print(f"Invalid Image skipping!!")
        return
    cropedPatches=[]
    cropedmasks=[]
    croppedboxes = []
    for i,patch in enumerate(all_patches):
        #print(patch.shape)
        x, y, w, h = cv2.boundingRect(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
        newImg = patch[y:y+h, x:x+w]
        mask=all_masks[i][y:y+h, x:x+w]
        mask = np.where(mask < 200, 0, 255).astype(np.uint8)
        cropedPatches.append(newImg)
        cropedmasks.append(mask)
        croppedboxes.append((x, y, w, h))
        #break
        
    predPoints={}
    for i,crop in enumerate(cropedPatches):
        img = PILImage.create(crop[...,::-1])
        pred=learn.predict(img)
        preds=scale_predictions(pred,img.shape)
        predPoints[i]=[tuple(preds[0].round().long().numpy()),tuple(preds[1].round().long().numpy())]
    
    # Correction
    corrected_imgs=apply_rotation(cropedPatches,predPoints)
    corrected_masks=apply_rotation(cropedmasks,predPoints)
        
    return (
        detected_cucumber,
        corrected_imgs,
        corrected_masks,
        croppedboxes,
        cropedPatches,
        predPoints,
        #original boxes are returned to get Gt based on location of boxes
        boxes
    )


def save_results_figs(
    detected_cucumber, cropedPatches, predPoints, corrected_imgs, fname, save_path
):
    """Save figures of results to visualize
    Args:
        detected_cucumber ([type]): Detection result from detectron2
        cropedPatches ([type]): Croped patch of original Image
        predPoints ([type]): predicted keypoints adjusted according to cropped patches
        corrected_imgs ([type]): orientation corrected Image
        fname ([type]): Filename for saving postfix
        save_path ([type]): "Path to save"
    """
    os.makedirs(save_path, exist_ok=True)
    fig = plt.figure(figsize=(10, 10))
    plt.title("Detected Pepper MASKRCNN", loc="center")
    plt.imshow(detected_cucumber[..., ::-1])
    fig.savefig(os.path.join(save_path, f"segmented_{fname}"))
    # show croped images
    fig = plt.figure(figsize=(10, 10))
    plt.title("Cropped", loc="center")
    nrows = np.ceil(len(cropedPatches) / 2)
    ncols = 2
    for i, patch in enumerate(cropedPatches):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(patch[..., ::-1])
    fig.savefig(os.path.join(save_path, f"Cropped_{fname}"))
    fig = check_points(cropedPatches, predPoints)
    fig.savefig(os.path.join(save_path, f"Points_{fname}"))
    fig = plt.figure(figsize=(10, 10))
    # plt.title("Corrected Orientation",loc='center')
    # plt.rcParams['figure.figsize'] = [45, 45]
    nrows = np.ceil(len(corrected_imgs) / 2)
    ncols = 2
    for i, correct_img in enumerate(corrected_imgs):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.imshow(correct_img[..., ::-1])
    fig.savefig(os.path.join(save_path, f"corrected_{fname}"))
    ## Concatenate Results
    merge_scans = os.path.join(save_path + "merge_results")
    os.makedirs(merge_scans, exist_ok=True)
    img_det = cv2.imread(os.path.join(save_path, f"segmented_{fname}"))
    img_crop = cv2.imread(os.path.join(save_path, f"Cropped_{fname}"))
    img_points = cv2.imread(os.path.join(save_path, f"Points_{fname}"))
    img_corrected = cv2.imread(os.path.join(save_path, f"corrected_{fname}"))
    joined_img = np.concatenate([img_det, img_crop, img_points, img_corrected], axis=1)
    cv2.imwrite(os.path.join(merge_scans, f"{fname}"), joined_img)
    
    
def save_results_cv(
    detected_cucumber, cropedPatches, predPoints, corrected_imgs, fname, save_path
):
    """Save images of results to visualize
    Args:
        detected_cucumber ([type]): Detection result from detectron2
        cropedPatches ([type]): Croped patch of original Image
        predPoints ([type]): predicted keypoints adjusted according to cropped patches
        corrected_imgs ([type]): orientation corrected Image
        fname ([type]): Filename for saving postfix
        save_path ([type]): "Path to save"
    """
    print(f"Save path {save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    cv2.imwrite(os.path.join(save_path, f"Segmented_{fname}"),detected_cucumber)
    # show croped images
    for i, patch in enumerate(cropedPatches):
        cv2.imwrite(os.path.join(save_path, f"Cropped_{i}_{fname}"),patch)
    
    point_imgs=check_points_imgs(cropedPatches, predPoints)
    
    for i, point_img in enumerate(point_imgs):
        cv2.imwrite(os.path.join(save_path, f"Points_{i}_{fname}"),point_img)
    
    for i, correct_img in enumerate(corrected_imgs):
        cv2.imwrite(os.path.join(save_path, f"Corrected_{i}_{fname}"),correct_img)
        

skipped_nan=0
def error(preds,gts,pred_ls_ps,pred_ws_ps):
    global skipped_nan
    fruits_idx=[]
    length_errs=[]
    width_errs=[]
    org_l=[]
    org_w=[]
    pred_l=[]
    pred_w=[]
    pred_l_p=[]
    pred_w_p=[]
    #print(gts)
    #for i in range(len(gts)):
    for i in gts.keys():
        if gts[i][1] is None or gts[i][2] is None:
            skipped_nan+=1
            continue
        #ignore ambigious predictions
        #if gts[i][0] in [43,84,67]:
        #    continue
        l_err=abs(preds[i][0]-gts[i][1])
        w_err=abs(preds[i][1]-gts[i][2])
        length_errs.append(l_err)
        width_errs.append(w_err)
        fruits_idx.append(gts[i][0])
        org_l.append(gts[i][1])
        org_w.append(gts[i][2])
        pred_l.append(preds[i][0])
        pred_w.append(preds[i][1])
        pred_l_p.append({gts[i][0]:pred_ls_ps[i]})
        pred_w_p.append({gts[i][0]:pred_ws_ps[i]})
    return length_errs,width_errs,fruits_idx,org_l,org_w,pred_l,pred_w,pred_l_p,pred_w_p

def vis_phenotypes(
    perimeter,
    area,
    max_height,
    mid_width_height,
    max_width,
    mid_height_width,
    curved_length,
    bottom_block,
    mid_block,
    top_block,
    box,
    ellipse,
    img,
    line_width=5,
):
    """Visualization of phenotypes"""
    fig = plt.figure(figsize=(10, 20))
    nrows = 7
    ncols = 2
    vis_imgs = []

    vis_img = img.copy()
    for i in range(len(perimeter)):
        cv2.circle(vis_img, tuple(perimeter[i]), line_width, (255, 0, 0))
    vis_imgs.append(vis_img)

    vis_img = img.copy()
    cv2.line(vis_img, max_height[0], max_height[1], (255, 0, 0), line_width)
    vis_imgs.append(vis_img)

    vis_img = img.copy()
    cv2.line(vis_img, max_width[0], max_width[1], (0, 255, 0), line_width)
    vis_imgs.append(vis_img)

    vis_img = img.copy()
    cv2.line(vis_img, mid_width_height[0], mid_width_height[1], (0, 0, 255), line_width)
    vis_imgs.append(vis_img)

    vis_img = img.copy()
    cv2.line(
        vis_img, mid_height_width[0], mid_height_width[1], (255, 255, 0), line_width
    )
    vis_imgs.append(vis_img)

    vis_img = img.copy()
    for i in range(len(curved_length)):
        cv2.circle(vis_img, curved_length[i], line_width, (255, 0, 0))
    vis_imgs.append(vis_img)
    # Shape
    # max height to maximum width
    vis_img = img.copy()
    cv2.line(vis_img, max_height[0], max_height[1], (255, 0, 0), line_width)
    cv2.line(vis_img, max_width[0], max_width[1], (0, 255, 0), line_width)
    vis_imgs.append(vis_img)
    
    # Height mid width/ width mid height
    vis_img = img.copy()
    cv2.line(vis_img, mid_width_height[0], mid_width_height[1], (0, 0, 255), line_width)
    cv2.line(
        vis_img, mid_height_width[0], mid_height_width[1], (255, 255, 0), line_width
    )
    vis_imgs.append(vis_img)

    # Top blockiness/ Mid height Width
    vis_img = img.copy()
    cv2.line(vis_img, top_block[0], top_block[1], (0, 0, 255), line_width)
    cv2.line(
        vis_img, mid_height_width[0], mid_height_width[1], (255, 255, 0), line_width
    )
    vis_imgs.append(vis_img)

    # Bottom blockiness/ Mid height Width
    vis_img = img.copy()
    cv2.line(vis_img, bottom_block[0], bottom_block[1], (0, 0, 255), line_width)
    cv2.line(
        vis_img, mid_height_width[0], mid_height_width[1], (255, 255, 0), line_width
    )
    vis_imgs.append(vis_img)

    # Upper blockiness/ Bottom blockiness
    vis_img = img.copy()
    cv2.line(vis_img, top_block[0], top_block[1], (0, 0, 255), line_width)
    cv2.line(vis_img, bottom_block[0], bottom_block[1], (255, 255, 0), line_width)
    vis_imgs.append(vis_img)
    
   
    # Ellipse vis
    vis_img = img.copy()
    x,y,MA,ma,angle=ellipse
    cv2.ellipse(vis_img, (x, y), (MA // 2, ma // 2), angle, 0, 360, (255,0,0),line_width)
    vis_imgs.append(vis_img)
    
    # Box vis
    vis_img = img.copy()
    x,y,w,h=box
    vis_img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),line_width)
    vis_imgs.append(vis_img)

    labels = [
        "Perimeter",
        "Max height",
        "Max Width",
        "Mid Width Height",
        "Mid Height Width",
        "Curved Length",
        "max height/max width",
        "height mid-width/width mid-height",
        "Proximal Fruit Blokiness",
        "Distal Fruit Blokiness",
        "Fruit Shape Triangle",
        "Elipse",
        "Rectangle",
    ]
    
    assert len(labels)==len(vis_imgs), f"Label length {len(labels)}  and vis images {len(vis_imgs)} not equal"
    for i, patch in enumerate(vis_imgs):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.set_title(labels[i])
        ax.imshow(patch[..., ::-1])
    #fig.subplots_adjust(hspace=0.5, wspace=0.04)
    fig.tight_layout()
    return fig,vis_imgs,labels
