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
from fastai.vision.all import *
import cv2


def visualize_points(points, img_dir, imgname):
    img = cv2.imread(os.path.join(img_dir, imgname))
    # print(imgPoints)
    imgPoints = points[imgname]
    for i, point in enumerate(imgPoints):
        if i == 0:
            if point is not None:
                cv2.circle(img, point, 14, (255, 255, 0), -1)
        if i == 1:
            if point is not None:
                cv2.circle(img, point, 14, (255, 0, 255), -1)
    # fig=plt.figure(figsize=(10,10))
    # plt.imshow(img[...,::-1])
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def scale_predictions(pred, imgShape, pred_size=224):
    scale_x = imgShape[1] / pred_size
    scale_y = imgShape[0] / pred_size
    pred = pred[0]
    pred[:, 0] = pred[:, 0] * scale_x
    pred[:, 1] = pred[:, 1] * scale_y
    return pred


def get_angle(p0, p1, p2):
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
    return np.degrees(angle)


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """
    height, width = mat.shape[:2]  # image shape has 3 dimensions
    image_center = (
        width / 2,
        height / 2,
    )  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]
    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def check_points(cropedPatches, points, RESIZE=224):
    # fig = plt.figure(figsize=(10, 10))
    # plt.title("Points ResNet34", loc="center")
    # plt.rcParams['figure.figsize'] = [45, 45]
    # nrows = int(np.ceil(len(points.keys()) / 2))
    # ncols = 2
    vis_img = None

    for i, img in enumerate(cropedPatches):
        img = img.copy()
        pointsize = img.shape[0] // 30
        assert img is not None, "Image not found!!"
        if points[i][0] is not None:
            cv2.circle(img, points[i][0], pointsize, (255, 255, 0), -1)
        if points[i][1] is not None:
            cv2.circle(img, points[i][1], pointsize, (255, 0, 255), -1)
        if points[i][0] is not None and points[i][1] is not None:
            # cv2.line(img,tuple(points[i][0]),tuple(points[i][1]),(255,255,255),2)
            ref = (img.shape[1], points[i][1][1])
            # cv2.line(img,tuple(points[i][1]),ref,(255,255,255),2)
        if vis_img is None:
            vis_img = cv2.resize(img, (RESIZE, RESIZE))
        else:
            vis_img = np.concatenate(
                [vis_img, cv2.resize(img, (RESIZE, RESIZE))], axis=1
            )
        # ax = fig.add_subplot(nrows, ncols, i + 1)
        # ax.imshow(img[..., ::-1])
        # plt.tight_layout()
    return vis_img


def apply_rotation(cropedPatches, predPoints, th=5):
    corrected_imgs = []
    for i, crop in enumerate(cropedPatches):
        img = crop.copy()
        h, w, _ = img.shape
        head, tail = None, None
        if (
            predPoints[i][0] is not None
            and predPoints[i][0][0] > 0
            and predPoints[i][0][1] > 0
        ):
            head = list(predPoints[i][0])
        if (
            predPoints[i][1] is not None
            and predPoints[i][1][0] > 0
            and predPoints[i][1][1] > 0
        ):
            tail = list(predPoints[i][1])
        if head is not None and tail is not None:
            if abs(head[0] - w) < th:
                head[0] -= th
            if abs(tail[0] - w) < th:
                tail[0] -= th
            ref = (img.shape[1], tail[1])
            angle = get_angle(head, tail, ref)
            rotated = rotate_image(img, -angle)
            corrected_imgs.append(rotated)
    return corrected_imgs
