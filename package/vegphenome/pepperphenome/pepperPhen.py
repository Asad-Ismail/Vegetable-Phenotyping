import numpy as np
import cv2
import os
import sys

# For running standalone
#sys.path.append("../")
#sys.path.append("vegphenome")
#from basephenome.vegPhenBase import VEGPHENOMEBASE
#from basephenome.utils import *
#from basephenome.predUtils import *

# For creating docs
# sys.path.append("vegphenome")
from vegphenome.basephenome.vegPhenBase import VEGPHENOMEBASE
from vegphenome.basephenome.utils import *
from vegphenome.basephenome.predUtils import *
from tqdm import tqdm


class PepperPhenSer:
    """Pepper phenotyping class provides methods for detection and phenotyping peppers. It chains Detection/Segmetnation network with keypoint network in
    series. Instance segmentation network is trained on whole images while orientation correction is based on fruits patches.Alternative and preffered way to that
    will be to have one multitask network that does both instance segmentation and keypoit detection"""

    def __init__(self, det_model, orient_model, imagepath, vis_results=True, save_path="results") -> None:
        """

        Args:
            det_model ([str]): Path to pretrained weights of MaskRCNN
            orient_model ([str]): Path to pretrianed weights of oritnation correctio model
            imagepath ([str]): Path to input Image directory which will be used
            vis_results (bool, optional): To visualize and save the results and intermediate. Defaults to True.
            save_path (str, optional): Path to save the results. Defaults to "results".

        """
        maskrcnn = det_model
        resnet34 = orient_model
        self.imagepath = imagepath
        self.vis = vis_results
        self.fnames = []
        self.all_imgs = []
        self.plot_names = []
        self.empty_images = []
        self.results = {}
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        print(f"Loading Detection Model!!")
        self.pepp = detectroninference(maskrcnn)
        print(f"Loading Point Model!!")
        self.learn = fast_AILearner(resnet34)
        print(f"Getting all files in the image")
        self.get_image_files()

    def get_image_files(self):
        """Get All images in the directory"""
        for file_index, filename in enumerate(os.listdir(self.imagepath)):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                f_p = os.path.join(self.imagepath, filename)
                self.fnames.append(filename)
                self.all_imgs.append(f_p)

    def run_detection(self, img):
        """Takes Input image and detect fruit, detect points and then run orrientation correctio

        Args:
            img ([np.array]): Image for detection

        Returns:
            [Tuple(List[])]: [description]
        """
        if img is not None:
            detected_cucumber, all_masks, all_patches, boxes, *_ = self.pepp.pred(img)
        else:
            print(f"Invalid Image skipping!!")
            return
        cropedPatches = []
        cropedmasks = []
        croppedboxes = []
        for i, patch in enumerate(all_patches):
            x, y, w, h = cv2.boundingRect(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
            newImg = patch[y : y + h, x : x + w]
            mask = all_masks[i][y : y + h, x : x + w]
            mask = np.where(mask < 200, 0, 255).astype(np.uint8)
            cropedPatches.append(newImg)
            cropedmasks.append(mask)
            croppedboxes.append((x, y, w, h))
            # break

        predPoints = {}
        for i, crop in enumerate(cropedPatches):
            img = PILImage.create(crop[..., ::-1])
            pred = self.learn.predict(img)
            preds = scale_predictions(pred, img.shape)
            predPoints[i] = [tuple(preds[0].round().long().numpy()), tuple(preds[1].round().long().numpy())]

        # Correction
        corrected_imgs = apply_rotation(cropedPatches, predPoints)
        corrected_masks = apply_rotation(cropedmasks, predPoints)

        return (
            detected_cucumber,
            corrected_imgs,
            corrected_masks,
            croppedboxes,
            cropedPatches,
            predPoints,
            # original boxes are returned to get Gt based on location of boxes
            boxes,
        )

    def measure_phenotype(self, mask, patch):
        """Meaure Phenotypes given a detected fruit

        Args:
            mask (np.array): A gray scale mask image of fruit H,W
            patch (np.array): RGB patch image of fruit H,W,3
            vis (bool, optional): To save the phenotpe results. Defaults to False.

        Returns:
            [tuple(List[floats],fig[plt.fig],List[np.array],List[str])]: Returns output phenotyes and viusalization images and labels
        """
        vegphen = VEGPHENOMEBASE(ppx=0.3, ppy=0.3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Get the maximum contour if multiple contours exist per fruit
        contours = max(contours, key=cv2.contourArea)
        points = contours.squeeze()
        # sort the points from left to right
        points = np.array(sorted(points, key=lambda x: x[0]))
        # Perimeter fruit
        perimeter = cv2.arcLength(contours, True)
        # Area fruit
        area = cv2.contourArea(contours)
        area_points = np.where(mask != 0)
        # color first tuple is rgb, second is LAB
        fruit_colors = vegphen.get_color(patch.copy(), area_points)
        # contour points are reverse of image points x is in 0 and y is in 1 dimension
        xs = points[:, 0]
        ys = points[:, 1]
        # sort xs. Might be redundant
        u_xs = sorted(set(xs))
        top = []
        bottom = []
        widths = []
        # Heights
        max_height_p, max_height_v = vegphen.find_max_height(xs, ys)
        mid_width_height_p, mid_width_height_v = vegphen.find_mid_width_height(xs, ys)
        # Find top, bottons and widths
        for i, x in enumerate(u_xs):
            # segments=ray_trace_segment(img,x)
            segments = vegphen.min_max_segment(x, xs, ys)
            top.append(segments[0])
            bottom.append(segments[1])
            widths.append(segments[2])
        assert len(top) == len(bottom) == len(widths), "The top, bottom and width points are not equal"
        # widths and curve lengths
        max_width_p, max_midth_v = vegphen.find_max_width(top, bottom)
        mid_height_width_p, mid_height_width_v = vegphen.find_mid_height_width(top, bottom)
        c_height = vegphen.find_curve_height(top, bottom, mask)
        b_block, m_block, t_block = vegphen.blockiness(top, bottom)
        # Convert pixels to real world distances
        phen_pixel = [
            area,
            perimeter,
            mid_height_width_p,
            max_width_p,
            mid_width_height_p,
            max_height_p,
            c_height,
            t_block,
            b_block,
        ]
        labels = [
            "area",
            "perimeter",
            "mid_height_width",
            "max_width",
            "mid_width_height",
            "max_height",
            "c_height",
            "t_block",
            "b_block",
        ]
        assert len(phen_pixel) == len(labels), "Length of labels and phenotyes should be same"
        phen_meas = vegphen.convert_pixels_to_measure(phen_pixel, labels=labels)
        # shape metrics
        # metric 8
        index_n = labels.index("max_height")
        index_d = labels.index("max_width")
        maxheight_to_maxwidth = phen_meas[index_n] / phen_meas[index_d]
        # metrix 9
        index_n = labels.index("mid_width_height")
        index_d = labels.index("mid_height_width")
        mid_width_height_to_mid_height_width = phen_meas[index_n] / phen_meas[index_d]
        # metric 10
        index_n = labels.index("c_height")
        index_d = labels.index("mid_height_width")
        c_height_to_mid_height_width = phen_meas[index_n] / phen_meas[index_d]
        # metric 11
        index_n = labels.index("t_block")
        index_d = labels.index("mid_height_width")
        proximal_blockiness = phen_meas[index_n] / phen_meas[index_d]
        # metric 12
        index_n = labels.index("b_block")
        index_d = labels.index("mid_height_width")
        distal_blockiness = phen_meas[index_n] / phen_meas[index_d]
        # metric 13
        index_n = labels.index("t_block")
        index_d = labels.index("b_block")
        fruit_triangle = phen_meas[index_n] / phen_meas[index_d]
        # metric 14
        ellipse, ellipse_err = vegphen.ellipse_fitting_normalized(mask)
        # meric 15
        box, box_err = vegphen.box_fit_normalized(mask)

        ## Vis labels
        if self.vis:
            vis_labels = [
                "Perimeter",
                "Area",
                "Max height",
                "Max Width",
                "Mid Width Height",
                "Mid Height Width",
                "Curved Height",
                "Max height/Max width",
                "Height mid-width/Width mid-height",
                "Proximal Fruit Blokiness",
                "Distal Fruit Blokiness",
                "Fruit Shape Triangle",
                "Rectangle",
                "Ellipse",
            ]
            # The legth of grouping elements should be one, two for line, four for box, five for ellipse and >5 for points
            vis_items = [
                points,
                list(zip(area_points[1], area_points[0])),
                max_height_p,
                max_width_p,
                mid_width_height_p,
                mid_height_width_p,
                c_height,
                [[max_height_p, max_width_p]],
                [[mid_width_height_p, mid_height_width_p]],
                [[t_block, mid_height_width_p]],
                [[b_block, mid_height_width_p]],
                [[t_block, b_block]],
                box,
                ellipse,
            ]
            fig, vis_images = vis_phenotypes(patch.copy(), vis_items, vis_labels)
        else:
            fig, vis_images = None, None

        outputs = (
            phen_meas[:-2]
            + [maxheight_to_maxwidth]
            + [mid_width_height_to_mid_height_width]
            + [c_height_to_mid_height_width]
            + [proximal_blockiness]
            + [distal_blockiness]
            + [fruit_triangle]
            + [ellipse_err]
            + [box_err]
            # + [fruit_colors]
        )
        return outputs, fig, vis_images, labels

    def process_phenotype(self, corrected_masks, corrected_imgs, frame_name):
        label = [
            "Area",
            "Perimeter",
            "Mid_Width",
            "Max_Width",
            "Mid_Height",
            "Max_Height",
            "Curved_Height",
            "Maxheight_to_maxwidth",
            "Midheight_to_midwidth",
            "Curveheight_to_midwidth",
            "Proximal_blockiness",
            "Distall_blockiness",
            "Fruit_triangle",
            "Ellipse_Error",
            "Box_Aspect",
        ]
        for j, mask in enumerate(corrected_masks):
            assert len(mask.shape) == 2, "Should be Gray scale Mask"
            out, fig, phen_imgs, labels = self.measure_phenotype(mask, corrected_imgs[j])
            if "Image" not in self.results:
                self.results["Image"] = [frame_name]
                for i, k in enumerate(out):
                    self.results[label[i]] = [k]
            else:
                self.results["Image"].append(frame_name)
                for i, k in enumerate(out):
                    self.results[label[i]].append(k)
            # Visualize save
            if self.vis:
                fig.savefig(os.path.join(self.save_path, f"Phenotype_{j}_{frame_name}"))

    def __call__(self):
        for index in tqdm(range(len(self.all_imgs)), total=len(self.all_imgs)):
            print(f"Processing file {self.fnames[index]}")
            img = cv2.imread(self.all_imgs[index])
            (
                detected,
                corrected_imgs,
                corrected_masks,
                cropped_boxes,
                cropedPatches,
                predPoints,
                boxes,
            ) = self.run_detection(img)
            # print(f"Boxes are {boxes}")
            # Append plot names to count plot
            plot_name = self.fnames[index].split("_")[0]
            if plot_name not in self.plot_names:
                self.plot_names.append(plot_name)
            if not len(corrected_masks):
                self.empty_images.append(self.fnames[index])
            if self.vis:
                save_results_cv(detected, cropedPatches, predPoints, corrected_imgs, self.fnames[index])
            self.process_phenotype(corrected_masks, corrected_imgs, self.fnames[index])
        return True


if __name__ == "__main__":
    # img = cv2.imread("veg_phenome/corrected_mask.jpg")
    # mask = img[..., 0]
    det_model = "/media/asad/8800F79D00F79104/sagemaker-data/model_final.pth"
    orient_model = "/media/asad/8800F79D00F79104/sagemaker-data/points.pkl"
    imagepath = "/media/asad/8800F79D00F79104/sagemaker-data/test_images"
    pepper_phen = PepperPhenSer(det_model, orient_model, imagepath=imagepath, vis_results=True)
    # Call the function
    status = pepper_phen()
    results = pepper_phen.results
    check_results(results, pepper_phen.plot_names)
    save_dictionary_to_xls(results)
    merge_results(results)
    # print(measure_phenotype(mask, img, vis=True))
