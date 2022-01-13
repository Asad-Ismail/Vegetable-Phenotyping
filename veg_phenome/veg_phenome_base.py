# Author Asad Ismail
# Provides Base class for veg phenotpying

import numpy as np
import cv2
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color


class VEGPHENOMEBASE:
    """
    Base Phenotyping class. Provides basic phenotype measurements. Each Veg can ingherit and modify the methods of this class
    """

    def __init__(self, ppx=None, ppy=None) -> None:
        self.ppx = ppx
        self.ppy = ppy

    def __repr__(self) -> str:
        return (
            "Base class of Veg phenotyping. Overload methods for each fruit/veg as necessary. Currently Implements following crop agnostic traits"
            + f"{dir(self)}"
        )

    def min_max_segment(self, ref, xs, ys):
        """Calculate min, max of a patch. Take a reference point x and calculate min and max y for it

        Args:
            ref int: Reference x for which we need to find min and max y
            xs (Union[list, np.array()]): All sorted xs
            ys (Union[list, np.array()]): All sorted ys

        Returns:
            [Tuple(List,int)]: [segment_point_min,segment_point_max,Segment_width]
        """
        m_p = np.argwhere(xs == ref)
        y_p = ys[m_p]
        ymax = np.max(y_p)
        ymin = np.min(y_p)
        out = [(ref, ymin), (ref, ymax), ymax - ymin]
        return out

    def find_max_height(self, xs, ys):
        """Find Max height of fruit. Height is defined across x dimension.
        Args:
            xs (Union[list, np.array()]): All sorted xs
            ys (Union[list, np.array()]): All sorted ys

        Returns:
             [Tuple(List,int)]: [height_points, max_height]
        """
        # Loop throgh all ys and find min and max x
        assert len(xs) == len(ys), "Lengths of xs and ys are not equal"
        unique_y = sorted(set(ys))
        height_points = []
        maxheight = -1
        height_points = None
        for y in unique_y:
            selected_index = np.argwhere(ys == y)
            selected_xs = xs[selected_index]
            xmin = np.min(selected_xs)
            xmax = np.max(selected_xs)
            height = xmax - xmin
            if height > maxheight:
                height_points = [(xmin, y), (xmax, y)]
                maxheight = height
        assert maxheight >= 0, "Max Height could not found"
        return height_points, maxheight

    def find_mid_height_width(self, tops, bottoms):
        """Finds Width of patch at mid height

        Args:
            tops (List[Points]): Top boundary of fruit
            bottoms (List[Points]): Lower boundary of fruit

        Returns:
            [Tuple(List[Points],int)]: [Mid width Points, mid_width]
        """
        mid_height = len(tops) // 2
        width_points = [tops[mid_height], bottoms[mid_height]]
        width = bottoms[mid_height][1] - tops[mid_height][1]
        return width_points, width

    def find_mid_width_height(self, xs, ys):
        """Find height of patch at mid width

        Args:
            xs (Union[list, np.array()]): All sorted xs
            ys (Union[list, np.array()]): All sorted ys
        Returns:
            [Tuple(List[Points],int)]: [Mid Height Points, mid_height]
        """
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

    def find_max_width(self, tops, bottoms):
        """Return Max width of patch

        Args:
            tops (List[Points]): Top boundary of fruit
            bottoms (List[Points]): Lower boundary of fruit

        Returns:
            [Tuple(List[Points],int)]: [Max Width Points, Max Width]
        """
        max_index, max_width = None, -1
        for i in range(len(tops)):
            width = bottoms[i][1] - tops[i][1]
            if width > max_width:
                max_width = width
                max_index = i
        width_points = [tops[max_index], bottoms[max_index]]
        return width_points, max_width

    @staticmethod
    def ray_trace_segment(img, init, direction="y", to_origin=True):
        """Ray trace from the inital points to some end  like end of non zero pixels in image. Is useful for fruits that turns and make an S shape

        Args:
            img ([np.array(H,W)]): Gray scale Image of dim HxW
            init ([int]): Point coordinate x or y to start the ray tracing from
            direction (str, optional): [description]. Direction to move to the ray trace. Defaults to "y".
            to_origin (int, optional): [description]. To go to origin or not(For x direction origin is to left). Defaults to True.

        Returns:
            [type]: [description]
        """
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
                        segments.append([firstPoint, secondPoint, secondPoint[1] - firstPoint[1]])
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

    def find_curve_height(self, top, bottom, mask, ign_pct=15):
        """Find curved height of patch. Find mid point of patch ignoring the start and end given ignore percentage and then extend to both sides
        Curved Height >= Mid Height
        Args:
            tops (List[Points]): Top boundary of fruit
            bottoms (List[Points]): Lower boundary of fruit
            mask ([np.array]): gray scale image for extending the curved height to boundaries
            ign_pct (int, optional): %age of fruit to ignore from start and end. Defaults to 15.

        Returns:
            [List[Points]]: List of points constituting the curved height
        """
        c_length = []
        # curved length causes issues at the borders so ignoring length at borders
        ign_pct = ign_pct / 100
        for i in range(int(len(top) * ign_pct), int(len(top) * (1 - ign_pct))):
            c_length.append((top[i][0], round((top[i][1] + bottom[i][1]) / 2)))
        # Ray trace to extend the curve length on left
        segment_left = VEGPHENOMEBASE.ray_trace_segment(mask, c_length[0], direction="x", to_origin=1)
        c_length = segment_left + c_length
        # Ray trace to extend the curve length on right
        segment_right = VEGPHENOMEBASE.ray_trace_segment(mask, c_length[-1], direction="x", to_origin=0)
        c_length = c_length + segment_right
        return c_length

    def ellipse_fitting_normalized(self, mask):
        """Normalized Ellipse Fitting. Fits ellipse to contour. Draw ellipse to image and find IoU between mask and drawn ellipse.
        The vaue returned is 1-Iou varies between 0 and 1. 0 means no error and 1 means maximum error

        Args:
            mask ([np.array]): Gray scale mask image of patch
        Returns:
            [int]: 1 - Iou
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = max(contours, key=cv2.contourArea)
        cont = contours
        (x, y), (MA, ma), angle = cv2.fitEllipse(cont)
        x, y = int(x), int(y)
        MA, ma = int(MA), int(ma)
        ellipse_img = np.zeros_like(mask)
        # draw complete ellipse form angle 0 to 360
        cv2.ellipse(ellipse_img, (x, y), (MA // 2, ma // 2), angle, 0, 360, 255, -1)
        inter = np.logical_and(mask, ellipse_img)
        union = np.logical_or(mask, ellipse_img)
        eps = 1e-6
        iou = np.sum(inter) / (np.sum(union) + eps)
        return (x, y, MA, ma, angle), 1 - iou

    def box_fit_normalized(self, mask):
        """Normalized Box Fitting. Draw box to image and find IoU between mask and drawn box.
        The vaue returned is 1-Iou varies between 0 and 1. 0 means no error and 1 means maximum error

        Args:
            mask ([np.array]): Gray scale mask image of patch
        Returns:
            [int]: 1-IoU
        """
        x, y, w, h = cv2.boundingRect(mask)
        zer_img = np.zeros_like(mask)
        cv2.rectangle(zer_img, (x, y), (w, h), 255, -1)
        # cv2.imwrite("box_test.png",zer_img)
        inter = np.logical_and(mask, zer_img)
        union = np.logical_or(mask, zer_img)
        eps = 1e-6
        iou = np.sum(inter) / (np.sum(union) + eps)
        return (x, y, w, h), 1 - iou

    def get_color(self, patch, points):
        """Get color of patch given a mask image, Returns Mean R,G,B and Mean L,A,B of patch

        Args:
            patch ([np.array]): Color image patch of size H,W,3
            points ([np.array]): Points to calcualte the color from

        Returns:
            [List]: [(MeanR,MeanG,MeanB),(MeanL,MeanA,MeanB)]
        """
        b = np.mean(patch[[*points, 0]])
        g = np.mean(patch[[*points, 1]])
        r = np.mean(patch[[*points, 2]])
        rgb = sRGBColor(r, g, b, is_upscaled=True)
        lab = convert_color(rgb, LabColor)
        return [rgb.get_upscaled_value_tuple(), lab.get_value_tuple()]

    def eucledian_distance(self, points):
        """Calculates eucledian distance for list of points

        Args:
            points ([List(Tupe)]): List of points to calculate the eucledian distance
            ppx ([float]): Optional. scaling factor to convert from image to real world distance
            ppy ([float]): Optional. scaling factor to convert from image to real world distance

        Returns:
            [float]: Returns float eucledian distance
        """
        length = 0
        if self.ppx and self.ppy:
            for i in range(1, len(points)):
                length += np.sqrt(
                    (((points[i][0] - points[i - 1][0]) * self.ppx) ** 2)
                    + (((points[i][1] - points[i - 1][1]) * self.ppy) ** 2)
                )
        else:
            for i in range(1, len(points)):
                length += np.sqrt(((points[i][0] - points[i - 1][0]) ** 2) + ((points[i][1] - points[i - 1][1]) ** 2))
        return length

    def convert_pixels_to_measure(self, items, labels):
        """Convert pixels to real units cmm/mm if ppx and ppy are defined otherwise find lengths of curves. Labels are used to determine how to convert for area,
           perimeter and points

        Args:
            items ([List]): List of items for conversion
            labels ([type]): Labels for items

        Returns:
            [List]: Returns list of results converted real world distance
        """

        assert len(items) == len(labels), "Items and Labels should be equal"
        results = []
        for index, item in enumerate(items):
            if labels[index] == "area":
                result = item * self.ppx * self.ppy
                results.append(result)
            elif labels[index] == "perimeter":
                result = item * self.ppx
                results.append(result)
            else:
                result = self.eucledian_distance(item)
                results.append(result)

        return results

    def blockiness(self, top, bottom, ign_pct=20):
        """Find Blockiness (Width of fruits at start mid and end of fruit)

        Args:
            tops (List[Points]): Top boundary of fruit
            bottoms (List[Points]): Lower boundary of fruit
            ign_pct (int, optional): Percentage to ignore top and bottom of fruit. Defaults to 20.

        Returns:
            [tuple([[Point,Point],[Point,Point],[Point,Point]])]: Returns bottom, mid and top width of fruit
        """
        ign_pct = ign_pct / 100
        bottomindex = int(len(top) * ign_pct)
        midindex = int(len(top) // 2)
        topindex = int(len(top) * (1 - ign_pct))
        bottom_block = [top[bottomindex], bottom[bottomindex]]
        mid_block = [top[midindex], bottom[midindex]]
        top_block = [top[topindex], bottom[topindex]]
        return bottom_block, mid_block, top_block
