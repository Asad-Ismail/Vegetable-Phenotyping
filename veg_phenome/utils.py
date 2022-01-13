import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, sys
import pandas as pd

np.random.seed(44)
##Generate some random colors for visualization
colors = [tuple(np.random.randint(0, 255) for _ in range(3)) for _ in range(255)]


def vis_phenotypes(img, items, labels, line_width=5):
    """Visualize phenotype and return the list of vis_images. Decides on type of vis by number of points in each item

    Args:
        img ([np.array]): Image
        items ([List[List]]): List of items to visualize
        labels ([List[str]]): Lables
        line_width (int, optional): width of item to be drawn on image. Defaults to 5.

    Returns:
        [pyplot.fig,List[np.array]]: fig and list of vis_mage
    """

    assert len(items) == len(labels), "Items and Labels should be equal"
    color_ind = 0
    fig = plt.figure(figsize=(40, 40))
    vis_imgs = []
    for index, item in enumerate(items):
        vis_img = img.copy()
        # If the length of item ==1. It should have a nested list with length 4 so we can draw ratios
        if len(item) == 1:
            for j, inneritem in enumerate(item[0]):
                assert len(inneritem) == 2, "For grouping the number of ratios should be 2"
                cv2.line(vis_img, inneritem[0], inneritem[1], colors[color_ind], line_width)
                color_ind += 1
            vis_imgs.append(vis_img)
        # If number of points are 2 draw a line
        elif len(item) == 2:
            for i in range(len(item)):
                cv2.line(vis_img, item[0], item[1], colors[color_ind], line_width)
                color_ind += 1
            vis_imgs.append(vis_img)
        # If number of points are 4 draw a box
        elif len(item) == 4:
            x, y, w, h = item
            vis_img = cv2.rectangle(vis_img, (x, y), (x + w, y + h), colors[color_ind], line_width)
            color_ind += 1
            vis_imgs.append(vis_img)
        # If number of points are 5 draw an Ellipse
        elif len(item) == 5:
            x, y, MA, ma, angle = item
            cv2.ellipse(vis_img, (x, y), (MA // 2, ma // 2), angle, 0, 360, colors[color_ind], line_width)
            color_ind += 1
            vis_imgs.append(vis_img)
        # If number of points are greater than 2 loop and draw
        else:
            for i in range(len(item)):
                cv2.circle(vis_img, tuple(item[i]), line_width, colors[color_ind])
            color_ind += 1
            vis_imgs.append(vis_img)

    ncols = 3
    nrows = len(labels) // 3 + 1
    assert len(labels) == len(vis_imgs), f"Label length {len(labels)}  and vis images {len(vis_imgs)} not equal"
    for i, patch in enumerate(vis_imgs):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.set_title(labels[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(patch[..., ::-1])
    # fig.tight_layout()
    # plt.show()
    return fig, vis_imgs


def vis_points_imgs(cropedPatches, points):
    """Vislualize keypoints on Images

    Args:
        cropedPatches ([List[np.array]]): List of images
        points ([List[np.array]]): List of keypoints

    Returns:
        [List[np.array]]: List of images with keypoint drawn on them
    """
    vis_imgs = []
    for i, img in enumerate(cropedPatches):
        img = img.copy()
        pointsize = img.shape[0] // 20
        assert img is not None, "Image not found!!"
        if points[i][0] is not None:
            cv2.circle(img, points[i][0], pointsize, (255, 255, 0), -1)
        if points[i][1] is not None:
            cv2.circle(img, points[i][1], pointsize, (255, 0, 255), -1)
        if points[i][0] is not None and points[i][1] is not None:
            cv2.line(img, tuple(points[i][0]), tuple(points[i][1]), (255, 255, 255), 2)
        vis_imgs.append(img)
    return vis_imgs


def save_results_cv(detected_cucumber, cropedPatches, predPoints, corrected_imgs, fname, save_path="results"):
    """Save images of results to visualize
    Args:
        detected_cucumber ([type]): Detection result from detectron2
        cropedPatches ([type]): Croped patch of original Image
        predPoints ([type]): predicted keypoints adjusted according to cropped patches
        corrected_imgs ([type]): orientation corrected Image
        fname ([type]): Filename for saving postfix
        save_path ([type]): "Path to save"
    """

    save_path = os.path.join(save_path, "Detection")
    os.makedirs(save_path, exist_ok=True)

    cv2.imwrite(os.path.join(save_path, f"Segmented_{fname}"), detected_cucumber)
    # show croped images
    for i, patch in enumerate(cropedPatches):
        cv2.imwrite(os.path.join(save_path, f"Cropped_{i}_{fname}"), patch)

    point_imgs = vis_points_imgs(cropedPatches, predPoints)

    for i, point_img in enumerate(point_imgs):
        cv2.imwrite(os.path.join(save_path, f"Points_{i}_{fname}"), point_img)

    for i, correct_img in enumerate(corrected_imgs):
        cv2.imwrite(os.path.join(save_path, f"Corrected_{i}_{fname}"), correct_img)


def check_results(inp, plot_names):
    """Check and perform sanity tests on results

    Args:
        inp ([dict]): Results dictionray
    """
    ex_p_names = [i.split("_")[0] for i in inp["Image"]]
    p_names = set(ex_p_names)
    o_p_names = set(plot_names)
    ## These plots contain no useful information
    print(f"Follwing plots are empty {o_p_names - p_names}")
    for k in inp.keys():
        mx = max(inp[k])
        mn = min(inp[k])
        if isinstance(mx, int) or isinstance(mn, int) or isinstance(mx, float) or isinstance(mn, float):
            assert mx >= 0 and mn <= 10000 and mx != mn, "The bounds of results are all zero"
            print(f"Max and min of {k} is {mx}, {mn}")


def save_dictionary_to_xls(inp, save_path="."):
    results_df = pd.DataFrame.from_dict(inp, orient="index").transpose()
    results_df.to_excel(os.path.join(save_path, "output.xlsx"))


def merge_results(inp, save_path="."):
    average_fruits = []
    agg_results = {}
    ex_p_names = [i.split("_")[0] for i in inp["Image"]]
    p_names = set(ex_p_names)
    name_map = {
        "Area": "FRAREA",
        "Perimeter": "FRPER",
        "Mid_Width": "FRWMH",
        "Max_Width": "FRMAXW",
        "Mid_Height": "FRMHMW",
        "Max_Height": "FRMH",
        "Curved_Height": "FRSPH",
        "Maxheight_to_maxwidth": "FRSIEMAX",
        "Midheight_to_midwidth": "FRSIEMID",
        "Curveheight_to_midwidth": "FRCSI",
        "Proximal_blockiness": "FRPB",
        "Distall_blockiness": "FRDB",
        "Fruit_triangle": "FRSTRI",
        "Ellipse_Error": "FRELIP",
        "Box_Aspect": "FRRECT",
    }
    for name in p_names:
        n_fruits = 0
        all_phen = [[] for i in range(len(name_map.keys()))]
        for i, n in enumerate(inp["Image"]):
            if name == n.split("_")[0]:
                n_fruits += 1
                for j, pn in enumerate(name_map.keys()):
                    all_phen[j].append(inp[pn][i])
        print(f"Fruits in {name} are {n_fruits}")
        assert n_fruits == len(all_phen[0]) == len(all_phen[-1])
        average_fruits.append(n_fruits)
        if "PLOT BID" not in agg_results:
            agg_results["PLOT BID"] = [name]
            for l, (k, v) in enumerate(name_map.items()):
                agg_results[v] = [np.mean(all_phen[l])]
                agg_results[v + "_CNT"] = [len(all_phen[l])]
                agg_results[v + "_MAX"] = [np.max(all_phen[l])]
                agg_results[v + "_MIN"] = [np.min(all_phen[l])]
                agg_results[v + "_SD"] = [np.std(all_phen[l])]
        else:
            agg_results["PLOT BID"].append(name)
            for l, (k, v) in enumerate(name_map.items()):
                agg_results[v].append(np.mean(all_phen[l]))
                agg_results[v + "_CNT"].append(len(all_phen[l]))
                agg_results[v + "_MAX"].append(np.max(all_phen[l]))
                agg_results[v + "_MIN"].append(np.min(all_phen[l]))
                agg_results[v + "_SD"].append(np.std(all_phen[l]))

    save_dictionary_to_xls(agg_results)
