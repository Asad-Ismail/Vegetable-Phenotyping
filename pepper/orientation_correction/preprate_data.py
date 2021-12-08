from utils import *
from scipy.spatial import distance

# Required for getting points for whole frames
from inference import detectroninference


class getLabelspatch:
    """
    Get Point labels where ech point was labelled on GT Patch or Detected Patch/fruit
    Directly get the points on fruit patch
    """

    def __init__(self, filename) -> None:
        self.filename = filename
        self.points = {}

    def get_labels(self, top="head", bottom="tail"):
        """Based on points annotation from MakeSense Overload to support Image Analysis"""
        with open(self.filename, "r") as f:
            all_lines = f.readlines()
        i = 0
        while i < len(all_lines):
            if i > len(all_lines):
                break
            line = all_lines[i].split(",")
            label = line[0]
            file = line[3]
            first_point = None
            second_point = None
            if label == top:
                first_point = (int(line[1]), int(line[2]))
            elif label == bottom:
                second_point = (int(line[1]), int(line[2]))
            i += 1
            if i < len(all_lines):
                line2 = all_lines[i].split(",")
                if line2[3] == file:
                    if line2[0] == top:
                        first_point = (int(line2[1]), int(line2[2]))
                    elif line2[0] == bottom:
                        second_point = (int(line2[1]), int(line2[2]))
                    i += 1
            self.points[file] = [first_point, second_point]

    def __call__(self):
        self.get_labels()
        return self.points


class getLabelsFrame:
    """
    Get Point labels where ech point was labelled on the origial Image.
    Runs the object Detector to get the mask of fruit in the original Image then get the closest point to the fruit in all the point annotated.
    It wont work if the object detector misses the objects assumes the Object detector works pretty well. Train Fruit detector before running it.

    Input: Img directory and label file
    Returns: Dict of imagenames and points
    """

    def __init__(self, img_dir, labelfile, weights, dst_dir, prefix="sc") -> None:
        self.points = {}
        self.points_out = {}
        self.imgdir = img_dir
        self.labelfile = labelfile
        self.inter_files = []
        self.matches_count = 0
        self.weights = weights
        self.imgs_names = []
        self.imgs_fpath = []
        # where patches needs to be stored
        self.dst_dir = dst_dir
        self.prefix = prefix

    def getFiles(self):
        """Get Image files and their full absolute paths"""
        for filename in os.listdir(self.imgdir):
            if filename.endswith(".json") or filename.endswith(".jpg.png"):
                continue
            f_p = os.path.join(self.imgdir, filename)
            self.imgs_fpath.append(f_p)
            self.imgs_names.append(filename)

    def interFiles(self, filename):
        if filename in self.points:
            imgPoints = self.points[filename]
            if len(imgPoints) >= 1:
                self.matches_count += 1
                self.inter_files.append(filename)

    def getInterfiles(self):
        """Get the Files which have the points annotation in the directory"""
        for i in range(len(self.imgs_names)):
            self.interFiles(self.imgs_names[i])

    def get_cropedpoints(self, f, img, orgImg, y, x):
        imgPoints = self.points[f]
        nz = np.nonzero(orgImg[..., 0])
        nz_points = [(x, y) for x, y in zip(nz[1], nz[0])]
        min_head_dis = 123456
        min_tail_dis = 123456
        cpoint_head = None
        cpoint_tail = None
        for point in imgPoints:
            if point is not None and point[0] is not None:
                # print(f"Original point Head {point[0]}")
                cpoint_head_tmp = min(
                    nz_points, key=lambda c: distance.euclidean(c, point[0])
                )
                tmp_head_distance = distance.euclidean(cpoint_head_tmp, point[0])
                if min_head_dis > tmp_head_distance:
                    # print("Updating head point")
                    # print(f"Closest point Head {cpoint_head}")
                    min_head_dis = tmp_head_distance
                    cpoint_head = (cpoint_head_tmp[0] - x, cpoint_head_tmp[1] - y)
                    # print(f"Modified Point head {cpoint_head}")
                    # cv2.circle(img,cpoint_head,4,(255,255,0),-1)
            if point is not None and point[1] is not None:
                cpoint_tail_tmp = min(
                    nz_points, key=lambda c: distance.euclidean(c, point[1])
                )
                tmp_tail_dist = distance.euclidean(cpoint_tail_tmp, point[1])
                if min_tail_dis > tmp_tail_dist:
                    # print("Updating bottom point")
                    min_tail_dis = tmp_tail_dist
                    cpoint_tail = (cpoint_tail_tmp[0] - x, cpoint_tail_tmp[1] - y)
            # cv2.circle(img,cpoint_tail,4,(255,0,255),-1)
        # fig=plt.figure(figsize=(10,10))
        # plt.imshow(img[...,::-1])
        return [cpoint_head, cpoint_tail]

    def get_Points(self, top="head", bottom="tail"):
        with open(self.labelfile, "r") as f:
            all_lines = f.readlines()
        i = 0
        while i < len(all_lines):
            if i > len(all_lines):
                break
            line = all_lines[i].split(",")
            label = line[0]
            file = line[3]
            first_point = None
            second_point = None
            if label == top:
                first_point = (int(line[1]), int(line[2]))
            elif label == bottom:
                second_point = (int(line[1]), int(line[2]))
            i += 1
            if i < len(all_lines):
                line2 = all_lines[i].split(",")
                if line2[3] == file:
                    if line2[0] == top:
                        first_point = (int(line2[1]), int(line2[2]))
                    elif line2[0] == bottom:
                        second_point = (int(line2[1]), int(line2[2]))
                    i += 1
            if file in self.points:
                # file already in dictionary append the list
                print(f"Appending the file to existing one {file}")
                self.points[file].append([first_point, second_point])
            else:
                self.points[file] = [[first_point, second_point]]

    def __call__(self,startIndex,**kwargs):
        self.get_Points(kwargs["top"],kwargs["bottom"])
        self.getFiles()
        self.getInterfiles()
        # build MASKRCNN inference
        print(f"Building Veg Detection Model!!")
        pepp = detectroninference(self.weights)
        for index, f in enumerate(self.inter_files,start=startIndex):
            img = cv2.imread(os.path.join(self.imgdir, f))
            if img is not None:
                detected_cucumber, all_masks, all_patches, boxes, *_ = pepp.pred(img)
                print(f"Processed Images {index}")
                for i, patch in enumerate(all_patches):
                    x, y, w, h = cv2.boundingRect(
                        cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    )
                    newImg = patch[y : y + h, x : x + w]
                    drawImg = newImg.copy()
                    # cropedPatches.append(newImg)
                    respoints = self.get_cropedpoints(f, drawImg, all_patches[i], y, x)
                    write_index = index * 100 + i
                    # print(write_index)
                    self.points_out[self.prefix + str(write_index) + ".png"] = respoints
                    print(f"The length of points are {len(self.points_out)}")
                    print(f"Writing Image file")
                    cv2.imwrite(
                        os.path.join(
                            self.dst_dir, self.prefix + str(write_index) + ".png"
                        ),
                        newImg,
                    )

        return self.points_out


def Copyfiles(points, src_dirs, dst_dir):
    import shutil

    skipped = 0
    filenames = points.keys()
    for filename in filenames:
        path = None
        for src_dir in src_dirs:
            path = os.path.join(src_dir, filename)
            if os.path.exists(path):
                break
        if not os.path.exists(path):
            skipped += 1
            continue
        # assert path is not None, "The file {} does not exists in any source directory"
        destination = os.path.join(dst_dir, filename)
        shutil.copy(path, destination)
    print(f"Coppied {len(filenames)-skipped}/{len(filenames)}")


def remove_bad(points, img_dir):
    imgs = get_image_files(Path(img_dir))
    removed = 0
    for imgfile in imgs:
        if imgfile.name in points:
            continue
        else:
            removed += 1
            imgfile.unlink()
    print(f"Removed {removed} Images")


def build_points(out_pkl):
    # Parsing Points from two different types of annotation one done on patch level and one on Frame level
    # 1 indicates patch level 2 indicates Frame level
    # Define labels by specifiying the labels and patch dir and frames and frame directories
    # Eidt the below files
    """
    files = {
        1: {
            "labels": [
                "/media/asad/ADAS_CV/datasets_Vegs/pepper/pepper_points/labels_my-project-name_2021-08-02-11-51-34.csv",
                "/media/asad/ADAS_CV/datasets_Vegs/pepper/pepper_points/labels_my-project-name_2021-08-02-08-37-48.csv",
            ],
            "src_dir": [
                "/media/asad/ADAS_CV/datasets_Vegs/pepper/sc_v1_v2_patches",
                "/media/asad/ADAS_CV/datasets_Vegs/pepper/sc_v1_v2_patches",
            ],
        },
        2: {
            "labels": [
                "/media/asad/ADAS_CV/datasets_Vegs/pepper/pepper_points/labels_my-project-name_2021-08-05-12-17-20.csv"
            ],
            "src_dir": ["/media/asad/ADAS_CV/scalecam_pepper_selected"],
        },
    }
    """
    # Hot peeper data
    """files = {
        1: {},
        2: {
            "labels": ["/media/asad/adas_cv_2/hot_pepper_2.csv"],
            "src_dir": [
                "/media/asad/ADAS_CV/datasets_Vegs/hot_peppers/scale_cam_test_batch/orig_img"
            ],
        },
    }"""
    
    files = {
        1: {
            "labels": [
                "/media/asad/ADAS_CV/datasets_Vegs/pepper/pepper_points/labels_my-project-name_2021-08-02-11-51-34.csv",
                "/media/asad/ADAS_CV/datasets_Vegs/pepper/pepper_points/labels_my-project-name_2021-08-02-08-37-48.csv",
            ],
            "src_dir": [
                "/media/asad/ADAS_CV/datasets_Vegs/pepper/sc_v1_v2_patches",
                "/media/asad/ADAS_CV/datasets_Vegs/pepper/sc_v1_v2_patches",
            ],
        },
        2: {
            "labels": [
                "/media/asad/ADAS_CV/datasets_Vegs/pepper/pepper_points/labels_my-project-name_2021-08-05-12-17-20.csv",
                "/media/asad/ADAS_CV/datasets_Vegs/pepper/pepper_points/scalecam_30_10_2021.csv"
                ],
            "src_dir": [
                "/media/asad/ADAS_CV/scalecam_pepper_selected",
                "/media/asad/8800F79D00F79104/aws_bayer_Data/VIP_pepper_filter/Scalecam_validate"
            ],
        },
    }
    # output patch directory
    # patch_out = "/media/asad/ADAS_CV/datasets_Vegs/pepper/Refactoringcode/patches"
    patch_out = (
        "/media/asad/8800F79D00F79104/aws_bayer_Data/VIP_pepper_filter/scalecm_validate_patches"
    )
    # weight directory
    weights = "/media/asad/ADAS_CV/pepp/06-8-02/model_final.pth"
    #weights = "/media/asad/ADAS_CV/hot-pepp/20-8/model_final.pth"

    points_out = {}
    patch_labels = files[1]["labels"] if "labels" in files[1] else []
    patch_srcs = files[1]["src_dir"] if "src_dir" in files[1] else []
    for i, patchlabel in enumerate(patch_labels):
        print(f"Getting Points of Patch label file {patchlabel}")
        pl = getLabelspatch(patchlabel)
        temppoints = pl()
        points_out = {**points_out, **temppoints}
    frame_labels = files[2]["labels"] if "labels" in files[2] else []
    frame_srcs = files[2]["src_dir"] if "src_dir" in files[2] else []
    #pass list of of name sin labels. Ideally should be same but is different unfortuantely
    #names_labels=[{"top":"head","bottom":"tail"},{"top":"head","bottom":"tail"},{"top":"top","bottom":"bottom"}]
    names_labels=[{"top":"top","bottom":"bottom"},{"top":"top","bottom":"bottom"}]
    for i, frame_label in enumerate(frame_labels):
        print(f"Getting Points of Frame label file {frame_labels}")
        frame_imgs = frame_srcs[i]
        fl = getLabelsFrame(frame_imgs, frame_label, weights=weights, dst_dir=patch_out)
        temppoints = fl(len(points_out)+1,**names_labels[i])
        points_out = {**points_out, **temppoints}
    print(f"{len(points_out)} Patches are Annotated!!")
    # Saving the Points as pickel file to be used later
    a_file = open(out_pkl, "wb")
    pickle.dump(points_out, a_file)
    a_file.close()
    return patch_srcs, patch_out


if __name__ == "__main__":
    """First Build dataset then postprocess it and optionally visualize iot to verify everything"""
    build_dataset = False
    visualize_dataset = True
    post_process = True
    copy_patches=True
    # out_pkl = "sc1_sc2_points.pkl"
    out_pkl = "points_10_21.pkl"
    if build_dataset:
        patch_srcs, patch_out = build_points(out_pkl)
    else:
        a_file = open(out_pkl, "rb")
        points = pickle.load(a_file)
        # Hard code if the dataset is already built otherwise use build dataset output
        patch_out = ("/media/asad/8800F79D00F79104/aws_bayer_Data/VIP_pepper_filter/31-21-patches")
        if copy_patches:
            patch_srcs = [
                "/media/asad/ADAS_CV/datasets_Vegs/pepper/sc_v1_v2_patches",
                "/media/asad/8800F79D00F79104/aws_bayer_Data/VIP_pepper_filter/scalecm_validate_patches",
            ]
            Copyfiles(points, patch_srcs, patch_out)
        if post_process:
            remove_bad(points, patch_out)
        if visualize_dataset:
            for _ in range(1000):
                files = os.listdir(patch_out)
                imgname = np.random.choice(files)
                visualize_points(points, patch_out, imgname)
