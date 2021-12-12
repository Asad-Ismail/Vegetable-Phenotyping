from utils import *
from inference import *
from train import *


def train(trainMaskRCNN, trainPointNet):
    if trainMaskRCNN:
        mrcnn = MaskRCNN(
            "/media/asad/ADAS_CV/datasets_Vegs/pepper/annotated/scalecam1_2"
        )
        mrcnn.train()
    if trainPointNet:
        pn = train_pointnet(
            img_dir="/media/asad/ADAS_CV/datasets_Vegs/pepper/Refactoringcode/combined_patches",
            labels="sc1_sc2_points.pkl",
        )
        # pn.load_weights()
        pn.learn.freeze()
        pn.learn.lr_find()
        pn.learn.fit_flat_cos(20, 1e-4)
        pn.learn.show_results()
        pn.learn.unfreeze()
        pn.learn.lr_find()
        pn.learn.fit_flat_cos(50, 1e-5)
        pn.learn.saveModel("pn_res34")


def get_files(img_path):
    all_imgs = []
    fnames = []
    for file_index, filename in enumerate(os.listdir(img_path)):
        if filename.endswith(".json") or filename.endswith(".jpg.png"):
            continue
        f_p = os.path.join(img_path, filename)
        fnames.append(filename)
        all_imgs.append(f_p)
    return all_imgs, fnames


def combine_images(fnames, save_path, merge_path):
    for index in range(len(fnames)):
        img_det = cv2.imread(os.path.join(save_path, f"segmented_{fnames[index]}"))
        img_crop = cv2.imread(os.path.join(save_path, f"Cropped_{fnames[index]}"))
        fig = plt.figure(figsize=(40, 5))
        plt.title("Combined", loc="center")
        nrows = 1
        ncols = 4
        ax = fig.add_subplot(nrows, ncols, 1)
        ax.imshow(img_det[..., ::-1])
        if img_crop is not None:
            # Resize every image to croped concatenated Image
            h, w, _ = img_crop.shape
            dims = (w, h)
            # img_det = image_resize(img_det, height=h)
            ax = fig.add_subplot(nrows, ncols, 2)
            ax.imshow(img_crop[..., ::-1])
            img_points = cv2.imread(os.path.join(save_path, f"Points_{fnames[index]}"))
            ax = fig.add_subplot(nrows, ncols, 3)
            ax.imshow(img_points[..., ::-1])
            # img_points = cv2.resize(img_points, dims)
            img_corrected = cv2.imread(
                os.path.join(save_path, f"corrected_{fnames[index]}")
            )
            # img_corrected = cv2.resize(img_corrected, dims)
            ax = fig.add_subplot(nrows, ncols, 4)
            ax.imshow(img_corrected[..., ::-1])
        fig.savefig(os.path.join(merge_path, f"{fnames[index]}"))
        # joined_img = np.concatenate(
        #    [img_det, img_crop, img_points, img_corrected], axis=1
        # )
        # cv2.imwrite(os.path.join(merge_path, f"{fnames[index]}"), joined_img)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def run_inferences(veg, learn, fnames, all_imgs, save_path):
    """Run Detection and Point Network and save the indiviual scans"""
    for index in range(len(fnames)):
        img = cv2.imread(fnames[index])
        if img is not None:
            detected_cucumber, all_masks, all_patches, boxes, *_ = veg.pred(img)
        else:
            continue
        print(f"Processing file {index}")
        cv2.imwrite(
            os.path.join(save_path, f"segmented_{all_imgs[index]}"), detected_cucumber
        )
        cropedPatches = []
        for i, patch in enumerate(all_patches):
            x, y, w, h = cv2.boundingRect(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
            newImg = patch[y : y + h, x : x + w]
            cropedPatches.append(newImg)
        vis_crop_img = None
        # Patch size for visualization
        RESIZE = 224
        # Crop Detected Images
        for i, patch in enumerate(cropedPatches):
            if vis_crop_img is None:
                vis_crop_img = cv2.resize(patch, (RESIZE, RESIZE))
            else:
                vis_crop_img = np.concatenate(
                    [vis_crop_img, cv2.resize(patch, (RESIZE, RESIZE))], axis=1
                )
        if vis_crop_img is not None:
            cv2.imwrite(
                os.path.join(save_path, f"Cropped_{all_imgs[index]}"), vis_crop_img
            )
        # Pred Points on cropped Images
        predPoints = {}
        for i, crop in enumerate(cropedPatches):
            img = PILImage.create(crop[..., ::-1])
            pred = learn.predict(img)
            preds = scale_predictions(pred, img.shape)
            predPoints[i] = [
                tuple(preds[0].round().long().numpy()),
                tuple(preds[1].round().long().numpy()),
            ]
        vis_point_img = check_points(cropedPatches, predPoints)
        if vis_point_img is not None:
            cv2.imwrite(
                os.path.join(save_path, f"Points_{all_imgs[index]}"), vis_point_img
            )
        # Correct Cropped Images
        corrected_imgs = apply_rotation(cropedPatches, predPoints)
        vis_corrected_img = None
        for i, correct_img in enumerate(corrected_imgs):
            if vis_corrected_img is None:
                vis_corrected_img = cv2.resize(correct_img, (RESIZE, RESIZE))
            else:
                vis_corrected_img = np.concatenate(
                    [vis_corrected_img, cv2.resize(correct_img, (RESIZE, RESIZE))],
                    axis=1,
                )
        if vis_corrected_img is not None:
            cv2.imwrite(
                os.path.join(save_path, f"corrected_{all_imgs[index]}"),
                vis_corrected_img,
            )


def infer():
    maskrcnn = "/media/asad/ADAS_CV/pepp/06-8-02/model_final.pth"
    model = "/home/asad/projs/rotation_correction/sc_v1_v2.pkl"
    img_path = "/media/asad/ADAS_CV/datasets_Vegs/pepper/sc_v1_v2_complete_images"
    save_path = "/media/asad/ADAS_CV/vegs_results/pepper/sc_v1_v2_orig/indiviual_scans"
    merge_path = "/media/asad/ADAS_CV/vegs_results/pepper/sc_v1_v2_orig/merge_scans"

    learn = fast_AILearner(model)
    veg = detectroninference(maskrcnn)
    # file names and full paths given
    all_imgs, fnames = get_files(img_path=img_path)
    # run_inferences(veg, learn, all_imgs, fnames, save_path=save_path)
    combine_images(fnames, save_path=save_path, merge_path=merge_path)


if __name__ == "__main__":

    is_train = True
    train_MaskRCNN = True
    train_pointNet = False
    inference = True
    if is_train:
        train(train_MaskRCNN, train_pointNet)
    if inference:
        infer()
