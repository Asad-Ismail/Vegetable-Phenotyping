import sys
sys.path.append("../scripts")
from utils import *
from inference import *
from train import *
import yaml
import argparse
import logging

logging.basicConfig(level=logging.DEBUG)
parser=argparse.ArgumentParser()
parser.add_argument("--config_file",type=str,default="configs/run_config.yaml")
args=parser.parse_args()

with open(args.config_file, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

def train(trainMaskRCNN, trainPointNet):
    os.makedirs(config["output_dir"],exist_ok=True)
    if trainMaskRCNN:
        logging.info("*"*50)
        logging.info(f"Training MaskRCNN!!")
        logging.info("*"*50)
        mrcnn = MaskRCNN(
            config["detection_data"],
            out_dir=config["output_dir"]
        )
        mrcnn.train()
    if trainPointNet:
        logging.info("*"*50)
        logging.info(f"Training PointNet!!")
        logging.info("*"*50)
        pn = train_pointnet(
            img_dir=config["points_images"],
            labels=config["points_annotation"],
        )
        # pn.load_weights()
        pn.learn.freeze()
        #pn.learn.lr_find()
        pn.learn.fit_flat_cos(20, 1e-4)
        pn.learn.show_results()
        pn.learn.unfreeze()
        pn.learn.fit_flat_cos(50, 1e-5)
        pn.learn.export(os.path.join(os.path.join(config["output_dir"],"pn_res34")))



def run_inference(veg, learn, fnames, all_imgs, save_path):
    """Run Detection and Point Network and save the indiviual scans"""
    for index in range(len(fnames)):
        img = cv2.imread(fnames[index])
        if img is not None:
            detected_cucumber, all_masks, all_patches, boxes, *_ = veg.pred(img)
        else:
            continue
        logging.info(f"Processing file {index}")
        cv2.imwrite(os.path.join(save_path, f"segmented_{all_imgs[index]}"), detected_cucumber)
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
    logging.info(f"Loading Point Model!!")
    learn = fast_AILearner(os.path.join(config["output_dir"],config["model"]))
    logging.info(f"Loading Detection Model!!")
    veg = detectroninference(os.path.join(config["output_dir"],config["maskrcnn"]))
    # file names and full paths given
    all_imgs, fnames = get_files(img_path=config["img_path"])
    run_inferences(veg, learn, all_imgs, fnames, save_path=config["save_path"])
    combine_images(fnames, save_path=config["save_path"], merge_path=config["merge_path"])


if __name__ == "__main__":

    is_train = config["train"]
    train_MaskRCNN =config["train_maskrcnn"] 
    train_pointNet = config["train_resnet34"]
    inference = not is_train
    if is_train:
        logging.info("Starting Training!!")
        train(train_MaskRCNN, train_pointNet)
    if inference:
        logging.info("Starting Inference!!")
        infer()
