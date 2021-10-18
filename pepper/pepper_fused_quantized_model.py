# Author Asad Ismail
from torch.quantization.fuse_modules import fuse_modules
from utils import *
from phenotypes_utils import *
import torch.nn as nn


class LossEvalHook(HookBase):
    """Hook Class for Calculating Evaluation

    Args:
        HookBase ([type]): Hooks for training intermediate output
    """

    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        logger.logging.info(f"Mean Validation loss {mean_loss}")
        self.trainer.storage.put_scalar("validation_loss", mean_loss)
        comm.synchronize()
        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()


class VegFusedQuantizedTrainer(DefaultTrainer):
    """Create a trainer for veg datasets

    Args:
        DefaultTrainer ([type]): Default trainer from detectron2

    Returns:
        [type]: [description]
    """

    def __init__(self, cfg, fuse=True):
        super().__init__(cfg)
        # self.fuse_modules()
        for module in self.model.modules():
            print(module)

        # self.model = torch.quantization.fuse_modules(
        #    self.model,
        #    [
        #        [
        #            "roi_heads.keypoint_head.conv_fcn8",
        #            "roi_heads.keypoint_head.conv_fcn_relu8",
        #        ]
        #    ],
        #    inplace=True,
        # )
        print(self.model)

    def get_children(self, m: torch.nn.Module):
        # get children form model!
        children = dict(m.named_children())
        output = {}
        if children == {}:
            # if module has no children; m is last child! :O
            return m
        else:
            # look for children from children... to the last child!
            for name, child in children.items():
                try:
                    output[name] = self.get_children(child)
                except TypeError:
                    output[name] = self.get_children(child)
        return output

    def fuse_modules(self):

        # out = self.get_children(self.model)
        # print(out)
        weights = {}
        for name, param in self.model.named_parameters():
            if name.endswith(".bias"):
                continue
            if "conv_fcn8" in name:
                foundname = True
            key_name = f"Layer {name.split('.')[1]}"
            weights[key_name] = param.detach().view(-1).cpu().numpy()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(
            dataset_name,
            output_dir=cfg.OUTPUT_DIR,
            kpt_oks_sigmas=cfg.TEST.KEYPOINT_OKS_SIGMAS,
        )

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True)
                ),
            ),
        )
        return hooks


def get_config():
    """Get configuration/hyperparameters of detectron2

    Returns:
        [type]: config
    """
    cfg = get_cfg()
    # Dataset dicts
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("pep_train",)
    cfg.DATASETS.TEST = ("pep_valid",)
    print(f"f One Time calculation of Dataset Lengths!!!")
    dataset_dicts_train = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
    print(f"Length of dataset Training dataset is {len(dataset_dicts_train)}")
    dataset_dicts_test = DatasetCatalog.get(cfg.DATASETS.TEST[0])
    print(f"Length of dataset Testing dataset is {len(dataset_dicts_train)}")
    num_epochs = 500
    # Path of pretrained weights
    cfg.MODEL.WEIGHTS = "/media/asad/ADAS_CV/vegs_results/models/pepper/scalecam_v1_v2-10-08/model_final.pth"
    cfg["MODEL"]["ANCHOR_GENERATOR"]["ASPECT_RATIOS"][0] = [0.5, 1.0, 1.5]
    cfg["INPUT"]["RANDOM_FLIP"] = "horizontal"
    cfg["INPUT"]["ROTATE"] = [-2.0, 2.0]
    cfg["INPUT"]["LIGHT_SCALE"] = 2
    cfg["INPUT"]["Brightness_SCALE"] = [0.5, 1.5]
    cfg["INPUT"]["Contrast_SCALE"] = [0.5, 2]
    cfg["INPUT"]["Saturation_SCALE"] = [0.5, 2]
    cfg["MODEL"]["KEYPOINT_ON"] = True
    cfg.MODEL["ROI_KEYPOINT_HEAD"]["NUM_KEYPOINTS"] = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 1e-2  # pick a good LR
    cfg.SOLVER.CHECKPOINT_PERIOD = (len(dataset_dicts_train) * num_epochs) // 10
    cfg.SOLVER.MAX_ITER = len(dataset_dicts_train) * num_epochs  #
    cfg.SOLVER.STEPS = (2000, 3000)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1
    cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.1, 0.1]
    cfg.TEST.EVAL_PERIOD = (len(dataset_dicts_train) * num_epochs) // 5
    cfg.OUTPUT_DIR = "output"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    print(f" The output directory is {cfg.OUTPUT_DIR}")
    # save config to be used for inderence
    yml_file = cfg.dump()
    with open("train_config.yml", "w") as outfile:
        outfile.write(yml_file)
    return cfg


if __name__ == "__main__":
    kp_file = "/media/asad/adas_cv_2/test_kp.csv"
    img_dir = "/media/asad/ADAS_CV/datasets_Vegs/pepper/one_annotated/"
    gt_points = load_points_dataset(kp_file)
    for d in ["train", "valid"]:
        DatasetCatalog.register(
            "pep_" + d, lambda d=d: get_veg_dataset(img_dir + d, gt_points)
        )
        MetadataCatalog.get("pep_" + d).set(thing_classes=["pep"])
        MetadataCatalog.get("pep_" + d).set(keypoint_names=["head", "tail"])
        MetadataCatalog.get("pep_" + d).set(
            keypoint_flip_map=[("head", "head"), ("tail", "tail")]
        )
        MetadataCatalog.get("pep_" + d).set(
            keypoint_connection_rules=[("head", "tail", (0, 255, 255))]
        )
    cfg = get_config()
    trainer = VegFusedQuantizedTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
