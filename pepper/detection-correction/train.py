from utils import *
from detectron2.engine import DefaultTrainer


class train_pointnet:
    """
    PointNet Class
    """

    def __init__(self, bs=32, img_dir="", labels=""):
        # Load labels for training
        "set up transforms and model for training"
        a_file = open(labels, "rb")
        self.points = pickle.load(a_file)
        self.img_dir = img_dir
        self.item_tfms = [Resize(224, method="squish", pad_mode="zeros")]
        self.batch_tfms = [
            Flip(pad_mode="zeros"),
            Rotate(pad_mode="zeros"),
            Warp(pad_mode="zeros"),
        ]
        getters = [
            ColReader("Images"),
            ColReader("labels"),
        ]
        dblock = DataBlock(
            blocks=(ImageBlock, PointBlock),
            splitter=RandomSplitter(valid_pct=0.01),
            getters=getters,
            item_tfms=self.item_tfms,
            batch_tfms=self.batch_tfms,
            n_inp=1,
        )
        df = self.build_data()
        self.dls = dblock.dataloaders(df, bs=bs)
        self.dls.c = self.dls.train.after_item.c
        self.learn = cnn_learner(
            self.dls,
            models.resnet34,
            lin_ftrs=[100],
            ps=0.01,
            concat_pool=True,
            loss_func=MSELossFlat(),
        )
        self.learn.summary()

    def get_labels(self, imgs):
        labels = []
        for img in imgs:
            f = img.name
            ps = self.points[f]
            im = PILImage.create(img)
            out = []
            # put out of bound in bound
            for x in ps:
                if x is None:
                    x = (-1, -1)
                if x[0] > im.size[0]:
                    x[0] = im.size[0] - 3
                if x[1] > im.size[1]:
                    x[1] = im.size[1] - 3
                out.append(x)
            out = np.array(out, dtype=np.float)
            labels.append(tensor(out))
        return labels

    def build_data(self):
        print(
            f"Building data frames for images and labels refactor if it gets too slow!!"
        )
        imgs = get_image_files(self.img_dir)
        df = pd.DataFrame({"Images": imgs, "labels": self.get_labels(imgs)})
        return df

    def load_learner(self, weights):
        "Load pretrain model"
        self.learn.load(weights)

    def load_weights(self, weights):
        "Load pretrain model"
        self.learn.load(weights)

    def saveModel(self, name):
        self.learn.save(name)
        self.learn.export(name)


class MaskRCNN:
    """
    MaskRCNN Detection Network
    """

    def __init__(
        self,
        data_dir,
        model_weights="/media/asad/ADAS_CV/pepp/06-8/model_0001999.pth",
        out_dir="/media/asad/8800F79D00F79104/aws_bayer_Data/VIP_pepper_filter/lib/model_weights/31-10-21-weights",
        veg="pep",
    ) -> None:
        
        for d in ["train", "valid"]:
            DatasetCatalog.register(
                veg + "_" + d, lambda d=d: self.get_veg_dicts(os.path.join(data_dir, d))
            )
            MetadataCatalog.get(veg + "_" + d).set(thing_classes=[veg])
        # self.veg_metadata = MetadataCatalog.get("pep_train")
        self.cfg = get_cfg()
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml"
            )
        )
        self.cfg.DATASETS.TRAIN = ("pep_train",)
        self.cfg.DATASETS.TEST = ("pep_valid",)
        self.cfg.DATASETS.VALID = ("pep_valid",)
        self.cfg.DATALOADER.NUM_WORKERS = 4
        self.cfg.MODEL.WEIGHTS = model_weights

        self.cfg["MODEL"]["ANCHOR_GENERATOR"]["ASPECT_RATIOS"][0] = [0.5, 1.0, 1.5]
        # cfg["INPUT"]["MASK_FORMAT"]='bitmask'
        self.cfg["INPUT"]["RANDOM_FLIP"] = "horizontal"
        # cfg["INPUT"]["RANDOM_FLIP"]="horizontal"
        self.cfg["INPUT"]["ROTATE"] = [-2.0, 2.0]
        self.cfg["INPUT"]["LIGHT_SCALE"] = 2
        self.cfg["INPUT"]["Brightness_SCALE"] = [0.5, 1.5]
        self.cfg["INPUT"]["Contrast_SCALE"] = [0.5, 2]
        self.cfg["INPUT"]["Saturation_SCALE"] = [0.5, 2]
        # cfg["BASE_LR"]=1e-5
        self.cfg.SOLVER.IMS_PER_BATCH = 1
        self.cfg.SOLVER.BASE_LR = 1e-3  # pick a good LR
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 500
        self.cfg.SOLVER.MAX_ITER = 5000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
            512  # faster, and good enough for this toy dataset (default: 512)
        )
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        self.cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
        self.cfg.MODEL.RETINANET.NUM_CLASSES = 1
        self.cfg.OUTPUT_DIR = out_dir
        self.extensions = [".jpg", ".JPEG", ".jpg", ".jpeg", ".png"]
        print(self.cfg)
        print(self.cfg.OUTPUT_DIR)

    def get_veg_dicts(self, img_dir):
        json_files = [
            json_file
            for json_file in os.listdir(img_dir)
            if json_file.endswith(".json")
        ]
        dataset_dicts = []
        for idx, json_file in enumerate(json_files):
            # if idx==100:
            #    return dataset_dicts
            for ext in self.extensions:
                filename = json_file.split(".")[0] + ext
                c_fname = os.path.join(img_dir, filename)
                img = cv2.imread(c_fname)
                if img is not None:
                    break
            if img is None:
                print(f"Image Not Found for {json_file}")
                # continue
                raise (f"Image Not Found for {json_file}")
            print(f"Processing json {json_file}")
            with open(os.path.join(img_dir, json_file)) as f:
                imgs_anns = json.load(f)

            record = {}
            height, width = img.shape[:2]

            record["file_name"] = c_fname
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
            # cv2.imshow("Image",img)
            # cv2.waitKey(0)
            annos = imgs_anns["shapes"]
            objs = []
            for anno in annos:
                # assert not anno["region_attributes"]
                px = [x for x, y in anno]
                py = [y for x, y in anno]
                poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                poly = [p for x in poly for p in x]

                obj = {
                    "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs
            print(f"Processed images {idx}")
            dataset_dicts.append(record)
        return dataset_dicts

    def train(self):
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
