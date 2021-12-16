# Detection, Orientation correction and phenotpe detection of peppers
Detection and orientation correction are based on seperate networks. The detection is based on MaskRCNN and orientation correction is based on Restnet 34. Eventually we merged the two into one and used one network for both detection and orientation correction


### Stucture
    .
    ├── preprate_data           # Prepare dataset i.e build point dataset from two types of sources from patches and from the whole images. To get the points from complete images we use the MaskRCNN to detect the fruit first then find the points in the annotation closest to this detection using minimum eucledian distance
    ├── train                   # Training MaskRCNN and resnet34 for point detection
    ├── inference               # Inference of the MaskRCNN and resnet34 model
    ├── utils                   # Utility functions
    ├── main                    # Main to run train or inference
    ├── data_config             # Data config for preparing the dataset for point detection. Detection/Segmentation network doed not need special format conversion
    ├── model_config            # Model Config for training and inference
    .
    

### RUN
Specify the config file with data paths
Run prepare_data to prepare the data for resnet34 training with points for orientation correction
Run main with train to train both maskrcnn and renset34. Specify the conffig in model_config.yaml
Run main with inference to infer on maskrcnn and resnet34. 
