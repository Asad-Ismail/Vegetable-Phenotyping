import sys
sys.path.append("../scripts")
from pred_utils import *
from utils import get_image_files
import pandas as pd
from tqdm import tqdm
import yaml
import argparse
import logging
import cv2
import os

logging.basicConfig(level=logging.INFO)
parser=argparse.ArgumentParser()

parser.add_argument("--config_file",type=str,default="configs/phenotype_config.yaml")
args=parser.parse_args()
with open(args.config_file, "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

phen_names=['Area', 'Perimeter', 'Mid_Width', 'Max_Width', 'Mid_Height', 'Max_Height', 'Curved_Height', 'Maxheight_to_maxwidth', 'Midheight_to_midwidth', 'Curveheight_to_midwidth', 'Proximal_blockiness', 'Distall_blockiness', 'Fruit_triangle', 'Ellipse_Error', 'Box_Aspect']
 
def infer():
    logging.info("Loading Pepper Detection model!!")
    pepp=detectroninference(config["maskrcnn_weights"])
    logging.info("Loading Pepper Points Detection model!!")
    learn=fast_AILearner(config["pointnet_weights"])
    # get full files path and filenames
    f_paths,fnames=get_image_files(config["img_path"])
    if config["disx"] is None or config["disy"] is None:
        ppx,ppy=None,None
    else: 
        ppx,ppy=config["disx"]/config["pixx"],config["disy"]/config["pixy"]
    logging.info(f"Using ppx and ppy as {ppx,ppy}")
    startsec=time.time()
    results={}
    globalIndex=0
    empty_images=0
    plot_names=[]
    vis_results=config["visualize"]
    save_path=config["save_path"]
    logging.info(f"Save directory is {save_path}")
    os.makedirs(save_path,exist_ok=True)
    for index in tqdm(range(len(f_paths)),total=len(f_paths)):    
        img=cv2.imread(f_paths[index])
        (detected,
        corrected_imgs,
        corrected_masks,
        cropped_boxes,
        cropedPatches,
        predPoints,boxes) = run_inference(img,pepp,learn)
        logging.info(f"Processing file {fnames[index]}")
        #Append plot names to count plot
        plot_name=fnames[index].split("_")[0]
        if plot_name not in plot_names:
            plot_names.append(plot_name)
        if not len(corrected_masks):
            empty_images+=1
        for j, mask in enumerate(corrected_masks):
            globalIndex+=1
            assert len(mask.shape) == 2, "Should be Gray scale Mask"
            phen_res = phenotype_measurement_pepper(mask, corrected_imgs[j],skip_outliers=True,vis=vis_results,ppx=ppx,ppy=ppy)
            ## Debug color
            #print(f"Fruit colors rgb is {fruit_colors[0]}, LAB is {fruit_colors[1]}")
            if vis_results:
                fig=phen_res[-5]
                fig.savefig(os.path.join(save_path, f"Phenotype_{j}_{fnames[index]}"))
            if "Image" not in results: 
                results["Image"]=[fnames[index]]
                for l,v in enumerate(phen_names): 
                    results[v]=[phen_res[l]]
            else:
                results["Image"].append(fnames[index])
                for l,v in enumerate(phen_names): 
                    results[v].append(phen_res[l])
        if vis_results:
            save_results_cv(
            detected,
            cropedPatches,
            predPoints,
            corrected_imgs,
            fnames[index],
            save_path+"/inter")
    logging.info("Writing results in XLS format")
    results_df=pd.DataFrame.from_dict(results,orient='index').transpose()
    results_df.to_excel('results.xlsx')
    endsec=time.time()
    logging.info(f"Total Time spent is {(endsec-startsec)/60} minutes")


if __name__=="__main__":
    infer()