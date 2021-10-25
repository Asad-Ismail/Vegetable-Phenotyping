import shutil
import os
import cv2
from tqdm import tqdm


src = "/media/asad/8800F79D00F79104/aws_bayer_Data/VIP_pepper"

maxheight=0
maxwidth=0

for subdir, dirs, files in tqdm(os.walk(src)):
    for file in files:
        is_skipped = False
        img=cv2.imread((os.path.join(subdir, file)),-1)
        #Skip depth channel image
        h,w=img.shape[:-1]
        if h>maxheight:
            maxheight=h
        if w>maxwidth:
            maxwidth=w           

print(f"Max height and width of Image {maxheight}, {maxwidth}")
