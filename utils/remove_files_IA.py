import shutil
import os
import cv2
from tqdm import tqdm


src = "/media/asad/8800F79D00F79104/aws_bayer_Data/VIP_pepper_filter/VIP_pepper"
skip_names = ["100.jpg", "320.jpg", "608.jpg", "416.jpg", "608__4chan"]
index = 0

total_remaining_files = 0
removed_files = 0

for subdir, dirs, files in tqdm(os.walk(src)):
    for file in files:
        is_skipped = False
        img=cv2.imread((os.path.join(subdir, file)),-1)
        #Skip depth channel image
        if img.shape[2]==4:
            is_skipped=True
        for name in skip_names:
            if name in file:
                is_skipped = True
        if is_skipped:
            removed_files += 1
            os.remove(os.path.join(subdir, file))
        else:
            total_remaining_files += 1

print(f"Removed files {removed_files}")
print(f"Remaining files {total_remaining_files}")
