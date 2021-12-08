import shutil
import os
import cv2
from tqdm import tqdm


Scale_Cam=True
write_new=True

#src = "/media/asad/8800F79D00F79104/aws_bayer_Data/VIP_pepper_filter/VIP_pepper"
src="/media/asad/8800F79D00F79104/pepper_2021/aws_data"
dst="/media/asad/8800F79D00F79104/pepper_2021/aws_filtered_data"
skip_names = ["100.jpg", "320.jpg", "608.jpg", "416.jpg", "608__4chan"]
index = 0

total_remaining_files = 0
removed_files = 0

for subdir, dirs, files in tqdm(os.walk(src)):
    for file in files:
        is_skipped = False
        img=cv2.imread((os.path.join(subdir, file)),-1)
        if img is None:
            print(f"{file} is not an image")
            continue
        #Skip depth channel image
        if img.shape[2]==4 and not Scale_Cam:
            is_skipped=True
        for name in skip_names:
            if name in file:
                is_skipped = True
        if is_skipped:
            removed_files += 1
            #os.remove(os.path.join(subdir, file))
        else:
            total_remaining_files += 1
            if write_new:
                cv2.imwrite(os.path.join(dst,file),img[...,:3])

print(f"Removed files {removed_files}")
print(f"Remaining files {total_remaining_files}")
