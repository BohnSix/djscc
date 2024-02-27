import os
import cv2
from glob import glob
from tqdm import tqdm


#######################################################################################
imgs = sorted(glob("validation_imgs/validation_snr*"))[::10]
print(f"Have {len(imgs)} images here.")

video_name = 'validation.mp4'
video=cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 60, (512*2, 256*2))
for i in tqdm(imgs):
    image = cv2.imread(i)
    video.write(image)
video.release()
print(f"{os.path.getsize(video_name)/1024/1024:.3f} MB")
#######################################################################################

