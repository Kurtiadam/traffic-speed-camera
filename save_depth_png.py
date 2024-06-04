import OpenEXR
import numpy as np
import os
from tqdm import tqdm
import cv2
import imageio

run_name = "run5_daylight_fog"
parent_dir = r"C:\Users\Adam\Documents\Unreal Projects\Gyorsitosav_sim\Saved\MovieRenders\\" + run_name
source_dir = r"C:\Users\Adam\Documents\Unreal Projects\Gyorsitosav_sim\Saved\MovieRenders\\" + run_name + "\\EXR"
files = os.listdir(source_dir)
target_dir = os.path.join(parent_dir, "depth_png")
os.makedirs(target_dir, exist_ok=True)

for idx, file in enumerate(tqdm(files)):
    exr = OpenEXR.InputFile(os.path.join(source_dir, file))
    depth_data = {'R': np.frombuffer(exr.channel('FinalImageMovieRenderQueue_WorldDepth.R'), dtype=np.float16)}
    # print(np.min(depth_data['R']), np.max(depth_data['R']))
    # reshaped_depth_data =  np.array(depth_data['R'].reshape((1080, 1920)))
    # print(np.max(reshaped_depth_data),np.min(reshaped_depth_data))

    reshaped_depth_data =  np.array(depth_data['R'].reshape((1080, 1920))).astype(np.uint16)
    # print(np.max(reshaped_depth_data2),np.min(reshaped_depth_data2), "\n")

    #cv2.imwrite(os.path.join(target_dir, "Depth" + "{:04d}".format(idx) + ".png"), reshaped_depth_data2)
    imageio.imwrite(os.path.join(target_dir, "Depth" + "{:04d}".format(idx) + ".png"), reshaped_depth_data)

    # saved_image = cv2.imread(os.path.join(target_dir, "Depth" + "{:04d}".format(idx) + ".png"), cv2.IMREAD_UNCHANGED)
    # saved_image = imageio.imread(os.path.join(target_dir, "Depth" + "{:04d}".format(idx) + ".png"))
    # print(saved_image.dtype)  # Should be uint16
    # print(saved_image.shape)  # Should be (1080, 1920)
    # print( np.max(saved_image),np.min(saved_image))