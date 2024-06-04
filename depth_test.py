import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from PIL import Image
from torchvision.transforms import Compose
from AdaBins.infer import InferenceHelper
depth_anything_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "Depth_Anything"))
sys.path.append(depth_anything_dir)
from Depth_Anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from Depth_Anything.depth_anything.dpt import DepthAnything
import yaml
from tqdm import tqdm
import imageio


class DepthGT:
    def __init__(self, folder_path: str, show_depth_map: str) -> None:
        self.folder_path = folder_path
        self.depth_maps = sorted(os.listdir(folder_path))
        self.show_depth_map = show_depth_map

    def create_depth_map(self, map_index: int, **kwargs) -> np.ndarray:
        depth_map = imageio.v2.imread(os.path.join(self.folder_path, self.depth_maps[map_index]))/100

        return depth_map
    

class DepthCalculator:
    """Depth calculator in the image."""

    def __init__(self, config):
        if config['input_media']['benchmark_type'] == "brazilian_road":
            self.real_sensor_resolution_w = 3280
            self.img_width_pixel = 1920
            self.resolution_ratio_w = self.real_sensor_resolution_w / self.img_width_pixel
            self.focal_length_mm = 3.04
            self.sensor_width = 3.68
            self.focal_length_pixel_x = ((self.focal_length_mm/self.sensor_width)
                                        * self.img_width_pixel) / self.resolution_ratio_w  # 928.44
            self.focal_length_pixel_y = self.focal_length_pixel_x

            self.min_distance = config['depth_estimation']['environment_dimensions_brazilian_road']['min_distance']
            self.max_distance = config['depth_estimation']['environment_dimensions_brazilian_road']['max_distance']
            self.ref_point_1 = np.array(config['depth_estimation']['environment_dimensions_brazilian_road']['ref_point_1'])
            self.ref_point_2 = np.array(config['depth_estimation']['environment_dimensions_brazilian_road']['ref_point_2'])
            self.ref_distance_irl = config['depth_estimation']['environment_dimensions_brazilian_road']['ref_distance_irl']
        
        if config['input_media']['benchmark_type'] == "ue5":
            self.focal_length_mm = config['depth_estimation']['camera_parameters_ue5']['focal_length_mm']
            self.sensor_width = config['depth_estimation']['camera_parameters_ue5']['sensor_width']
            self.img_width_pixel = config['depth_estimation']['camera_parameters_ue5']['img_width_pixel']
            self.focal_length_pixel_x = (self.focal_length_mm/self.sensor_width) * self.img_width_pixel 
            
            self.sensor_height = config['depth_estimation']['camera_parameters_ue5']['sensor_height']
            self.img_height_pixel = config['depth_estimation']['camera_parameters_ue5']['img_height_pixel']
            self.focal_length_pixel_y = (self.focal_length_mm/self.sensor_height) * self.img_height_pixel

            self.min_distance = config['depth_estimation']['environment_dimensions_ue5']['min_distance']
            self.max_distance = config['depth_estimation']['environment_dimensions_ue5']['max_distance']
            self.ref_point_1 = np.array(config['depth_estimation']['environment_dimensions_ue5']['ref_point_1'])
            self.ref_point_2 = np.array(config['depth_estimation']['environment_dimensions_ue5']['ref_point_2'])
            self.ref_distance_irl = config['depth_estimation']['environment_dimensions_ue5']['ref_distance_irl']

        self.correction_mode = config['depth_estimation']['correction_mode']
        self.show_correction = config['depth_estimation']['show_correction']
        self.lp_depth_calculation_mode = config['depth_estimation']['lp_depth_calculation_mode']
        self.scaling_ratios = np.array([], dtype=np.float16)
        self.avg_running_scaling_ratio = 0
        self.avg_scaling_ratio = 0
        self.threshold = 0
        self.iter = 1


    def get_license_plate_depth(self, cropped_lp_depth_map: np.ndarray):
        """License plate depth calculation with median or average value.

        Args:
            cropped_lp_depth_map (np.ndarray): Depth map cropped for the license plate area.
            mode (str): Calculation mode: 'median' or 'average'.

        Returns:
            lp_depth(np.float16): The depth of the license plate from the camera in meters.
        """
        if self.lp_depth_calculation_mode == "average":
            return np.float16(np.average(cropped_lp_depth_map))
        elif self.lp_depth_calculation_mode == "median":
            return np.median(cropped_lp_depth_map)
        else:
            raise(NotImplementedError("mode can only be 'average' or 'median' but got {}".format(self.lp_depth_calculation_mode)))


    def get_3D_coordinates(self, depth_map: np.ndarray, u: int, v: int, z: int):
        """Get the 3D coordinates of an object on the image given its depth map.

        Args:
            depth_map (np.ndarray): Depth map of the image.
            u (int): Objects horizontal coordinate on the image plane.
            v (int): Objects vertical coordinate on the image plane.
            z (int): Objects depth from the depth map.

        Returns:
            np.array([x,y,z]): 3D reconstructed coordinates.
        """
        px = depth_map.shape[1]/2
        py = depth_map.shape[0]/2
        x = (u*z - px*z) / self.focal_length_pixel_x
        y = (v*z - py*z) / self.focal_length_pixel_y

        return np.array([x, y, z], dtype = np.float16)

    def correct_depth_map(self, depth_map: np.ndarray):
        """Depth map correction.

        Args:
            depth_map (np.ndarray): Depth map to correct.

        Returns:
            depth_map_modified(np.ndarray): Corrected depth map.
        """
        # Rejection of lower and upper 1%
        lower_threshold = np.percentile(depth_map, 1)
        upper_threshold = np.percentile(depth_map, 99)

        # mask = (depth_map >= lower_threshold) & (depth_map <= upper_threshold)
        # depth_map[~mask] = np.nan

        # depth_map[depth_map < lower_threshold] = lower_threshold
        # depth_map[depth_map > upper_threshold] = upper_threshold

        if self.correction_mode == 'normalization':
            depth_map_modified = ((depth_map-np.min(depth_map))*(self.max_distance-self.min_distance))/(
                np.max(depth_map)-np.min(depth_map)) + self.min_distance

        elif self.correction_mode == 'scaling':
            scaling_ratio, _ = self.get_scaling_ratio(depth_map)
            depth_map_modified = depth_map*scaling_ratio

        elif self.correction_mode == 'None':
            depth_map_modified = depth_map

        else:
            raise ValueError(
                "method can be either 'scaling' or 'normalizing' but got {}".format(self.correction_mode))

        return depth_map_modified

    def get_scaling_ratio(self, depth_map: np.ndarray):
        """Scaling ratio calculator.

        Args:
            depth_map (np.ndarray): Input depth map.

        Returns:
            list(np.float16, bool): Scaling ratio, whether to skip a frame because of incorrect scaling or not.
        """
        skip_frame = False
        px = depth_map.shape[1]/2
        py = depth_map.shape[0]/2

        u_1 = self.ref_point_1[0]
        v_1 = self.ref_point_1[1]
        z_1 = np.average(depth_map[v_1-5:v_1+5, u_1-5:u_1+5])
        x_1 = (u_1*z_1 - px*z_1) / self.focal_length_pixel_x
        y_1 = (v_1*z_1 - py*z_1) / self.focal_length_pixel_y
        ref_point_reconstructed_1 = np.array(
            [np.float16(x_1), np.float16(y_1), np.float16(z_1)])

        u_2 = self.ref_point_2[0]
        v_2 = self.ref_point_2[1]
        z_2 = np.average(depth_map[v_2-5:v_2+5, u_2-5:u_2+5])
        x_2 = (u_2*z_2 - px*z_2) / self.focal_length_pixel_x
        y_2 = (v_2*z_2 - py*z_2) / self.focal_length_pixel_y
        ref_point_reconstructed_2 = np.array(
            [np.float16(x_2), np.float16(y_2), np.float16(z_2)])

        # ref_point_reconstructed_distance = np.float16(
        #     np.sqrt(np.sum((ref_point_reconstructed_1-ref_point_reconstructed_2)**2)))
        ref_point_reconstructed_distance = np.float16(np.sqrt(np.sum((z_1-z_2)**2)))
        scaling_ratio = np.float16(
            self.ref_distance_irl/(ref_point_reconstructed_distance + 1e-16))

        self.scaling_ratios = np.append(self.scaling_ratios, scaling_ratio)
        if self.scaling_ratios.size % 40 == 0:
            self.avg_running_scaling_ratio = np.average(self.scaling_ratios)
            print("AVG RUNNING RATIO CHANGED", self.avg_running_scaling_ratio)
            if self.avg_scaling_ratio == 0:
                self.avg_scaling_ratio = self.avg_running_scaling_ratio
                print("AVG RATIO CHANGED", self.avg_scaling_ratio)
            else:
                self.avg_scaling_ratio = np.average(
                    [self.avg_running_scaling_ratio, self.avg_scaling_ratio])
                print("AVG RATIO CHANGED", self.avg_scaling_ratio)
            self.threshold = self.avg_scaling_ratio*0.5

        if self.avg_scaling_ratio != 0:
            if not (self.scaling_ratios[-1] >= self.avg_scaling_ratio - self.threshold) & (self.scaling_ratios[-1] <= self.avg_scaling_ratio + self.threshold):
                self.scaling_ratios = self.scaling_ratios[:-1]
                skip_frame = True
                print(scaling_ratio)

        self.iter += 1

        return scaling_ratio, skip_frame


class DepthEstimatorZoeDepth:
    """Depth map estimation using ZoeDepth"""

    def __init__(self, depth_calculator: DepthCalculator, config):
        """

        Args:
            depth_calculator (DepthCalculator): depth calculator object
        """
        # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
        repo = "isl-org/ZoeDepth"
        model = config['depth_estimation']['zoedepth']['model']
        model_zoe = torch.hub.load(repo, model, pretrained=True)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using", DEVICE)
        self.zoe = model_zoe.to(DEVICE)
        self.depth_calculator = depth_calculator
        self.show_depth_map = config['depth_estimation']['show_depth_map']

    def create_depth_map(self, input_frame: np.ndarray, **kwargs):
        """Estimate a depth map using ZoeDepth.

        Args:
            input_frame (np.ndarray): Input RGB frame to estimate depth map from.

        Returns:
            depth_map(np.ndarray): Estimated depth map.
        """
        img = Image.fromarray(input_frame)
        input_img = img.convert("RGB")
        depth_map = self.zoe.infer_pil(
            input_img, pad_input=False, with_flip_aug=False)
        
        if self.show_depth_map:
            im = plt.imshow(depth_map, cmap= 'magma')
            cbar = plt.colorbar(im, orientation='horizontal', pad=0.05, shrink=0.7)
            plt.show()
        
        return depth_map


class DepthEstimatorAdabins:
    """Depth map estimation using AdaBins"""

    def __init__(self, depth_calculator: DepthCalculator, config):
        """

        Args:
            depth_calculator (DepthCalculator): depth calculator object
        """
        self.infer_helper = InferenceHelper(dataset=config['depth_estimation']['adabins']['inference_helper_dataset'])
        self.depth_calculator = depth_calculator
        self.show_depth_map = config['depth_estimation']['show_depth_map']

    def create_depth_map(self, input_frame: np.ndarray, **kwargs):
        """Estimate a depth map using AdaBins.

        Args:
            input_frame (np.ndarray): Input RGB frame to estimate depth map from.

        Returns:
            depth_map(np.array): Estimated depth map.
        """
        img = Image.fromarray(input_frame)
        input_img = img.resize((640, 480)).convert("RGB")
        bin_centers, depth_map = self.infer_helper.predict_pil(input_img)
        depth_map = depth_map[0, 0, :, :]
        depth_map_resized = Image.fromarray(depth_map).resize((input_frame.shape[1], input_frame.shape[0]))
        depth_map_array = np.array(depth_map_resized)

        if self.show_depth_map:
            im = plt.imshow(depth_map, cmap= 'magma')
            cbar = plt.colorbar(im, orientation='horizontal', pad=0.05, shrink=0.7)
            plt.show()

        return self.depth_calculator.correct_depth_map(depth_map_array)


class DepthEstimatorDepthAnything:
    """Depth map estimation using Depth Anything"""

    def __init__(self, depth_calculator: DepthCalculator, config):
        """

        Args:
            depth_calculator (DepthCalculator): depth calculator object
        """
        self.depth_calculator = depth_calculator
        self.encoder = config['depth_estimation']['depth_anything']['encoder']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.chdir(os.path.join(os.getcwd(), "Depth_Anything"))
        self.model = DepthAnything.from_pretrained(
            'LiheYoung/depth_anything_{:}14'.format(self.encoder)).to(device=self.device).eval()
        self.transform = Compose([Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,),
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            PrepareForNet(),])
        os.chdir(os.path.join(os.getcwd(), ".."))
        self.show_depth_map = config['depth_estimation']['show_depth_map']

    def create_depth_map(self, input_frame: np.ndarray, **kwargs):
        """Estimate a depth map using Depth Anything.

        Args:
            input_frame (np.ndarray): Input RGB frame to estimate depth map from.

        Returns:
            depth_map(np.array): Estimated depth map.
        """
        input_img = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB) / 255.0
        input_img = self.transform({'image': input_img})['image']
        input_img = torch.from_numpy(input_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            disparity = self.model(input_img)
        disparity = torch.nn.functional.interpolate(disparity[None], (input_frame.shape[0], input_frame.shape[1]), mode='bilinear', align_corners=False)[0, 0]

        depth = (disparity - disparity.min()) / (disparity.max() - disparity.min())
        depth = 1 - depth
        depth_map_array = depth.cpu().numpy()

        if self.show_depth_map:
            im = plt.imshow(depth_map_array, cmap= 'magma')
            cbar = plt.colorbar(im, orientation='horizontal', pad=0.05, shrink=0.7)
            plt.show()

        return self.depth_calculator.correct_depth_map(depth_map_array)


def main(config_path = ".\config\config.yaml"):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    depth_calculator = DepthCalculator(config=config)

    if config['depth_estimation']['model'] == "zoedepth":
        depth_estimator = DepthEstimatorZoeDepth(
            depth_calculator, config)
    elif config['depth_estimation']['model'] == "adabins":
        depth_estimator = DepthEstimatorAdabins(depth_calculator, config)
    elif config['depth_estimation']['model'] == "depth_anything":
        depth_estimator = DepthEstimatorDepthAnything(
            depth_calculator, config)
    
    depth_gt = DepthGT(config['input_media']['depth_labels_path'], config['depth_estimation']['show_depth_map'])

    file_names = sorted(os.listdir(config['input_media']['media_path']))
    deltas1 = np.array([], dtype=np.float16)
    deltas2 = np.array([], dtype=np.float16)
    deltas3 = np.array([], dtype=np.float16)
    absrels = np.array([], dtype=np.float16)

    # zoe = DepthEstimatorZoeDepth(depth_calculator, config)
    # ada = DepthEstimatorAdabins(depth_calculator, config)
    # da = DepthEstimatorDepthAnything(depth_calculator, config)

    last = None
    for idx, image in enumerate(tqdm(file_names)):
        if idx % 1 == 0:
            image_path = os.path.join(config['input_media']['media_path'], image)
            frame = cv2.imread(image_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth_map = depth_estimator.create_depth_map(frame)
            gt_depth_map = depth_gt.create_depth_map(idx)

            # Delta
            delta = np.maximum(gt_depth_map /depth_map, depth_map / gt_depth_map)
            delta_1 = np.mean(delta < 1.25)
            delta_2 = np.mean(delta < 1.25**2)
            delta_3 = np.mean(delta < 1.25**3)

            deltas1 = np.append(deltas1, delta_1)
            deltas2 = np.append(deltas2, delta_2)
            deltas3 = np.append(deltas3, delta_3)

            # AbsRel
            abs_rel = np.mean(np.abs(gt_depth_map - depth_map) / gt_depth_map)
            absrels = np.append(absrels, abs_rel)


            # zoedepth = zoe.create_depth_map(frame)
            # adadepth = ada.create_depth_map(frame)
            # dadepth = da.create_depth_map(frame)

            # # Predictions normalized
            # fig, axes = plt.subplots(2, 2, figsize=(20, 8))

            # # Flatten the 2x2 array of axes to iterate over it
            # for ax in axes.flat:
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            #     ax.set_xticklabels([])
            #     ax.set_yticklabels([])

            # # Plot on each axis using the correct indexing (row, column)
            # im0 = axes[0, 0].imshow(gt_depth_map, cmap='magma')
            # axes[0, 0].set_title("GT", fontsize=28)

            # im1 = axes[0, 1].imshow(adadepth, cmap='magma')
            # axes[0, 1].set_title("AdaBins", fontsize=28)

            # im2 = axes[1, 0].imshow(zoedepth, cmap='magma')
            # axes[1, 0].set_title("ZoeDepth", fontsize=28)

            # im3 = axes[1, 1].imshow(dadepth, cmap='magma')
            # axes[1, 1].set_title("Depth Anything", fontsize=28)

            # # Add a single colorbar to the right of the subplots
            # cbar = fig.colorbar(im3, ax=axes, orientation='vertical', fraction=0.05, pad=0.04)
            # cbar.set_label('Depth [m]', fontsize=24)
            # cbar.ax.tick_params(labelsize=24)

            # plt.show()

            
            # # Delta bool maps
            # adadelta = np.maximum(gt_depth_map /adadepth, adadepth / gt_depth_map)
            # adadelta_1 = adadelta < 1.25
            # adadelta_2 = adadelta < 1.25**2
            # adadelta_3 = adadelta < 1.25**3
            # adaabs_rel = np.abs(gt_depth_map - adadepth) / gt_depth_map


            # zoedelta = np.maximum(gt_depth_map /zoedepth, zoedepth / gt_depth_map)
            # zoedelta_1 = zoedelta < 1.25
            # zoedelta_2 = zoedelta < 1.25**2
            # zoedelta_3 = zoedelta < 1.25**3
            # zoeabs_rel = np.abs(gt_depth_map - zoedepth) / gt_depth_map


            # dadelta = np.maximum(gt_depth_map /dadepth, dadepth / gt_depth_map)
            # dadelta_1 = dadelta < 1.25
            # dadelta_2 = dadelta < 1.25**2
            # dadelta_3 = dadelta < 1.25**3
            # daabs_rel = np.abs(gt_depth_map - dadepth) / gt_depth_map


            # fig, axes = plt.subplots(2, 2, figsize=(20, 8))

            # for ax in axes.flat:
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            #     ax.set_xticklabels([])
            #     ax.set_yticklabels([])

            # im0 = axes[0, 0].imshow(frame)
            # axes[0, 0].set_title("Input", fontsize=28)

            # im1 = axes[0, 1].imshow(adaabs_rel, cmap='magma')
            # axes[0, 1].set_title(f"AdaBins - {np.mean(adaabs_rel):,.2f}", fontsize=28)
            # cbar1 = plt.colorbar(im1, ax=axes[0,1], orientation='vertical', pad=0.05, shrink=1)
            # cbar1.set_label('AbsRel [m]', fontsize=28)
            # cbar1.ax.tick_params(labelsize=28)

            # im2 = axes[1, 0].imshow(zoeabs_rel, cmap='magma')
            # axes[1, 0].set_title(f"ZoeDepth - {np.mean(zoeabs_rel):,.2f}", fontsize=28)
            # cbar2 = plt.colorbar(im2, ax=axes[1,0], orientation='vertical', pad=0.05, shrink=1)
            # cbar2.set_label('AbsRel [m]', fontsize=28)
            # cbar2.ax.tick_params(labelsize=28)

            # im3 = axes[1, 1].imshow(daabs_rel, cmap='magma')
            # axes[1, 1].set_title(f"Depth Anything - {np.mean(daabs_rel):,.2f}", fontsize=28)
            # cbar3 = plt.colorbar(im3, ax=axes[1,1], orientation='vertical', pad=0.05, shrink=1)
            # cbar3.set_label('AbsRel [m]', fontsize=28)
            # cbar3.ax.tick_params(labelsize=28)

            # plt.show()

             # Correction
    #         if config['depth_estimation']['show_correction']:
    #             fig, (ax1, ax2) = plt.subplots(1, 3, figsize=(20, 8))
    #             for ax in [ax1, ax2]:
    #                 ax.set_xticks([])
    #                 ax.set_yticks([])
    #                 ax.set_xticklabels([])
    #                 ax.set_yticklabels([])
    #             im1 = ax1.imshow(depth_map, cmap='magma')
    #             ax1.set_title("Predicted")
    #             cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7)
    #             cbar1.set_label('Depth [m]', fontsize=28)
    #             cbar1.ax.tick_params(labelsize=28)
    #             im2 = ax2.imshow(gt_depth_map, cmap='magma')
    #             ax2.set_title("GT")
    #             cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.7)
    #             cbar2.set_label('Depth [m]', fontsize=28)
    #             cbar2.ax.tick_params(labelsize=28)
    #             plt.show()

            # Last
            if last is not None:
                print(1-abs(np.mean(depth_map)-np.mean(last)))
            if last is not None and (1-abs(np.mean(depth_map)-np.mean(last))) < 0.2:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                for ax in [ax1, ax2]:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                im1 = ax1.imshow(last, cmap='magma')
                ax1.set_title("Output at image index-1")
                cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal', pad=0.05, shrink=0.7)
                cbar1.set_label('Depth [m]', fontsize=28)
                cbar1.ax.tick_params(labelsize=28)
                im2 = ax2.imshow(depth_map, cmap='magma')
                ax2.set_title("Output at image index")
                cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal', pad=0.05, shrink=0.7)
                cbar2.set_label('Depth [m]', fontsize=28)
                cbar2.ax.tick_params(labelsize=28)
                plt.show()
            last = depth_map





    print("Model:", config['depth_estimation']['model'], "\nSet:", config['input_media']['media_path'].split("\\")[-2], "\nAbsRel (LTB):", np.mean(absrels), "\nDelta_1 (HTB):", np.mean(deltas1), "\nDelta_2 (HTB):", np.mean(deltas2), "\nDelta_3 (HTB):", np.mean(deltas3))

if __name__ == "__main__":
    main()