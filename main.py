from ultralytics import YOLO
import cv2
import argparse
import numpy as np
import math
import time
from sort import *
import os
import sys
import torch
from torchvision import transforms
from OCR.config_utils import load_config
from OCR.ocr_model import create_network
from OCR.ocr_model_multirow import create_network as create_network_multi
from collections import Counter
import pandas as pd
from PIL import Image
from typing import Tuple, Dict
from AdaBins.infer import InferenceHelper
import xml.etree.ElementTree as ET
import matplotlib.path as mplPath


class TrafficSpeedCamera:
    """Class for running the speed camera algorithm"""

    def __init__(self, input_path: str, xml_input_path:str, input_mode: str, fps: float = 30, depth_estimator_algorithm: str = "zoe"):
        """
        Args:
            input_path (str): Path to the input file or directory.
            input_mode (str): Input mode, either "burst_photos" or "video".
            fps (int, optional): Frames per second of the source. Defaults to 6.
        """
        self.region_checker = RegionChecker()
        self.depth_calculator = DepthCalculator()
        self.io_handler = IOHandler(input_path)
        self.vehicle_detector = VehicleDetector(self.region_checker)
        self.oc_recognizer = OCR()

        if depth_estimator_algorithm == "zoe":
            self.depth_estimator = DepthEstimatorZoe(self.depth_calculator)
        elif depth_estimator_algorithm == "adabins":
            self.depth_estimator = DepthEstimatorAdabins(self.depth_calculator)    
        else:
            raise ValueError("depth_estimator_algorithm can be either 'zoe' or 'adabins' but got {}".format(depth_estimator_algorithm))

        self.license_plate_detector = LicensePlateDetector(
            self.oc_recognizer, self.depth_estimator, self.depth_calculator, self.region_checker)
        self.gt_speed_handler = GTHandler(xml_input_path)
        self.object_tracker = ObjectTracker(self.region_checker, self.gt_speed_handler)
        self.speed_estimator = SpeedEstimator(fps)
        self.fps_count = 0
        self.input_mode = input_mode
        self.input_path = input_path
        self.last_frame_time = time.time()
        self.iter = 0

    def process_frame(self, frame: np.ndarray, show_tracking: bool) -> None:
        """Process a single frame

        Args:
            frame (np.ndarray): The frame to process.
            show_tracking (bool): Flag indicating whether to display tracking information.
        """
        show_frame = frame.copy()
        vehicle_detections = self.vehicle_detector.detect_vehicles(show_frame)
        vehicle_dictionary = self.object_tracker.track_objects(
            show_frame, vehicle_detections, show_tracking)
        if len(vehicle_detections) != 0:
            vehicle_dictionary_lp, skip_speed_measurement = self.license_plate_detector.detect_license_plates(
                frame, show_frame, vehicle_dictionary, self.iter)
            if not skip_speed_measurement:
                self.speed_estimator.measure_distance(vehicle_dictionary_lp)
                self.speed_estimator.estimate_speed(show_frame, vehicle_dictionary_lp)

            # for key, value in vehicle_dictionary_lp.items():
            #     print("\n" + str(key))
            #     for subkey, subvalue in value.items():
            #         print(subkey, subvalue)
            # print("----------------------------------------------------------------------")
        
        self.measure_fps(show_frame)
        cv2.imshow("Frame", show_frame)

        return vehicle_dictionary

    def run(self, show_tracking: bool, ret: bool = True) -> None:
        """Run the speed camera algorithm

        Args:
            show_tracking (bool): Flag indicating whether to display tracking information.
            ret (bool, optional): Flag indicating if the stream is running or not. Defaults to True.
        """
        print("STARTING")
        if self.input_mode == "burst_photos":
            file_names = sorted(os.listdir(self.input_path))
            for file_name in file_names:
                self.iter += 1
                image_path = os.path.join(self.input_path, file_name)
                frame = cv2.imread(image_path)
                vehicle_dictionary = self.process_frame(frame, show_tracking)
                self.io_handler.check_stream(ret)
                self.io_handler.write_results(vehicle_dictionary)
                self.io_handler.end_stream()
            self.io_handler.terminate = False
            self.io_handler.write_results(vehicle_dictionary)
            self.io_handler.end_stream()

        elif self.input_mode == "video":
            while ret:
                ret, frame = self.io_handler.cap.read()
                self.io_handler.terminate = self.io_handler.check_stream(ret)
                if self.io_handler.terminate:
                    self.io_handler.write_results(vehicle_dictionary)
                    self.io_handler.end_stream()
                self.iter += 1
                # print(self.iter)
                vehicle_dictionary = self.process_frame(frame, show_tracking)

        else:
            raise ValueError("input_mode can only be 'video' or 'burst_photos' but got {}".format(self.input_mode))

    def measure_fps(self, frame) -> None:
        """Measure and display the frames per second

        Args:
            frame (np.ndarray): The frame to display the FPS on.
        """
        curr_time = time.time()
        frame_time = curr_time-self.last_frame_time
        fps = int(1/frame_time)
        # cv2.putText(frame, str(fps), [
        #             frame.shape[1]-100, frame.shape[0]-50], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        self.last_frame_time = time.time()

        # cv2.putText(frame, str(self.iter), [
        #             50, frame.shape[0]-50], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


class RegionChecker:
    def __init__(self) -> None:
        self.speed_measurement_area = [(0, 460), (1920, 370), (1920, 650), (0, 800), ]
        self.lane_1_area = [(222, 9), (599, 8), (519, 1071), (0, 1072)]
        self.lane_2_area = [(599, 9), (962, 8), (1349, 1067), (519, 1071)]
        self.lane_3_area = [(962, 8), (1307, 12), (1920, 1075), (1349, 1067)]
        self.invalid_detection_area = [(0, 180), (1450, 170), (1920, 775), (1920, 0), (0, 0)]


    def is_point_in_polygon(self, point, polygon):
        poly_path = mplPath.Path(polygon)
        inside = False
        if poly_path.contains_point(point):
            inside = True
        
        return inside


class GTHandler:
    def __init__(self, xml_input_path):
        with open(xml_input_path, 'r') as file:
            xml_content = file.read()
        self.root = ET.fromstring(xml_content)

    def get_vehicle_gt_params(self, lane, iframe):
        min_diff = float('inf')
        speed_gt = -1
        lane_gt = -1
        gt_iframe = -1

        for idx, vehicle in enumerate(self.root.find('gtruth')):
            if int(vehicle.attrib['lane']) == lane:
                diff = abs(int(vehicle.attrib['iframe']) - int(iframe))
                
                if diff < min_diff:
                    min_diff = diff
                    lane_gt = vehicle.attrib['lane']
                    gt_iframe = vehicle.attrib['iframe']

                    if vehicle.attrib['radar'] == 'True':
                        radar_element = vehicle.find('radar')
                        speed_gt = radar_element.attrib['speed']
                    else:
                        # If GT is not given
                        speed_gt = 0

                # If diff is too big, break
                elif diff > 200:
                    break

        return np.float16(speed_gt), int(lane_gt), int(gt_iframe)


class VehicleDetector:
    def __init__(self, region_checker: RegionChecker):
        self.model_vd = YOLO('./models/yolov8m.pt')
        # ["bicycle", "car", "motorcycle", "bus", "truck"]
        self.searched_class_indices = [1, 2, 3, 5, 7]
        self.region_checker = region_checker

    def detect_vehicles(self, frame: np.ndarray) -> np.ndarray:
        """Detect vehicles in a frame.

        Args:
            frame (np.ndarray): The frame in which to detect vehicles.

        Returns:
            np.ndarray: Detected vehicle bounding boxes. [x1,y1,x2,y2,conf]
        """
        detections = np.empty((0, 5))
        st_time = time.time()
        vd_results = self.model_vd(
            frame, stream=True, classes=self.searched_class_indices, conf=0.4, iou=0.3, agnostic_nms=True, verbose = False)

        for result in vd_results:
            boxes = result.boxes
            for box in boxes:
                x1_vd, y1_vd, x2_vd, y2_vd = tuple(map(int, box.xyxy[0]))
                mid_vd = ((x2_vd + x1_vd) / 2, (y2_vd + y1_vd) / 2)

                in_invalid_area = self.region_checker.is_point_in_polygon(mid_vd, self.region_checker.invalid_detection_area)
                if not in_invalid_area:
                    cv2.rectangle(frame, (x1_vd, y1_vd),
                                (x2_vd, y2_vd), (0, 165, 255), 2)
                    conf = math.ceil((box.conf[0]*100))/100
                    cls = int(box.cls[0])
                    cv2.putText(frame, f'{str(self.model_vd.model.names[cls])} {conf}', [
                                x1_vd, y1_vd], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                    bbox_arr = np.array([x1_vd, y1_vd, x2_vd, y2_vd, conf])
                    detections = np.vstack((detections, bbox_arr))
                    
        en_time = time.time()
        elapsed_time = en_time - st_time
        elapsed_time_ms = elapsed_time * 1000
        # print(f"Elapsed time: {elapsed_time_ms} ms")

        return detections


class ObjectTracker:
    """Class for object tracking"""

    def __init__(self, region_checker: RegionChecker, gt_speed_handler: GTHandler):
        self.tracker = Sort(max_age=2, min_hits=5, iou_threshold=0.3)
        self.vehicle_dictionary = {}
        self.region_checker = region_checker
        self.gt_speed_handler = gt_speed_handler

    def track_objects(self, frame: np.ndarray, detections: np.ndarray, show: bool) -> Dict[int, Dict]:
        """Track objects in a frame.

        Args:
            frame (np.ndarray): The frame to track objects in.
            detections (np.ndarray): Detected object bounding boxes.
            show (bool): Flag indicating whether to display tracking information.

        Returns:
            Dict[int, Dict]: Tracked object information.
        """
        track_results = self.tracker.update(detections)
        # [x1,y1,x2,y2,idx] - X coordinate from left, Y coordinate from top
        for vehicle in self.vehicle_dictionary:
            self.vehicle_dictionary[vehicle]['vd_tracked'] = False
            self.vehicle_dictionary[vehicle]['lp_tracked'] = False

        for result in track_results:
            idx = int(result[-1])
            bboxes = result[:-1]
            bboxes_int = tuple(map(int, bboxes))
            # Prevent any minus values
            bboxes_int = np.clip(bboxes_int, a_min=0, a_max=None)
            center = (int((result[2]+result[0])/2),
                      int((result[3]+result[1])/2))

            if idx not in self.vehicle_dictionary:
                self.vehicle_dictionary[idx] = {}
                self.vehicle_dictionary[idx]['vd_center'] = []
                self.vehicle_dictionary[idx]['vd_bbox_coords'] = []
                self.vehicle_dictionary[idx]['lp_bbox_coords'] = []
                self.vehicle_dictionary[idx]['lp_center'] = []
                self.vehicle_dictionary[idx]['lp_center_reconstructed'] = []
                self.vehicle_dictionary[idx]['lp_detected_frames'] = []
                self.vehicle_dictionary[idx]['gt_iframe'] = 0
                self.vehicle_dictionary[idx]['lp_conf'] = 0
                self.vehicle_dictionary[idx]['lp_type'] = ""
                self.vehicle_dictionary[idx]['lp_texts'] = []
                self.vehicle_dictionary[idx]['lp_text'] = ""
                self.vehicle_dictionary[idx]['ocr_conf'] = 0
                self.vehicle_dictionary[idx]['vd_tracked'] = False
                self.vehicle_dictionary[idx]['lp_tracked'] = False
                self.vehicle_dictionary[idx]['tracking_window_opened'] = False
                self.vehicle_dictionary[idx]['in_measurement_area'] = False
                self.vehicle_dictionary[idx]['travelled_distance'] = []
                self.vehicle_dictionary[idx]['speeds'] = []
                self.vehicle_dictionary[idx]['stationary'] = False
                self.vehicle_dictionary[idx]['was_stationary'] = False
                self.vehicle_dictionary[idx]['lane'] = 0
                self.vehicle_dictionary[idx]['gt_lane'] = 0
                self.vehicle_dictionary[idx]['predicted_speed'] = 0
                self.vehicle_dictionary[idx]['gt_speed'] = 0

            self.vehicle_dictionary[idx]['vd_bbox_coords'] = bboxes_int
            self.vehicle_dictionary[idx]['vd_tracked'] = True

            # Check which lane the vehicle is in
            if len(self.vehicle_dictionary[idx]['lp_center']) != 0:
                in_first_lane = self.region_checker.is_point_in_polygon(self.vehicle_dictionary[idx]['lp_center'][-1], self.region_checker.lane_1_area)
                in_second_lane = self.region_checker.is_point_in_polygon(self.vehicle_dictionary[idx]['lp_center'][-1], self.region_checker.lane_2_area)
                in_third_lane = self.region_checker.is_point_in_polygon(self.vehicle_dictionary[idx]['lp_center'][-1], self.region_checker.lane_3_area)
                if in_first_lane:
                    self.vehicle_dictionary[idx]['lane'] = 1
                elif in_second_lane:
                    self.vehicle_dictionary[idx]['lane'] = 2
                elif in_third_lane:
                    self.vehicle_dictionary[idx]['lane'] = 3

            # Getting ground truth frame index, lane, speed
            if int(self.vehicle_dictionary[idx]['gt_speed']) <= 0 and len(self.vehicle_dictionary[idx]['lp_center_reconstructed']) != 0:
                avg_detected_frame = int(np.average(self.vehicle_dictionary[idx]['lp_detected_frames']))
                gt_speed, gt_lane, gt_iframe = self.gt_speed_handler.get_vehicle_gt_params(self.vehicle_dictionary[idx]['lane'], avg_detected_frame)
                self.vehicle_dictionary[idx]['gt_speed'] = gt_speed
                self.vehicle_dictionary[idx]['gt_lane'] = gt_lane
                self.vehicle_dictionary[idx]['gt_iframe'] = gt_iframe

            # Check if the vehicle is stationary
            self.vehicle_dictionary[idx]['stationary'] = False
            if len(self.vehicle_dictionary[idx]['vd_center']) > 1:
                mid_diff_x = np.abs(center[0] - self.vehicle_dictionary[idx]['vd_center'][-1][0])
                mid_diff_y = np.abs(center[1] - self.vehicle_dictionary[idx]['vd_center'][-1][1])
                if 4 > ((mid_diff_x+mid_diff_y)/2):
                    self.vehicle_dictionary[idx]['stationary'] = True
                    self.vehicle_dictionary[idx]['was_stationary'] = True

            if show:
                self.vehicle_dictionary[idx]['vd_center'].append(center)
                if idx % 2 == 0:
                    R = 255
                    B = 0
                    G = 0
                else:
                    R = 0
                    B = 0
                    G = 255
                for i in range(len(self.vehicle_dictionary[idx]['vd_center']) - 1):
                    center_x = self.vehicle_dictionary[idx]['vd_center'][i]
                    center_x = tuple(map(int, center_x))
                    center_y = self.vehicle_dictionary[idx]['vd_center'][i + 1]
                    center_y = tuple(map(int, center_y))
                    cv2.line(frame, center_y, center_x, (B, G, R), thickness=2)
                cv2.putText(frame, f'{str(idx)}', [
                            bboxes_int[2], bboxes_int[1]], cv2.FONT_HERSHEY_SIMPLEX, 1, (B, G, R), 2)
            else:
                self.vehicle_dictionary[idx]['vd_center'] = center
                cv2.putText(frame, f'{str(idx)}', [
                            bboxes_int[2], bboxes_int[1]], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        return self.vehicle_dictionary


class OCR:
    """Class for optical character recognition"""

    def __init__(self):
        self.config =load_config()
        self.config_multi = load_config("ocr_config_multirow.yaml")
        self.single_net = create_network(source='./OCR/singlerow_model.pth')
        self.multi_net = create_network_multi(source='./OCR/multirow_model.pth')

    def read_license_plate(self, cropped_lp, lp_type: str):
            """Reading license plate characters."""
            cropped_lp_ = torch.tensor(cropped_lp, dtype=torch.float32)
            cropped_lp_tensor = cropped_lp_.permute(2, 0, 1)
            self.single_net.eval()
            self.multi_net.eval()
            mean = torch.mean(cropped_lp_tensor, dim=[1, 2])
            std = torch.std(cropped_lp_tensor, dim=[1, 2])
            transform_single = transforms.Compose([transforms.Resize([self.config["data"]["img_target_height"],self.config["data"]["img_target_width"]], antialias=True),
                                            transforms.Normalize(mean, std)])
            transform_multi = transforms.Compose([transforms.Resize([self.config_multi["data"]["img_target_height"],self.config_multi["data"]["img_target_width"]], antialias=True),
                                            transforms.Normalize(mean, std)])       

            if lp_type == "single":
                with torch.inference_mode():
                    # print("Using single row model")
                    img_tensor = transform_single(cropped_lp_tensor).unsqueeze(0)
                    inputs = img_tensor.cuda()
                    outputs = self.single_net(inputs)
            elif lp_type == "multi":
                with torch.inference_mode():
                    # print("Using multi row model")
                    self.multi_uses += 1
                    img_tensor = transform_multi(cropped_lp_tensor).unsqueeze(0)
                    inputs = img_tensor.cuda()
                    outputs = self.multi_net(inputs)

            _, predicted = torch.max(outputs, dim=0)
            predicted_wo_encoding = [(self.config["data"]["classes"][idx]) for idx in predicted]
            predicted_wo_encoding_str = ''.join(predicted_wo_encoding)

            return predicted_wo_encoding_str


class DepthCalculator:
    def __init__(self):
        self.real_sensor_resolution_w = 3280
        self.img_width_pixel = 1920
        self.resolution_ratio_w = self.real_sensor_resolution_w / self.img_width_pixel
        self.focal_length_mm = 3.04
        self.sensor_width = 3.68
        self.focal_length_pixel_x = ((self.focal_length_mm/self.sensor_width) * self.img_width_pixel) / self.resolution_ratio_w # 928.44

        # self.sensor_height = 2.76
        # self.img_height_pixel = 1080
        # self.focal_length_pixel_y = (self.focal_length_mm/self.sensor_height) * self.img_height_pixel

        self.max_distance = 13.5
        self.min_distance = 4

        # self.ref_point_1 = np.array([1108, 248])
        # self.ref_point_2 = np.array([1400, 251])
        self.ref_point_1 = np.array([649, 262])
        self.ref_point_2 = np.array([647, 43])
        self.ref_distance_irl = 4.8
        self.scaling_ratios = np.array([], dtype=np.float16)
        self.avg_running_scaling_ratio = 0
        self.avg_scaling_ratio = 0
        self.threshold = 0
        self.iter = 1

    @staticmethod
    def get_license_plate_depth(cropped_lp_depth_map):
        lp_depth = np.float16(np.average(cropped_lp_depth_map))

        return lp_depth

    def get_3D_coordinates(self, depth_map, u, v, z):
        px = depth_map.shape[1]/2
        py = depth_map.shape[0]/2
        x = (u*z - px*z) / self.focal_length_pixel_x
        y = (v*z - py*z) / self.focal_length_pixel_x

        return [np.float16(x), np.float16(y), np.float16(z)]

    def normalize_depth_map(self, depth_map, scaling_ratio = 1, method = 'scaling'):
   
        # Rejection of lower and upper 1%
        lower_threshold = np.percentile(depth_map, 1)
        upper_threshold = np.percentile(depth_map, 99)

        # mask = (depth_map >= lower_threshold) & (depth_map <= upper_threshold)
        # depth_map[~mask] = np.nan


        # depth_map[depth_map < lower_threshold] = lower_threshold
        # depth_map[depth_map > upper_threshold] = upper_threshold

        
        if method == 'normalizing':
            depth_map_modified = ((depth_map-np.min(depth_map))*(self.max_distance-self.min_distance))/(np.max(depth_map)-np.min(depth_map)) + self.min_distance

            # plt.hist(depth_map_normalized, bins=[2.9, 4])
            # # plt.show()

            # print(np.min(depth_map_normalized))
            # filtered_values = depth_map_normalized[(depth_map_normalized >= 2.9) & (depth_map_normalized <= 4)]

            # # Print the filtered values
            # print(filtered_values.shape)
            # print(filtered_values)

        elif method == 'scaling':
            depth_map_modified = depth_map*scaling_ratio

        else:
            raise ValueError("method can be either 'scaling' or 'normalizing' but got {}".format(method))


        return depth_map_modified
    
    def get_scaling_ratio(self, depth_map):
        skip_frame = False
        px = depth_map.shape[1]/2
        py = depth_map.shape[0]/2

        u_1 = self.ref_point_1[0]
        v_1 = self.ref_point_1[1]
        z_1 = np.average(depth_map[v_1-5:v_1+5, u_1-5:u_1+5])
        x_1 = (u_1*z_1 - px*z_1) / self.focal_length_pixel_x
        y_1 = (v_1*z_1 - py*z_1) / self.focal_length_pixel_x
        ref_point_reconstructed_1 = np.array([np.float16(x_1), np.float16(y_1), np.float16(z_1)])

        u_2 = self.ref_point_2[0]
        v_2 = self.ref_point_2[1]
        z_2 = np.average(depth_map[v_2-5:v_2+5, u_2-5:u_2+5])
        x_2 = (u_2*z_2 - px*z_2) / self.focal_length_pixel_x
        y_2 = (v_2*z_2 - py*z_2) / self.focal_length_pixel_x
        ref_point_reconstructed_2 = np.array([np.float16(x_2), np.float16(y_2), np.float16(z_2)])

        ref_point_reconstructed_distance = np.float16(np.sqrt(np.sum((ref_point_reconstructed_1-ref_point_reconstructed_2)**2)))
        scaling_ratio = np.float16(self.ref_distance_irl/ref_point_reconstructed_distance)

        self.scaling_ratios = np.append(self.scaling_ratios, scaling_ratio)
        if self.scaling_ratios.size % 40 == 0:
            self.avg_running_scaling_ratio = np.average(self.scaling_ratios)
            print("AVG RUNNING RATIO CHANGED", self.avg_running_scaling_ratio)
            if self.avg_scaling_ratio == 0:
                self.avg_scaling_ratio = self.avg_running_scaling_ratio
                print("AVG RATIO CHANGED", self.avg_scaling_ratio)
            else:
                self.avg_scaling_ratio = np.average([self.avg_running_scaling_ratio, self.avg_scaling_ratio])
                print("AVG RATIO CHANGED", self.avg_scaling_ratio)
            self.threshold = self.avg_scaling_ratio*0.5
            
        
        if self.avg_scaling_ratio != 0:
            if not (self.scaling_ratios[-1] >= self.avg_scaling_ratio - self.threshold) & (self.scaling_ratios[-1] <= self.avg_scaling_ratio + self.threshold):
                self.scaling_ratios = self.scaling_ratios[:-1]
                skip_frame = True
                print(scaling_ratio)
        
        self.iter += 1

        return scaling_ratio, skip_frame
    

class DepthEstimatorZoe:
    def __init__(self, depth_calculator: DepthCalculator):
        # torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
        repo = "isl-org/ZoeDepth"
        model = "ZoeD_NK"
        model_zoe = torch.hub.load(repo, model, pretrained=True)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using", DEVICE)
        self.zoe = model_zoe.to(DEVICE)
        self.depth_calculator = depth_calculator

    def create_depth_map(self, input_frame: np.ndarray, scaling_ratio: np.float16 = 1):
        img = Image.fromarray(input_frame)
        input_img = img.convert("RGB")
        depth_map = self.zoe.infer_pil(input_img, pad_input = False, with_flip_aug = False)

        return depth_map


class DepthEstimatorAdabins:
    def __init__(self, depth_calculator: DepthCalculator):
        self.infer_helper = InferenceHelper(dataset='nyu')
        self.depth_calculator = depth_calculator

    def create_depth_map(self, input_frame: np.ndarray):
        img = Image.fromarray(input_frame)
        input_img = img.resize((640,480)).convert("RGB")
        bin_centers, depth_map = self.infer_helper.predict_pil(input_img)
        depth_map = depth_map[0, 0, :, :]
        depth_map_resized = Image.fromarray(depth_map).resize((1920,1080))
        depth_map_array = np.array(depth_map_resized)

        return depth_map_array
     

class LicensePlateDetector:
    def __init__(self, oc_recognizer: OCR, depth_estimator, depth_calculator: DepthCalculator, region_checker: RegionChecker):
        """
        Args:
            oc_recognizer(OCR): The object that performs license plate OCR.
        """
        self.model_lp = YOLO(
            './models/lp_detector.pt')
        self.oc_recognizer = oc_recognizer
        self.depth_estimator = depth_estimator
        self.depth_calculator = depth_calculator
        self.region_checker = region_checker
        self.depth_estimation_ran = False
        self.skip_frame = False

    def detect_license_plates(self, frame: np.ndarray, show_frame: np.ndarray, vehicle_dictionary: Dict, iter: int) -> Dict:
        """Detect license plates in a frame and update vehicle predictions.

        Args:
            frame (np.ndarray): The frame to detect license plates in.
            show_frame (np.ndarray): The frame to display license plate information on.
            vehicle_dictionary (Dict): Tracked vehicle information.

        Returns:
            Dict: Updated vehicle predictions.
        """
        self.depth_estimation_ran = False
        self.skip_frame = False
        for idx in vehicle_dictionary.keys():
            bbox_vd = vehicle_dictionary[idx]['vd_bbox_coords']
            cropped_vehicle = np.array(
                frame[bbox_vd[1]:bbox_vd[3], bbox_vd[0]:bbox_vd[2]])

            if vehicle_dictionary[idx]['vd_tracked']:
                # cv2.imshow(str(idx), cropped_vehicle)
                cv2.putText(show_frame, vehicle_dictionary[idx]['lp_text'], [
                            bbox_vd[0], bbox_vd[3]+35], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                vehicle_dictionary[idx]['tracking_window_opened'] = True
            elif vehicle_dictionary[idx]['vd_tracked'] == False and vehicle_dictionary[idx]['tracking_window_opened'] == True:
                cv2.destroyWindow(str(idx))
                if cv2.getWindowProperty(str(idx) + " license plate", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow(str(idx) + " license plate")
                vehicle_dictionary[idx]['tracking_window_opened'] = False

            if self.skip_frame:
                break

            if vehicle_dictionary[idx]['vd_tracked'] and not vehicle_dictionary[idx]['stationary']:
                lp_preds = self.model_lp(cropped_vehicle, imgsz=640, iou=0.5, verbose = False)
                cls_array = []
                for result in lp_preds: # Get predictions
                    if len(result) > 0 and not self.depth_estimation_ran:
                        depth_map_out = self.depth_estimator.create_depth_map(frame)
                        # plt.imshow(depth_map_out, cmap='magma')
                        # plt.colorbar(label='Depth')
                        # plt.show()
                        scaling_ratio, self.skip_frame = self.depth_calculator.get_scaling_ratio(depth_map_out)                        
                        if self.skip_frame:
                            print("Frame skipped because of incorrect depth map")
                            break
                        depth_map = self.depth_calculator.normalize_depth_map(depth_map_out, scaling_ratio, method='normalizing')

                        # plt.imshow(depth_map, cmap='magma')
                        # plt.colorbar(label='Depth')
                        # plt.show()
                        self.depth_estimation_ran = True
                    boxes = result.boxes # Get bounding boxes
                    for box in boxes: # Iterate through bounding boxes
                        if len(box.xyxy[0]) != 0: # Empty bounding box check
                            cls = int(box.cls[0])
                            if cls == 0:
                                vehicle_dictionary[idx]['lp_type'] = "single"
                            elif cls == 1:
                                vehicle_dictionary[idx]['lp_type'] = "multi"
                            vehicle_dictionary[idx]['lp_tracked'] = True
                            x1_lp, y1_lp, x2_lp, y2_lp = tuple(
                                map(int, box.xyxy[0]))
                            bbox_arr_lp = np.array(
                                [x1_lp, y1_lp, x2_lp, y2_lp])
                            lp_center = (
                                int(bbox_vd[0] + (x1_lp+x2_lp)/2), int(bbox_vd[1] + (y1_lp+y2_lp)/2))
                            vehicle_dictionary[idx]['lp_bbox_coords'] = bbox_arr_lp
                            cropped_lp = np.array(
                                cropped_vehicle[y1_lp:y2_lp, x1_lp:x2_lp])
                            conf = math.ceil((box.conf[0]*100))/100
                            vehicle_dictionary[idx]['lp_conf'] = conf
                            vehicle_dictionary[idx]['lp_center'].append(lp_center)

                            lp_text = self.oc_recognizer.read_license_plate(cropped_lp, vehicle_dictionary[idx]['lp_type'])
                            vehicle_dictionary[idx]['lp_texts'].append(lp_text)
                            if len(vehicle_dictionary[idx]['lp_texts']) > 3:
                                lp_texts = vehicle_dictionary[idx]['lp_texts']
                                stripped_texts = [lp_texts.strip() for lp_texts in lp_texts]
                                counter = Counter(stripped_texts)
                                most_common_string = counter.most_common(1)[0][0]
                                vehicle_dictionary[idx]['lp_text'] = most_common_string

                            cropped_vehicle_depth_map = np.array(depth_map[bbox_vd[1]:bbox_vd[3], bbox_vd[0]:bbox_vd[2]])
                            cropped_lp_depth_map = np.array(cropped_vehicle_depth_map[y1_lp:y2_lp, x1_lp:x2_lp])
                            lp_depth = self.depth_calculator.get_license_plate_depth(cropped_lp_depth_map)
                            vehicle_dictionary[idx]['lp_detected_frames'].append(iter)

                            is_point_in_meas_area = self.region_checker.is_point_in_polygon((lp_center[0], lp_center[1]), self.region_checker.speed_measurement_area)
                            vehicle_dictionary[idx]['in_measurement_area'] = is_point_in_meas_area
                            if is_point_in_meas_area and vehicle_dictionary[idx]['lp_tracked']:
                                world_coordinates = self.depth_calculator.get_3D_coordinates(depth_map, lp_center[0], lp_center[1], lp_depth)
                                vehicle_dictionary[idx]['lp_center_reconstructed'].append(world_coordinates)

                            # if vehicle_dictionary[idx]['lp_tracked']:
                            #     cv2.imshow(
                            #         str(idx) + " license plate", cropped_lp)
                            #     cv2.moveWindow(
                            #         str(idx) + " license plate", x_w, y_w + h_w)
                            #     vehicle_dictionary[idx]['tracking_window_opened'] = True
                            # elif vehicle_dictionary[idx]['vd_tracked'] == False and vehicle_dictionary[idx]['tracking_window_opened'] == True:
                            #     cv2.destroyWindow(str(idx) + " license plate")
                            #     vehicle_dictionary[idx]['tracking_window_opened'] = False

                            break # Take only the first license plate detection if duplicates are found
                    break # Take only the first license plate detection if duplicates are found

        return vehicle_dictionary, self.skip_frame


class SpeedEstimator:
    """Class for the speed estimation."""

    def __init__(self, fps):
        """
        Args:
            fps (int): Frames per second of the source file.
        """
        self.fps = fps

    def measure_distance(self, vehicle_dictionary: Dict) -> Dict:
        """
        Measure the distance to tracked objects.

        Args:
            vehicle_dictionary (Dict): Dictionary containing information about tracked objects.

        Returns:
            vehicle_dictionary: Updated dictionary of tracked objects with distance measurements.
        """
        for idx in vehicle_dictionary.keys():
            if vehicle_dictionary[idx]['lp_tracked'] and len(vehicle_dictionary[idx]['lp_center_reconstructed']) >= 2 and vehicle_dictionary[idx]['in_measurement_area']:
                src = np.array([vehicle_dictionary[idx]['lp_center_reconstructed'][-2][0], 
                                   vehicle_dictionary[idx]['lp_center_reconstructed'][-2][1], 
                                   vehicle_dictionary[idx]['lp_center_reconstructed'][-2][2]])
                dst = np.array([vehicle_dictionary[idx]['lp_center_reconstructed'][-1][0], 
                                    vehicle_dictionary[idx]['lp_center_reconstructed'][-1][1], 
                                    vehicle_dictionary[idx]['lp_center_reconstructed'][-1][2]])
                travelled_distance = np.float16(np.sqrt(np.sum((dst-src)**2)))
                vehicle_dictionary[idx]['travelled_distance'].append(travelled_distance)


        return vehicle_dictionary

    def estimate_speed(self, frame: np.ndarray, vehicle_dictionary: Dict) -> Dict:
        """
        Estimate the speed of tracked objects.

        Args:
            frame(np.ndarray): Current frame of the video.
            vehicle_dictionary (Dict): Dictionary containing information about tracked objects.
            iter(int): Current frame number.

        Returns:
            vehicle_dictionary: Updated dictionary of tracked objects with speed estimates.
        """
        final_speed = 0
        for idx in vehicle_dictionary.keys():
            if len(vehicle_dictionary[idx]['travelled_distance']) != 0 and vehicle_dictionary[idx]['lp_tracked'] and vehicle_dictionary[idx]['in_measurement_area']:
                speed = np.float16((vehicle_dictionary[idx]['travelled_distance'][-1] / ((vehicle_dictionary[idx]['lp_detected_frames'][-1]-vehicle_dictionary[idx]['lp_detected_frames'][-2])/self.fps))*3.6)
                if len(vehicle_dictionary[idx]['speeds']) >= 3:
                    avg_speed = np.average(vehicle_dictionary[idx]['speeds'])
                    upper_limit = avg_speed*1.05
                    lower_limit = avg_speed*0.95
                    speed = np.float16(max(lower_limit, min(speed, upper_limit)))
                vehicle_dictionary[idx]['speeds'].append(speed)
                if len(vehicle_dictionary[idx]['speeds']) >= 3:
                    final_speed = np.average(vehicle_dictionary[idx]['speeds'])
                    vehicle_dictionary[idx]['predicted_speed'] = final_speed
            if vehicle_dictionary[idx]['predicted_speed'] != 0 and vehicle_dictionary[idx]['vd_tracked']:
                cv2.putText(frame, str(int(np.round(vehicle_dictionary[idx]['predicted_speed']))) + " km/h", (vehicle_dictionary[idx]['vd_bbox_coords'][2],
                                                            vehicle_dictionary[idx]['vd_bbox_coords'][3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                cv2.putText(frame, str(int(np.round(float(vehicle_dictionary[idx]['gt_speed'])))) + " km/h", (vehicle_dictionary[idx]['vd_bbox_coords'][2],
                                                            vehicle_dictionary[idx]['vd_bbox_coords'][3]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                                        
        return vehicle_dictionary


class IOHandler:
    """Class for input output handling."""

    def __init__(self, input_path: str):
        """
        Args:
            input_path (str): Path to the input video or photos.
        """
        self.input_path = input_path
        self.cap = cv2.VideoCapture(input_path)
        self.terminate = False

    def check_stream(self, ret: bool) -> None:
        """
        Check if the video stream has ended or if the user has quit.

        Args:
            ret (bool): Flag indicating if the video capture was successful.
        """
        if cv2.waitKey(10) & 0xFF == ord('p'):
            cv2.waitKey(0)
        if cv2.waitKey(10) & 0xFF == ord('q') or not ret:
            self.terminate = True
        
        return self.terminate

    def write_results(self, vehicle_dictionary: Dict):
        if self.terminate:
            df = pd.DataFrame.from_dict(vehicle_dictionary, orient='index')
            df['predicted_speed'] = df['predicted_speed'].apply(lambda x: "{:.2f}".format(x))
            df['gt_speed'] = df['gt_speed'].apply(lambda x: "{:.2f}".format(x))
            df['lp_first_detected_frame'] = df['lp_detected_frames'].apply(lambda x: x[0] if x else None)
            df.to_excel('speed_estimation_results.xlsx', columns=['lane', 'gt_lane', 'predicted_speed', 'gt_speed', 'lp_first_detected_frame', 'gt_iframe', 'was_stationary'])

    
    def end_stream(self):
        if self.terminate:
            self.cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    def pop_stationary(vehicle_dictionary: Dict):
        for idx in vehicle_dictionary.keys():
            if vehicle_dictionary[idx]['was_stationary']:
                pass

            
def main():
    parser = argparse.ArgumentParser(description="Traffic Speed Camera Script")

    parser.add_argument("--input_path", default=r"./samples/ocr_sample.MOV", help="Path to the input media")
    parser.add_argument("--xml_input_path", default=r"vehicles.xml", help="Path to the speed measurement benchmark XML file")
    parser.add_argument("--input_mode", default="video", choices=["video", "burst_photos"], help="Input mode (video or burst_photos)")
    parser.add_argument("--fps", type=float, default=30.15, help="Frames per second of the input video")
    parser.add_argument("--depth_estimator_algorithm", default="zoe", choices=["zoe", "adabins"], help="Depth estimator algorithm used")

    args = parser.parse_args()

    speed_camera = TrafficSpeedCamera(
        input_path=args.input_path,
        xml_input_path=args.xml_input_path,
        input_mode=args.input_mode,
        fps=args.fps,
        depth_estimator_algorithm=args.depth_estimator_algorithm
    )
    speed_camera.run(show_tracking=True)


if __name__ == '__main__':
    main()
