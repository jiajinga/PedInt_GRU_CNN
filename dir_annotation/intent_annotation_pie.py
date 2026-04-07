"""
Person Intent Tracker

This script processes videos to annotate object intents in frames.
Key features:
- Handles videos from moving vehicle perspective
- Scales bounding boxes (video frames are half size of reference)
- Annotates vertical intent (along road) and lateral intent (across frame)  意图
- Considers motion relative to road, not the moving vehicle  运动是相对于道路的，而不是相对于移动的车辆

This code should be run in an environment with access to the YOLOv8 model.
标注和意图推理我用的是两个不同的环境，一个是 yolo，另一个是虚拟机建立的专门用于推理的环境，后者比较老，害怕新版的不适配
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional, Any
import argparse
from tqdm import tqdm
import json
import requests
import tempfile
import random
import os
from collections import deque
from itertools import islice
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.transforms import functional as F
from multiprocessing import Pool, cpu_count
import xml.etree.ElementTree as ET  # 用于解析、编辑 XML 文件

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize models
yolo_model = YOLO('yolov8x.pt')


class IntentAnalyzer:
    def __init__(self, frame_size: Tuple[int, int], motion_threshold: float = 0.2):
        """
        Initialize intent analyzer

        Args:
            frame_size: Size of the video frames (width, height)
            motion_threshold: Minimum pixel movement to consider as motion
        """
        self.frame_size = frame_size
        self.position_threshold = frame_size[0] / 3  # Center line for left/right position
        self.motion_threshold = motion_threshold
        self.track_histories = {}  # Store centroid history for each track
        # TODO: 为什么要进行缩放？
        self.scale_factor = 0.5  # Video frames are half size of reference
        self.camera_motion = []  # Store camera centroids for each frame

    def set_camera_motion(self, camera_centroids: list):
        """
        Set the camera centroids for the video (should be called before intent analysis)
        """
        self.camera_motion = camera_centroids

    def scale_bbox(self, bbox: List[int]) -> List[int]:
        """Scale bounding box coordinates to match video frame size"""
        return [int(coord * self.scale_factor) for coord in bbox]

    def update_track_history(self, track_id: int, centroid: Tuple[int, int]):
        """Update tracking history for a specific track ID"""
        if track_id not in self.track_histories:
            self.track_histories[track_id] = []
        self.track_histories[track_id].append(centroid)

    # 相对于自车的左右，一分为三
    def determine_position(self, centroid: Tuple[int, int]) -> str:
        """Determine object position relative to ego vehicle"""
        if centroid[0] < self.position_threshold:
            return "Left of ego vehicle"
        if centroid[0] > self.frame_size[0] - self.position_threshold:
            return "Right of ego vehicle"
        return "Front of ego vehicle"

    # TODO: 这个函数实际上使用的时候是直接用行人整体轨迹的首位位移差减去相机的运动
    def determine_intent(self, track_id: int, cam_dx, cam_dy) -> List[str]:
        """
        Determine vertical and lateral intent based on track history and camera motion.
        Returns [lateral_intent, vertical_intent]
        """
        history = list(self.track_histories.get(track_id, []))
        if len(history) < 3:
            return ["stationary", "stationary"]

        # If camera motion is available, subtract it from object centroids
        if self.camera_motion and len(self.camera_motion) >= len(history) and False:
            rel_history = [
                (obj[0] - cam[0], obj[1] - cam[1])
                for obj, cam in zip(history, self.camera_motion[-len(history):])
            ]
        else:
            rel_history = history

        # Net movement (first to last)
        net_dx = rel_history[-1][0] - rel_history[0][0]
        net_dy = rel_history[-1][1] - rel_history[0][1]

        # Calculate motion over multiple windows to catch subtle movements
        windows = [(0, len(rel_history) // 2), (len(rel_history) // 2, len(rel_history))]
        dx_values = []
        dy_values = []
        for start, end in windows:
            if end - start < 2:
                continue
            window = rel_history[start:end]
            dx = [window[i + 1][0] - window[i][0] for i in range(len(window) - 1)]
            dy = [window[i + 1][1] - window[i][1] for i in range(len(window) - 1)]
            dx_values.extend(dx)
            dy_values.extend(dy)
        if not dx_values or not dy_values:
            return ["stationary", "stationary"]

        # Calculate average and consistency of motion
        avg_dx = np.mean(dx_values)
        avg_dy = np.mean(dy_values)
        std_dx = np.std(dx_values)
        std_dy = np.std(dy_values)

        # Weighted combination: net movement gets higher weight
        final_dx = 1 * net_dx + 0 * avg_dx - cam_dx
        final_dy = 1 * net_dy + 0 * avg_dy - cam_dy

        # Determine lateral intent (horizontal motion)
        lateral_intent = "stationary"
        # if abs(final_dx) > self.motion_threshold and std_dx < abs(final_dx) * 2:
        if abs(final_dx) > self.motion_threshold:
            lateral_intent = "goes to the right" if final_dx > 0 else "goes to the left"

        # Determine vertical intent (along road)
        # Moving up in the frame (negative dy) means moving away from ego vehicle
        vertical_intent = "stationary"
        if abs(final_dy) > self.motion_threshold:
            vertical_intent = "moves away from ego vehicle" if final_dy < 0 else "moves towards ego vehicle"
        return [lateral_intent, vertical_intent]

    def generate_description(self, intent: List[str]) -> str:
        """Generate human-readable description from intent"""
        lateral, vertical = intent
        return f"Lateral: {lateral}, Vertical: {vertical}"


def build_dataset(video_root, anno_root):
    """数据集加载方式，适配 PIE，获取了系列 ID 视频路径 标注路径"""
    samples = {}
    sample_id = []

    for set_name in sorted(os.listdir(video_root)):
        set_id = set_name[-1] + "_"
        set_path = os.path.join(video_root, set_name)

        for video_name in sorted(os.listdir(set_path)):
            video_path = os.path.join(set_path, video_name)
            sample_id = set_id + video_name.split(".")[0][-2:]
            samples[sample_id] = {"video_path": video_path}

    for set_name in sorted(os.listdir(anno_root)):
        set_id = set_name[-1] + "_"
        set_path = os.path.join(anno_root, set_name)

        for anno_name in sorted(os.listdir(set_path)):
            if not anno_name.endswith(".xml"):
                continue
            sample_id = set_id + anno_name.split(".")[0][8:10]
            anno_path = os.path.join(set_path, anno_name)
            samples[sample_id]["anno_path"] = anno_path

    return samples


def parse_pedestrians(anno_path: str) -> Dict[str, Dict[str, Any]]:
    """
    从标注 XML 文件中提取所有 pedestrian 轨迹。

    Args:
        anno_path: XML 文件路径

    Returns:
        字典，键为行人 id，值为包含以下字段的字典：
            - "boxes": numpy 数组，形状 (N, 4)，每行为 [xtl, ytl, xbr, ybr]
            - "frame_nums": list，对应每一帧的 frame 编号
    """
    tree = ET.parse(anno_path)
    root = tree.getroot()
    pedestrians = {}

    for track in root.findall('track'):
        if track.get('label') != 'pedestrian':
            continue

        boxes_data = []
        frames = []

        for box in track.findall('box'):
            # 获取 frame 和坐标
            frame = int(box.get('frame'))
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            boxes_data.append([xtl, ytl, xbr, ybr])
            frames.append(frame)

        # 获取 id（通常第一个 box 的 attribute 中包含）
        first_box = track.find('box')
        if first_box is None:
            continue
        id_attr = first_box.find('attribute[@name="id"]')
        if id_attr is None:
            continue
        ped_id = id_attr.text

        pedestrians[ped_id] = {
            "boxes": boxes_data,
            "frame_nums": frames
        }

    return pedestrians


def save_xml_annotation(anno_path, output_root, track_histories, matches):
    """向标注文件中添加新的行人及方向意图"""

    # 解析 XML 文件
    tree = ET.parse(anno_path)
    root = tree.getroot()

    # 先取出来已经匹配的意图
    match_intent = {v: track_histories[k]['dir_intent'] for k, v in matches.items()}

    id_max = 0
    ped_id = None

    # 对匹配好了的轨迹添加 dir_intent
    for track in root.findall('track'):
        if track.get('label') != 'pedestrian':
            continue

        for box in track.findall('box'):
            # 获取该 box 对应的行人 id
            id_attr = box.find("attribute[@name='id']")
            if id_attr is None:
                continue
            ped_id = id_attr.text

            if ped_id not in match_intent:
                continue

            # 检查是否已存在 dir_intent 属性，若存在则先移除（避免重复）
            existing = box.find("attribute[@name='dir_intent']")
            if existing is not None:
                box.remove(existing)

            # 添加新属性
            attr = ET.SubElement(box, 'attribute')
            attr.set('name', 'dir_intent')
            attr.text = ', '.join(match_intent[ped_id])

        id_max = int(ped_id.split('_')[-1]) if int(ped_id.split('_')[-1]) > id_max else id_max

    set_num, v_num = ped_id.split('_')[0], ped_id.split('_')[1]
    # 接着创建未匹配的轨迹的标注
    for track_id, track_data in track_histories.items():
        boxes = track_data.get('boxes')
        frame_nums = track_data.get('frame_nums')
        dir_intent = track_data.get('dir_intent')

        if not boxes or not frame_nums or len(boxes) != len(frame_nums) or track_id in matches:
            print(f"Warning: track_id {track_id} has mismatched boxes and frame_nums or has been matched, skipping")
            continue

        id_max += 1
        new_id = '_'.join([set_num, v_num, str(id_max)])

        # 创建 track 元素
        track_elem = ET.SubElement(root, 'track')
        track_elem.set('label', 'pedestrian')

        # 为每一帧添加 box
        for frame, box_coords in zip(frame_nums, boxes):
            if len(box_coords) != 4:
                print(f"Warning: track_id {track_id} has invalid box coordinates: {box_coords}")
                continue

            xtl, ytl, xbr, ybr = box_coords

            # 创建 box 元素
            box_elem = ET.SubElement(track_elem, 'box')
            box_elem.set('frame', str(frame))
            box_elem.set('keyframe', '0')
            box_elem.set('occluded', '0')
            box_elem.set('outside', '0')
            box_elem.set('xbr', f"{xbr:.2f}")
            box_elem.set('xtl', f"{xtl:.2f}")
            box_elem.set('ybr', f"{ybr:.2f}")
            box_elem.set('ytl', f"{ytl:.2f}")

            # 添加 attribute 元素
            # id
            id_attr = ET.SubElement(box_elem, 'attribute')
            id_attr.set('name', 'id')
            id_attr.text = new_id

            # gesture (默认 __undefined__)
            gesture_attr = ET.SubElement(box_elem, 'attribute')
            gesture_attr.set('name', 'gesture')
            gesture_attr.text = '__undefined__'

            # action (默认 standing)
            action_attr = ET.SubElement(box_elem, 'attribute')
            action_attr.set('name', 'action')
            action_attr.text = 'standing'

            # cross (默认 not-crossing)
            cross_attr = ET.SubElement(box_elem, 'attribute')
            cross_attr.set('name', 'cross')
            cross_attr.text = 'not-crossing'

            # look (默认 not-looking)
            look_attr = ET.SubElement(box_elem, 'attribute')
            look_attr.set('name', 'look')
            look_attr.text = 'not-looking'

            # occlusion (默认 none)
            occlusion_attr = ET.SubElement(box_elem, 'attribute')
            occlusion_attr.set('name', 'occlusion')
            occlusion_attr.text = 'none'

            # dir_intent
            dir_intent_attr = ET.SubElement(box_elem, 'attribute')
            dir_intent_attr.set('name', 'dir_intent')
            dir_intent_attr.text = ', '.join(dir_intent)

    # 生成保存路径，创建输出目录
    anno_path_normalized = os.path.normpath(anno_path)  # 为了避免分割出错
    output_path = os.path.join(output_root, anno_path_normalized.split(os.sep)[-2],
                               anno_path_normalized.split(os.sep)[-1])
    os.makedirs(os.path.join(output_root, anno_path.split(os.sep)[-2]), exist_ok=True)

    # 保存修改后的 XML 文件
    tree.write(output_path)
    print(f"new annotations have been saved to {output_path}")


# 用光流法估计相机运动
def estimate_camera_motion(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[float, float]:
    """
    Estimate camera motion between two frames using optical flow.
    Returns the average motion vector (dx, dy).
    """
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    p0 = cv2.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    if p0 is None:
        return 0, 0

    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Calculate motion vectors
        motion_vectors = good_new - good_old
        # Get median motion to remove outliers
        if len(motion_vectors) > 0:
            median_motion = np.median(motion_vectors, axis=0)
            return median_motion[0], median_motion[1]

    return 0, 0


# 预测未来位置，考虑最近的运动趋势和加速度，并返回一个置信度分数，在 should_merge_tracks 中使用
def predict_next_position(track: Dict, num_frames: int = 1) -> Tuple[np.ndarray, float]:
    """
    Predict future position based on recent motion with confidence score
    """
    if len(track['centroids']) < 2:
        return np.array(track['centroids'][-1]), 0.5

    # Get recent positions
    recent = np.array(track['centroids'][-5:])
    if len(recent) < 2:
        return recent[-1], 0.5

    # Calculate velocities for each consecutive pair
    velocities = recent[1:] - recent[:-1]

    # Calculate acceleration (change in velocity)
    if len(velocities) > 1:
        accelerations = velocities[1:] - velocities[:-1]
        avg_acceleration = np.mean(accelerations, axis=0)
    else:
        avg_acceleration = np.zeros(2)

    # Get last velocity
    last_velocity = velocities[-1]

    # Predict velocity considering acceleration
    predicted_velocity = last_velocity + (avg_acceleration * num_frames)

    # Predict position using physics equations
    last_pos = recent[-1]
    predicted_pos = last_pos + (predicted_velocity * num_frames) + (0.5 * avg_acceleration * num_frames * num_frames)

    # Calculate prediction confidence based on motion consistency
    velocity_consistency = 1.0 - min(1.0, np.std(velocities) / (np.linalg.norm(last_velocity) + 1e-6))
    temporal_factor = np.exp(-0.1 * num_frames)  # Confidence decreases with prediction distance
    confidence = velocity_consistency * temporal_factor

    return predicted_pos, confidence


# TODO: 这个没有用到，考虑删掉，或者改成在 should_merge_tracks 中使用，作为一个额外的特征来判断两个轨迹是否应该合并
def calculate_appearance_similarity(frame1: np.ndarray, frame2: np.ndarray,
                                    box1: List[float], box2: List[float]) -> float:
    """
    Calculate appearance similarity between two object patches
    """
    try:
        # Extract patches
        x1, y1 = int(box1[0]), int(box1[1])
        x2, y2 = int(box1[2]), int(box1[3])
        patch1 = frame1[y1:y2, x1:x2]

        x1, y1 = int(box2[0]), int(box2[1])
        x2, y2 = int(box2[2]), int(box2[3])
        patch2 = frame2[y1:y2, x1:x2]

        # Resize patches to same size
        size = (64, 64)
        patch1 = cv2.resize(patch1, size)
        patch2 = cv2.resize(patch2, size)

        # Convert to grayscale
        gray1 = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)

        # Calculate histogram similarity
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])

        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return max(0, similarity)  # Ensure non-negative
    except:
        return 0.0


# 评判两个轨迹是否应该合并，考虑时间间隔、空间距离、运动一致性和外观相似度等多个因素
def should_merge_tracks(track1: Dict, track2: Dict,
                        max_frame_gap: int = 15,
                        max_spatial_dist: float = 100.0) -> bool:
    """
    Determine if two tracks should be merged based on multiple metrics
    """
    # Check if same class
    if track1['class'] != track2['class']:
        return False

    # Get end frame of track1 and start frame of track2
    end_frame = track1['frame_nums'][-1]
    start_frame = track2['frame_nums'][0]

    # Check temporal proximity
    frame_gap = start_frame - end_frame
    if frame_gap <= 0 or frame_gap > max_frame_gap:
        return False

    # 1. Check spatial proximity with prediction，先检查空间距离
    predicted_pos, confidence = predict_next_position(track1, frame_gap)
    start_point = np.array(track2['centroids'][0])
    spatial_dist = np.linalg.norm(predicted_pos - start_point)

    # Adjust max_spatial_dist based on prediction confidence and frame gap
    adjusted_max_dist = max_spatial_dist * (1 + (1 - confidence) * 0.5) * (1 + frame_gap / max_frame_gap)

    if spatial_dist > adjusted_max_dist:
        return False

    # 2. Check velocity consistency with more tolerance for short gaps，再检查速度
    if len(track1['centroids']) >= 2 and len(track2['centroids']) >= 2:
        vel1 = np.array(track1['centroids'][-1]) - np.array(track1['centroids'][-2])
        vel2 = np.array(track2['centroids'][1]) - np.array(track2['centroids'][0])
        vel_diff = np.linalg.norm(vel1 - vel2)
        vel_threshold = adjusted_max_dist * 0.75  # More lenient velocity threshold
        if vel_diff > vel_threshold:
            return False

    # 3. Calculate track direction similarity with tolerance，接着是方向
    if len(track1['centroids']) >= 2 and len(track2['centroids']) >= 2:
        dir1 = np.array(track1['centroids'][-1]) - np.array(track1['centroids'][0])
        dir2 = np.array(track2['centroids'][-1]) - np.array(track2['centroids'][0])
        dir_similarity = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2) + 1e-6)
        if dir_similarity < -0.5:  # More lenient direction threshold
            return False

    # Combine all metrics into a score with confidence weighting
    spatial_score = 1.0 - (spatial_dist / adjusted_max_dist)
    temporal_score = 1.0 - (frame_gap / max_frame_gap)

    # Weight the scores based on prediction confidence
    final_score = (0.6 * spatial_score + 0.4 * temporal_score) * (0.5 + 0.5 * confidence)

    # More lenient threshold for short gaps
    threshold = 0.2 if frame_gap <= 3 else 0.3

    return final_score > threshold


def merge_tracks(track1: Dict, track2: Dict) -> Dict:
    """
    Merge two track histories into one.
    """
    # print(f"Merging tracks {track1['centroids']} \n and {track2['centroids']} to get {track1['centroids'] + track2['centroids']}\n")
    return {
        'boxes': track1['boxes'] + track2['boxes'],
        'centroids': track1['centroids'] + track2['centroids'],
        'frame_nums': track1['frame_nums'] + track2['frame_nums'],
        'class': track1['class']  # They should be the same class
    }


def link_broken_tracks(track_histories: Dict[int, Dict],
                       max_frame_gap: int = 15,
                       max_spatial_dist: float = 100.0) -> Dict[int, Dict]:
    """
    Link tracks that likely belong to the same object.

    Args:
        track_histories: Dictionary of track histories
        max_frame_gap: Maximum frame gap to consider for linking
        max_spatial_dist: Maximum spatial distance to consider for linking

    Returns:
        Dictionary of merged track histories
    """

    if len(track_histories.keys()) == 1:
        return track_histories

    # Remove camera from consideration
    camera_track = track_histories.pop('camera', None)

    # Sort tracks by start frame，按照每个 id 轨迹出现的起始帧时间进行排序，不过是改变了 id 的顺序，并没有改变每个 id 轨迹内部的时间顺序
    sorted_tracks = sorted(track_histories.items(),
                           key=lambda x: x[1]['frame_nums'][0])

    merged_tracks = {}
    used_tracks = set()
    next_track_id = max(track_histories.keys()) + 1

    # Try to link tracks
    for i, (track_id1, track1) in enumerate(sorted_tracks):
        if track_id1 in used_tracks:
            continue

        current_track = track1
        used_tracks.add(track_id1)

        # Look for tracks to merge with current_track
        for track_id2, track2 in sorted_tracks[i + 1:]:
            if track_id2 in used_tracks:
                continue

            if should_merge_tracks(current_track, track2, max_frame_gap, max_spatial_dist):
                current_track = merge_tracks(current_track, track2)
                used_tracks.add(track_id2)

        # 这里直接把轨迹用一个新的 id 存储起来，无论是否合并，原来的 id 直接不用了
        merged_tracks[next_track_id] = current_track
        next_track_id += 1

    # Add back camera track if it existed
    if camera_track:
        merged_tracks['camera'] = camera_track

    return merged_tracks


# 这个函数是整个流程的核心，处理视频帧，跟踪对象，分析意图，并返回跟踪历史
def process_video(video_path: str, intent_analyzer: IntentAnalyzer,
                  conf_threshold: float = 0.25) -> Dict[int, Dict]:
    """
    Process video to track objects and analyze their intent
    Returns dictionary mapping object IDs to their tracking histories
    """
    # 这个处理的是一个视频
    cap = cv2.VideoCapture(video_path)

    # Get video dimensions，实际上咱们的视频尺寸就是1920*1080，但是这里读一读也无所谓
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = width * height

    track_histories = {}  # Store tracking histories
    camera_motion = []  # Store camera motion vectors
    prev_frame = None
    frame_count = 0

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()  # ret 就是标志视频有没有损坏或者是读完，frame 就是读取的这一帧
        if not ret:
            break

        # Estimate camera motion if we have a previous frame，这个应该是从第二帧开始，使用光流法根据相邻两帧来估计相机的运动，
        # 得到一个运动向量（dx, dy），然后存储在 camera_motion 列表中
        if prev_frame is not None:
            dx, dy = estimate_camera_motion(prev_frame, frame)
            camera_motion.append((dx, dy))
        prev_frame = frame.copy()

        # Run YOLOv8 tracking for persons
        # 这里的 track 方法本身就是为了单帧设计的，会有方法实现跨帧跟踪，传入 persist=True 就是为了让它在内部维护一个跟踪器，这样就能跨帧跟踪了，
        # conf_threshold 是置信度阈值，classes=[0] 是只检测人类，tracker="botsort.yaml" 是指定使用的跟踪算法
        results = yolo_model.track(frame, persist=True, conf=conf_threshold, classes=[0], verbose=False,
                                   tracker="botsort.yaml")  # person class only

        # person_detections = []  # 看起来每一帧都会重置
        if results[0].boxes.id is not None:  # 如果在这一帧中检测到行人
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # 对于每一帧检测到的所有边界框和跟踪id
            for box, track_id in zip(boxes, track_ids):

                # Convert boxes to xyxy format for person detections
                x, y, w, h = box
                xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]  # 在图像中，y轴坐标是向下的，PIE 使用的也是这样的计算方法
                # person_detections.append((xyxy, track_id))

                # Filter out unrealistic detections，过滤异常框
                box_area = w * h
                if box_area < 0.001 * frame_area or box_area > 0.9 * frame_area:
                    continue

                # 初始化/更新该track_id的轨迹
                if track_id not in track_histories:
                    track_histories[track_id] = {
                        'boxes': [],
                        'centroids': [],
                        'frame_nums': [],
                        'class': 'Pedestrians'
                    }

                # TODO: 这里的 track_id 是直接从 yolov8 里面获取到的，不一定和咱们最后要标注的一致。另外这里的track_histories格式也要进行一定的修改
                track_histories[track_id]['boxes'].append(xyxy)
                track_histories[track_id]['centroids'].append((float(x), float(y)))
                track_histories[track_id]['frame_nums'].append(frame_count)

                # Draw centroid on frame
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.putText(frame, f'ID: {track_id}', (int(x), int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow('Pedestrian Tracking', frame)  # 显示画面
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 Q 退出
                    break  # 退出循环

        # Display frame
        # display_frame_with_grid(frame)
        frame_count += 1

    cap.release()
    # os.remove(video_path)  # 这意思着每处理完一个视频就删除它

    # Add camera motion to the return dictionary
    track_histories['camera'] = {
        'centroids': [(0, 0)],
        'frame_nums': [0],
        'class': 'Camera'
    }

    # Accumulate camera motion
    cam_x, cam_y = 0, 0
    for i, (dx, dy) in enumerate(camera_motion, 1):
        cam_x += dx
        cam_y += dy
        track_histories['camera']['centroids'].append((cam_x, cam_y))
        track_histories['camera']['frame_nums'].append(i)

    # Link broken tracks
    track_histories = link_broken_tracks(track_histories,
                                         max_frame_gap=15,
                                         max_spatial_dist=100.0)

    # Calculate and print displacements
    # 算出来的这个位移有传出去吗？起到一个什么作用呢？可能只是打印检查
    '''displacements = {}
    for track_id, track_data in track_histories.items():
        # print(f"Trackid: {track_id}Centroids: {track_data['centroids']}")
        displacement, start, end, dx, dy = calculate_displacement(track_data['centroids'])
        displacements[track_id] = {
            'class': track_data['class'],
            'displacement': displacement,
            'start_point': track_data['centroids'][0],
            'end_point': track_data['centroids'][-1],
            'start_frame': track_data['frame_nums'][0],
            'end_frame': track_data['frame_nums'][-1],
            'dx': dx,
            'dy': dy,
            'start': start,
            'end': end
        }'''

    # 处理完一个视频后重置跟踪器，避免 ID 混乱
    yolo_model.predictor.trackers[0].reset()
    return track_histories


# 这个函数用于在视频帧上绘制网格和坐标，帮助可视化对象位置和运动轨迹
# TODO: 可视化或许能用到
def display_frame_with_grid(frame: np.ndarray):
    """
    Display the frame with a grid and x, y coordinates using cv2_imshow.
    """
    height, width, _ = frame.shape

    # Define grid step
    grid_step_x = width // 10  # 10 intervals on x axis
    grid_step_y = height // 10  # 10 intervals on y axis

    # Draw grid lines
    for x in range(0, width, grid_step_x):
        cv2.line(frame, (x, 0), (x, height), (255, 255, 255), 1)  # Vertical lines
    for y in range(0, height, grid_step_y):
        cv2.line(frame, (0, y), (width, y), (255, 255, 255), 1)  # Horizontal lines

    # Add x and y axis labels (simple numbering)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for x in range(0, width, grid_step_x):
        cv2.putText(frame, str(x), (x + 2, height - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, height, grid_step_y):
        cv2.putText(frame, str(y), (5, y + 15), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame with the grid
    # cv2_imshow(frame)


# 这个函数实现了一个简单的贪心匹配算法，用于在成本矩阵中找到最佳匹配对，作为匈牙利算法的备选方案
def greedy_match(cost_matrix, track_ids, input_ids):
    # Flatten and sort (i,j) pairs by cost
    flat = [(i, j, c) for i, row in enumerate(cost_matrix)
            for j, c in enumerate(row)]
    flat.sort(key=lambda x: x[2])
    matches, used_t, used_i = {}, set(), set()
    for i, j, c in flat:
        if i not in used_t and j not in used_i:
            matches[track_ids[i]] = input_ids[j]
            used_t.add(i);
            used_i.add(j)
    return matches


# 计算输入框与跟踪框的最佳IoU，用于匹配对象。这对应的是原文中获取到全部数据后和 ground truth 进行匹配，以补全
# TODO: 这个用于计算最大 IoU 的函数估计也得改，因为 input_box 可能不是一个静态框，而是一个轨迹，或许是一个 (N,4) 的数组
def best_iou_batch(track_boxes: np.ndarray, input_box: np.ndarray) -> float:
    # track_boxes: (M,4), input_box: (4,)
    x1 = np.maximum(track_boxes[:, 0], input_box[0])
    y1 = np.maximum(track_boxes[:, 1], input_box[1])
    x2 = np.minimum(track_boxes[:, 2], input_box[2])
    y2 = np.minimum(track_boxes[:, 3], input_box[3])

    inter_w = np.clip(x2 - x1, 0, None)
    inter_h = np.clip(y2 - y1, 0, None)
    inter = inter_w * inter_h

    track_areas = (track_boxes[:, 2] - track_boxes[:, 0]) * (track_boxes[:, 3] - track_boxes[:, 1])
    input_area = (input_box[2] - input_box[0]) * (input_box[3] - input_box[1])

    ious = inter / (track_areas + input_area - inter + 1e-8)
    return float(np.max(ious))


# 匹配跟踪对象和输入框，使用匈牙利算法进行全局最优匹配，并根据IoU阈值过滤匹配结果
# TODO: 这个匹配函数需要修改
def match_objects(type_tracks, input_boxes, iou_thresh=0.3):
    if not type_tracks or not input_boxes:
        return {}
    # prepare data
    input_list, input_ids = [], []
    for bid, bdata in input_boxes.items():
        # 为啥要除以2？可能是因为对方输入框的坐标是按照原始视频尺寸给出的，而跟踪框的坐标是按照处理后的视频尺寸给出的，
        # 所以需要将输入框的坐标缩放到相同的尺度上进行匹配。
        input_list.append(np.array(bdata['Box']) / 2)
        input_ids.append(bid)
    track_ids = list(type_tracks)
    cost = np.zeros((len(track_ids), len(input_list)), dtype=float)

    # build cost matrix
    for i, tid in enumerate(track_ids):
        tb = np.array(type_tracks[tid]['boxes'])  # (M,4)
        for j, ib in enumerate(input_list):
            best = best_iou_batch(tb, ib)
            cost[i, j] = 1 - best

    # if only one candidate, shortcut
    if cost.size == 1:
        return {track_ids[0]: input_ids[0]}

    # Hungarian with fallback
    try:
        # 使用 匈牙利算法 求解二分图的最小权完美匹配，如果行数 > 列数，则会为每个列进行匹配，最后返回的长度应该是列的大小
        r, c = linear_sum_assignment(cost)
        pairs = list(zip(r, c))
    except ValueError:
        pairs = greedy_match(cost, track_ids, input_ids).items()

    # filter by IoU threshold，对于匹配上的再使用 iou_thresh 过滤一遍
    # TODO: 看来这里返回的应该是只有匹配成功的，我们需要考虑多出来的
    matches = {}
    for i, j in pairs:
        if cost[i, j] < 1 - iou_thresh:
            matches[track_ids[i]] = input_ids[j]  # 返回的是个字典，键是 track_histories 的 id，值是 input_id

    return matches


# TODO: 没有用到，考虑删
def find_best_match(tracked_box: List[int], input_boxes: Dict[str, Dict]) -> Optional[str]:
    """Find the best matching input box for a tracked box"""
    best_iou = 0.3  # Lower threshold for matching
    best_id = None

    for obj_id, obj_data in input_boxes.items():
        iou = box_iou(tracked_box, obj_data['Box'])
        if iou > best_iou:
            best_iou = iou
            best_id = obj_id

    return best_id


def convert_bbox_format(bbox):
    """Convert bounding box format if needed"""
    x1, y1 = bbox[0]
    x2, y2 = bbox[2]
    return [x1, y1, x2, y2]


from matplotlib import animation


# 这个函数使用光流法计算视频帧之间的运动，并创建一个动画来可视化这些运动。它可以帮助我们理解视频中对象的动态行为。
# TODO: 或许可以用于可视化检查结果
def animate_optical_flow(video_path, max_frames=100, step=1):
    cap = cv2.VideoCapture(video_path)

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frames = []

    for _ in range(max_frames):
        for _ in range(step):
            ret, frame = cap.read()
            if not ret:
                break
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        amplification = 5.0
        flow *= amplification
        # Calculate magnitude and angle of optical flow
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame)
        hsv[..., 1] = 255
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        frames.append(rgb_flow)
        prev_gray = gray

    cap.release()

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(frames[0])
    ax.axis("off")

    def update(i):
        im.set_array(frames[i])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
    # display(HTML(ani.to_jshtml()))


# 这个函数计算在不同边距下的光流，并返回所有边距的中位数。它可以帮助我们理解在不同空间范围内对象的运动趋势。
def get_median_optical_flow_multiple_margins(video_path, point, box_size=(50, 150),
                                             margins=[50, 70, 100, 120, 150, 170, 200, 50], max_frames=100,
                                             y_shift=False, margin_add=0):
    """
    For each margin, sums per-pixel flow across frames in adjacent box, then computes the median dx, dy.

    Args:
        video_path (str): Path to video file.
        point (tuple): (x, y) reference point.
        box_size (tuple): Width and height of each box.
        margins (list): List of margin values to test.
        max_frames (int): Max number of frames to process.

    Returns:
        overall_median_dx, overall_median_dy: Median of summed per-pixel motion across all margins.
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return None, None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    x, y = point
    bw, bh = box_size
    bw = w
    margin_medians = []

    margins = [x + margin_add for x in margins]
    for i, margin in enumerate(margins):
        if i == len(margins) - 1:
            y_shift = True
        # Decide box direction
        if x < w / 3:
            box_x = x + margin
        elif x > 2 * w / 3:
            box_x = x - margin - bw
        else:
            box_x = x + margin if random.random() > 0.5 else x - margin - bw

        if y_shift:
            box_y = y + margin - bh

        # Clamp to image bounds
        box_x = int(max(0, min(w - bw, box_x)))
        box_y = int(max(0, min(h - bh, y - bh // 2)))

        # Prepare flow accumulator for the box region
        flow_sum = np.zeros((bh, bw, 2), dtype=np.float32)

        # Rewind and process
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        frame_count = 0
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)

            # Extract and accumulate flow in the box
            flow_box = flow[box_y:box_y + bh, box_x:box_x + bw]
            flow_sum += flow_box

            prev_gray = gray
            frame_count += 1

        # Now compute medians from summed flow
        total_dx = flow_sum[..., 0].flatten()
        total_dy = flow_sum[..., 1].flatten()
        median_dx = float(np.median(total_dx))
        median_dy = float(np.median(total_dy))
        print(f"Margin {margin}: Median Total dx = {median_dx:.2f}, dy = {median_dy:.2f}")
        margin_medians.append((median_dx, median_dy))

    cap.release()

    # Final median over margins
    all_dx = [dx for dx, _ in margin_medians]
    all_dy = [dy for _, dy in margin_medians]
    all_dx.sort()
    all_dy.sort()
    overall_median_dx = float(np.mean(all_dx))
    overall_median_dy = float(np.mean(all_dy))

    print(f"\n→ Overall Median dx = {overall_median_dx:.2f}, dy = {overall_median_dy:.2f}")
    return overall_median_dx, overall_median_dy


# 这个函数是上一个函数的简化版本，只计算一个边距下的光流中位数。它可以作为快速评估对象运动趋势的工具。
def get_median_optical_flow(video_path, point, box_h=150, max_frames=100, y_shift=False):
    """
    For each margin, sums per-pixel flow across frames in adjacent box, then computes the median dx, dy.

    Args:
        video_path (str): Path to video file.
        point (tuple): (x, y) reference point.
        box_size (tuple): Width and height of each box.
        margins (list): List of margin values to test.
        max_frames (int): Max number of frames to process.

    Returns:
        overall_median_dx, overall_median_dy: Median of summed per-pixel motion across all margins.
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return None, None

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    x, y = point
    bh = box_h
    bw = w
    box_x = 0

    if y_shift:
        box_y = y + 50 - bh

    # Clamp to image bounds
    box_x = int(max(0, min(w - bw, box_x)))
    box_y = int(max(0, min(h - bh, y - bh // 2)))

    # Prepare flow accumulator for the box region
    flow_sum = np.zeros((bh, bw, 2), dtype=np.float32)

    # Rewind and process
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)

        # Extract and accumulate flow in the box
        flow_box = flow[box_y:box_y + bh, box_x:box_x + bw]
        flow_sum += flow_box

        prev_gray = gray
        frame_count += 1

    # Now compute medians from summed flow
    total_dx = flow_sum[..., 0].flatten()
    total_dy = flow_sum[..., 1].flatten()

    avg_dx_per_col = np.mean(flow_sum[..., 0], axis=0)
    avg_dy_per_col = np.mean(flow_sum[..., 1], axis=0)

    median_dx = float(np.median(total_dx))
    '''
    Uncomment to plot per-column flow

    # Plotting both
    plt.figure()
    plt.plot(avg_dx_per_col, label='Average X Flow')
    plt.plot(avg_dy_per_col, label='Average Y Flow')
    plt.title("Average Optical Flow per Column")
    plt.xlabel("Column Index")
    plt.ylabel("Average Displacement")
    plt.legend()
    plt.grid(True)
    plt.show()
    '''
    positive_dy = avg_dy_per_col[avg_dy_per_col > 0].flatten()
    median_dy = float(np.median(positive_dy))
    '''
    # Plot histogram
    plt.figure()
    plt.hist(positive_dy, bins=30)
    plt.title("Histogram of Positive Y Flow Values")
    plt.xlabel("Y Displacement")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    cap.release()
    '''

    # print(f"\n→ Overall Median dx = {median_dx:.2f}, dy = {median_dy:.2f}")
    return median_dx, median_dy


# 这个函数应该是完整的数据处理流程
def process_dataset(video_root: str, anno_root: str, dataset_filepath: str, original_dataset_filepath: str,
                    output_dir: str, output_json: str, diff_keys=None):
    """Process entire dataset and save results"""

    # 加载路径
    dataset_path = build_dataset(video_root, anno_root)
    # Create output directory，如果存在就跳过，应该只用生成 xml 标注文件
    os.makedirs(output_dir, exist_ok=True)

    # Process each sample
    output_data = {}
    flag = 1
    count = 0
    # 这里是处理前50个样本，并且加进度条 (tqdm)
    # TODO: 总数是53，测试时可以先只用1个
    for sample_id, sample_data in tqdm(islice(dataset_path.items(), len(dataset_path)), desc="Processing samples",
                                       total=len(dataset_path)):

        # 循环每一个视频及其对应标注
        # TODO: 标注文件是XML ETree格式的，需要解析成字典格式
        video_path = sample_data['video_path']
        anno_path = sample_data['anno_path']

        # 获取 PIE 数据集中已经标注好的行人 id, boxes, frame_nums，最终目的是服务于匹配
        output_data[sample_id] = parse_pedestrians(anno_path)

        # Get video dimensions from first frame
        # animate_optical_flow(video_path)
        width = 1920
        height = 1080

        # Initialize intent analyzer with more sensitive threshold
        intent_analyzer = IntentAnalyzer(frame_size=(width, height), motion_threshold=0.15)

        # Track objects in video and get their histories，这里应该是相当于获取行人和相机的轨迹，行人轨迹已经合并了，并且生成的是新的id
        track_histories = process_video(video_path, intent_analyzer)

        # TODO: 对 track_histories 过滤

        # Set camera motion in intent analyzer，只是单纯赋了个值
        if 'camera' in track_histories:
            intent_analyzer.set_camera_motion(track_histories['camera']['centroids'])

        # Plot 3D tracks for this sample，这是在 for 循环里的啊
        plot_3d_tracks(track_histories, f"Object Tracks - Sample {sample_id}")

        # Match objects using Hungarian algorithm
        matches = match_objects(track_histories, output_data)

        # TODO: 这里的处理方式还是按照匹配好了的来，需要添加未匹配的行人轨迹处理，输出似乎也没有包括轨迹边界框
        # Process matches
        for track_id in track_histories.keys():
            track_data = track_histories[track_id]

            # Update track history for intent analysis
            for i, centroid in enumerate(track_data['centroids']):
                if i == 0:
                    # Get camera motion
                    cam_dx, cam_dy = get_median_optical_flow(video_path, (centroid[0], centroid[1]))

                    iter = 0
                    while cam_dy < 0:
                        cam_dx, cam_dy = get_median_optical_flow_multiple_margins(video_path,
                                                                                  (centroid[0], centroid[1]),
                                                                                  y_shift=True,
                                                                                  margin_add=iter * 5)
                        iter += 1
                        if iter > 10:
                            cam_dx, cam_dy = 0, 0
                            break

                # 这个就是简单赋值
                intent_analyzer.update_track_history(track_id, centroid)

            # 这里输出的是 [lateral intent, vertical intent]
            track_histories[track_id]['dir_intent'] = intent_analyzer.determine_intent(track_id, cam_dx, cam_dy)

        # 对方输出的description或许我并不需要
        save_xml_annotation(anno_path, output_dir, track_histories, matches)
        count += 1


# 这个函数是上一个函数的并行版本，使用 multiprocessing 库来加速处理整个数据集。它将每个样本的处理任务分配给多个进程，以提高效率。
def process_dataset_parallel(dataset_filepath: str, original_dataset_filepath: str, output_dir: str, output_json: str):
    """Process entire dataset and save results"""
    # Load dataset
    dataset = load_json_data(dataset_filepath)
    original_dataset = load_json_data(original_dataset_filepath)
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Prepare items, skipping the first 550 just like `if count < 550: continue`
    items = list(islice(dataset.items(), 0, 5686))
    output_data = {}
    count = 0

    def worker(item):
        sample_id, sample_data = item

        # Copy original data
        local_output = {sample_id: sample_data.copy()}

        # Find corresponding annotation
        annotation_index = next(
            (i for i, ann in enumerate(original_dataset)
             if ann.get('s3_fileUrl') == sample_data['image_path']),
            None
        )
        if annotation_index is None:
            print(f"No annotation found for frame {sample_id}")
            return None

        # Process pedestrian annotations
        if (original_dataset[annotation_index].get('Agent-classifier') == 'Pedestrian' and
                original_dataset[annotation_index].get('pedestrian_motion_direction') not in ["N/A", []]):
            local_output[sample_id]['Pedestrians'][str(len(sample_data['Pedestrians']) + 1)] = {
                "Box": convert_bbox_format(original_dataset[annotation_index].get('geometry')),
                "Intent": original_dataset[annotation_index].get('pedestrian_motion_direction')[0]
            }

        # Process cyclist annotations
        if (original_dataset[annotation_index].get('Agent-classifier') == 'Cyclist' and
                original_dataset[annotation_index].get('pedestrian_motion_direction') not in ["N/A", []]):
            local_output[sample_id]['Cyclists'][str(len(sample_data['Cyclists']) + 1)] = {
                "Box": convert_bbox_format(original_dataset[annotation_index].get('geometry')),
                "Intent": original_dataset[annotation_index].get('pedestrian_motion_direction')[0]
            }

        try:
            # Get video dimensions from first frame
            video_path = download_video(local_output[sample_id]['video_path'])
            # animate_optical_flow(video_path)
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Initialize intent analyzer with more sensitive threshold
            intent_analyzer = IntentAnalyzer(frame_size=(width, height), motion_threshold=0.15)

            # Track objects in video and get their histories
            track_histories = process_video(local_output[sample_id]['video_path'], intent_analyzer)

            # Set camera motion in intent analyzer
            if 'camera' in track_histories:
                intent_analyzer.set_camera_motion(track_histories['camera']['centroids'])

            # Plot 3D tracks for this sample
            # plot_3d_tracks(track_histories, f"Object Tracks - Sample {sample_id}")

            # Process each object type separately
            for object_type in ['Pedestrians', 'Cyclists']:
                if object_type not in local_output[sample_id]:
                    continue

                # Get tracked objects of this type
                type_tracks = {k: v for k, v in track_histories.items()
                               if v['class'] == object_type}
                if not type_tracks:
                    continue

                # Get final boxes for matching
                track_ids = list(type_tracks.keys())
                tracked_boxes = [track_data['boxes'][-1] for track_data in type_tracks.values()]

                # Remove duplicate boxes from input data
                filtered_input_boxes = remove_duplicate_boxes(local_output[sample_id][object_type])

                # Match objects using Hungarian algorithm
                matches = match_objects(type_tracks, filtered_input_boxes)
                # Process matches
                for track_idx, obj_id in matches.items():
                    track_id = track_idx
                    track_data = type_tracks[track_id]

                    # Update track history for intent analysis
                    for i, centroid in enumerate(track_data['centroids']):
                        if i == 0:
                            cam_dx, cam_dy = get_median_optical_flow(video_path, (centroid[0], centroid[1]))

                            iter = 0
                            while cam_dy < 0:
                                cam_dx, cam_dy = get_median_optical_flow_multiple_margins(
                                    video_path,
                                    (centroid[0], centroid[1]),
                                    y_shift=True,
                                    margin_add=iter * 5
                                )
                                iter += 1
                                if iter > 10:
                                    cam_dx, cam_dy = 0, 0
                                    break
                        # Get camera motion

                        intent_analyzer.update_track_history(track_id, centroid)

                    # Analyze intent
                    intent = intent_analyzer.determine_intent(track_id, cam_dx, cam_dy)
                    position = intent_analyzer.determine_position(track_data['centroids'][-1])

                    # Update output data
                    local_output[sample_id][object_type][obj_id].update({
                        'Intent': intent,
                        'Position': position,
                        'Description': intent_analyzer.generate_description(intent)
                    })

            return local_output

        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            print(sample_data)

    # Execute in parallel
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(worker, items),
                           total=len(items),
                           desc="Processing samples"):
            if result:
                output_data.update(result)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} samples")
                save_json_data(output_data, output_json)

    # Save final results
    save_json_data(output_data, output_json)


def box_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate IoU between two boxes"""
    # Convert to x1,y1,x2,y2 format if needed
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - inter

    return inter / union if union > 0 else 0


def plot_3d_tracks(track_histories: Dict[int, Dict], title: str = "Object Tracks in 3D"):
    """
    Create a 3D plot of object tracks with time as the third dimension.

    Args:
        track_histories: Dictionary mapping track_id to track data containing centroids
        title: Title for the plot
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # First plot camera motion if available
    if 'camera' in track_histories:
        camera_data = track_histories.pop('camera')
        camera_points = np.array(camera_data['centroids'])
        if len(camera_points) > 1:
            ax.plot(camera_points[:, 0], camera_points[:, 1],
                    camera_data['frame_nums'],  # Use actual frame numbers
                    c='red', linewidth=2, label='Camera Motion',
                    linestyle='--')

    # Then plot object tracks
    colors = plt.cm.rainbow(np.linspace(0, 1, len(track_histories)))

    for (track_id, track_data), color in zip(track_histories.items(), colors):
        centroids = track_data.get('centroids', [])
        frame_nums = track_data.get('frame_nums', [])  # Get frame numbers
        if len(centroids) < 2:
            continue

        # Convert centroids to numpy arrays
        points = np.array(centroids)
        xs = points[:, 0]
        ys = points[:, 1]

        # Plot the track
        ax.plot(xs, ys, frame_nums, c=color,
                label=f'Track {track_id} ({track_data["class"]}) [F{frame_nums[0]}-{frame_nums[-1]}]')
        # Plot start point
        ax.scatter(xs[0], ys[0], frame_nums[0], c=color, marker='o')
        # Plot end point
        ax.scatter(xs[-1], ys[-1], frame_nums[-1], c=color, marker='^')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Frame Number')
    # ax.set_title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Track persons and analyze their intent in videos')
    parser.add_argument('--dataset_filepath', type=str, required=True,
                        help='Path to the input JSON file containing video URLs')
    parser.add_argument('--original_dataset_filepath', type=str, required=True,
                        help='Path to the original DRAMA dataset JSON file containing video URLs')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory for output videos')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Path for output JSON with tracking and intent data')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold for detections')
    parser.add_argument('--parallel', type=bool, default=False,
                        help='True for multithreading')

    args = parser.parse_args()
    if args.parallel:
        process_dataset_parallel(args.dataset_filepath, args.original_dataset_filepath, args.output_dir,
                                 args.output_json)
    else:
        process_dataset(args.dataset_filepath, args.original_dataset_filepath, args.output_dir, args.output_json)


if __name__ == "__main__":
    main()
