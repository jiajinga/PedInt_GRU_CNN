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
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import random
import os
from itertools import islice
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import torch
from multiprocessing import Pool, cpu_count
import xml.etree.ElementTree as ET  # 用于解析、编辑 XML 文件
import json


# Check for GPU availability
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Using device: {device}")

# Initialize models
#yolo_model = YOLO('yolov8s.pt')


# ---------------------- 输入输出模块 --------------------------
def build_dataset(video_root, anno_root, vehicle_root):
    """数据集加载方式，适配 PIE，获取了系列 ID 视频路径 标注路径 车辆标注路径"""
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

    for set_name in sorted(os.listdir(vehicle_root)):
        set_id = set_name[-1] + "_"
        set_path = os.path.join(vehicle_root, set_name)

        for v_name in sorted(os.listdir(set_path)):
            if not v_name.endswith(".xml"):
                continue
            sample_id = set_id + v_name.split(".")[0][8:10]
            v_path = os.path.join(set_path, v_name)
            samples[sample_id]["vehicle_path"] = v_path

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


def parse_vehicle(vehicle_path: str) -> Dict[int, Dict[str, float]]:
    """
    从标注 XML 文件中提取所有 vehicle 运动信息

    Args:
        vehicle_path: XML 文件路径

    Returns:
        字典，键为帧，值包含以下字段：
            - "GPS_speed", "OBD_speed", "latitude", "longitude", "heading_angle"
    """
    tree = ET.parse(vehicle_path)
    root = tree.getroot()
    vehicle = {}

    for track in root.findall('frame'):
        vehicle[int(track.get('id'))] = {
            "GPS_speed": float(track.get('GPS_speed', 0.0)),
            "OBD_speed": float(track.get('OBD_speed', 0.0)),
            "latitude": float(track.get('latitude', 0.0)),
            "longitude": float(track.get('longitude', 0.0)),
            "heading_angle": float(track.get('heading_angle', 0.0)),
            "pitch": float(track.get('pitch', 0.0)),
            "roll": float(track.get('roll', 0.0)),
            "yaw": float(track.get('yaw', 0.0))}

    return vehicle


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

    anno_path_normalized = os.path.normpath(anno_path)  # 为了避免分割出错
    set_num = anno_path_normalized.split(os.sep)[-2][-1]
    # TODO: 这个编号有问题，虽然的确是video的编号，但实际PIE用的是最后一位（对于单位数）
    v_num = anno_path_normalized.split(os.sep)[-1].split(".")[0][8:10]

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
    output_path = os.path.join(output_root, anno_path_normalized.split(os.sep)[-2],
                               anno_path_normalized.split(os.sep)[-1])
    os.makedirs(os.path.join(output_root, anno_path.split(os.sep)[-2]), exist_ok=True)

    # 保存修改后的 XML 文件
    tree.write(output_path)
    print(f"new annotations have been saved to {output_path}")


def convert_to_serializable(obj):
    """递归转换 numpy/tensor 为 Python 原生类型"""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()  # 或 obj.item() 若为标量
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj


def save_results_to_json(results, output_json_path):
    serializable = convert_to_serializable(results)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_json_path}")


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------- 轨迹连接模块 ----------------------------------
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

    return merged_tracks


# -------------------- 模块整合（检测、追踪、连接、过滤） -----------------------
# 这个函数是整个流程的核心，处理视频帧，跟踪对象，分析意图，并返回跟踪历史
# 最短的 GT 跟踪轨迹时长是31帧
def process_video(video_path: str,
                  conf_threshold: float = 0.25,
                  time_thresh: int = 30) -> (Dict[int, Dict], Dict[int, List]):
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
    frame_count = 0

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()  # ret 就是标志视频有没有损坏或者是读完，frame 就是读取的这一帧
        if not ret:
            break

        results = yolo_model.track(frame, persist=True, conf=conf_threshold, classes=[0], verbose=False,
                                   tracker="botsort.yaml")  # person class only

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                xyxy = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]

                box_area = w * h
                if box_area < 0.001 * frame_area or box_area > 0.9 * frame_area:
                    continue

                if track_id not in track_histories:
                    track_histories[track_id] = {
                        'boxes': [],
                        'centroids': [],
                        'frame_nums': [],
                        'class': 'Pedestrians'
                    }

                track_histories[track_id]['boxes'].append(xyxy)
                track_histories[track_id]['centroids'].append([float(x), float(y)])
                track_histories[track_id]['frame_nums'].append(frame_count)

        # Display frame
        # display_frame_with_grid(frame)
        frame_count += 1

    cap.release()

    # 保存结果
    save_results_to_json(track_histories, r"yolo_tracks.json")
    print(f"轨迹已全部收集完成，总计{len(track_histories)}")

    # Link broken tracks
    # TODO: 如有需要，可以修改 link_broken_tracks 的参数
    track_histories = link_broken_tracks(track_histories,
                                         max_frame_gap=15,
                                         max_spatial_dist=100.0)
    save_results_to_json(track_histories, r"linked_tracks.json")
    print(f"轨迹已连接，现有轨迹数量{len(track_histories)}")

    # TODO: 对 track_histories 过滤
    short_track_id = []
    for i, data in track_histories.items():
        if len(data['frame_nums']) < time_thresh:
            short_track_id.append(i)

    for i in short_track_id:
        del track_histories[i]
    save_results_to_json(track_histories, r"filtered_tracks.json")
    print(f"过滤短时轨迹，数量为{len(short_track_id)}")

    # 处理完一个视频后重置跟踪器，避免 ID 混乱
    yolo_model.predictor.trackers[0].reset()
    return track_histories


# -------------------------- 轨迹匹配模块 -------------------------------
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
            used_t.add(i)
            used_i.add(j)
    return matches


def box_iou(box1: List, box2: List) -> float:
    """计算两个框的 IoU """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def compute_tube_iou(track, input, min_overlap_frames=5):
    """计算两个轨迹在共同帧内的平均 IoU。如果重叠帧数太少，直接返回 0。"""
    # 建立帧到框的映射
    if len(track['frame_nums']) != len(track['boxes']):
        raise ValueError(f"Length mismatch: frames={len(track['frame_nums'])}, "
                         f"boxes={len(track['boxes'])}")
    track_dict = dict(zip(track['frame_nums'], track['boxes']))
    input_dict = dict(zip(input['frame_nums'], input['boxes']))

    # 找到它们共同存在的帧
    common_frames = set(input['frame_nums']).intersection(set(track['frame_nums']))

    # 如果重叠时间太短（比如连 5 帧都不到），认为不匹配，排除假阳性
    if len(common_frames) < min_overlap_frames:
        return 0.0

    iou_sum = 0.0
    for f in common_frames:
        # 获取同一帧下，两者 box IoU
        iou_sum += box_iou(track_dict[f], input_dict[f])

    # 返回重叠区间内的平均 IoU
    return iou_sum / len(common_frames)


# 匹配跟踪对象和输入框，使用匈牙利算法进行全局最优匹配，并根据IoU阈值过滤匹配结果
# TODO: 如有需要，可以修改匹配参数 iou_thresh, min_overlap_frames
def match_objects(tracks, inputs, iou_thresh=0.3, min_overlap_frames=5):
    if not tracks or not inputs:
        raise ValueError("没有轨迹或者输入")

    track_ids = list(tracks.keys())
    input_ids = list(inputs.keys())
    # prepare data
    cost = np.zeros((len(tracks), len(inputs)), dtype=float)  # 字典的长度实际上就是键的长度

    # build cost matrix
    for i, tid in enumerate(track_ids):
        for j, ib in enumerate(input_ids):
            avg_iou = compute_tube_iou(tracks[tid], inputs[ib], min_overlap_frames)
            cost[i, j] = 1 - avg_iou

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
    matches = {}
    # pairs_iou = []
    for i, j in pairs:
        # pairs_iou.append(1-cost[i,j])
        if cost[i, j] < 1 - iou_thresh:
            matches[track_ids[i]] = input_ids[j]  # 返回的是个字典，键是 track_histories 的 id，值是 input_id
    print(f"轨迹已完成匹配，匹配数量为{len(matches)}")
    print(matches)
    return matches  # , pairs, pairs_iou


# ---------------------------------- 相机运动 + 意图定义模块 -----------------------------------
def _safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _normalize_heading_deg(h):
    """Normalize heading angle to [0, 360)."""
    h = float(h) % 360.0
    if h < 0:
        h += 360.0
    return h


def _latlon_to_local_xy_m(lat, lon, lat0, lon0):
    """
    Equirectangular approximation from lat/lon to local ENU-like meters.
    Returns (x_east, y_north) in meters relative to (lat0, lon0).
    """
    # Earth radius (m)
    r = 6378137.0
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    lat0 = np.deg2rad(lat0)
    lon0 = np.deg2rad(lon0)

    x = (lon - lon0) * np.cos(0.5 * (lat + lat0)) * r
    y = (lat - lat0) * r
    return float(x), float(y)


def load_camera_calibration(calibration_json_path: str) -> Dict[str, Any]:
    calib = load_json(calibration_json_path)
    k = np.array(calib.get("K", []), dtype=np.float64)
    if k.shape != (3, 3):
        raise ValueError(f"Invalid K shape: {k.shape}")

    d_raw = np.array(calib.get("D", []), dtype=np.float64).reshape(-1)
    if d_raw.size not in (4, 5, 8):
        raise ValueError(f"Invalid D length: {d_raw.size}, expected 4/5/8")

    dim = calib.get("dim", [1920, 1080])
    if not isinstance(dim, list) or len(dim) != 2:
        dim = [1920, 1080]

    return {
        "K": k,
        "D": d_raw,
        "dim": (int(dim[0]), int(dim[1])),
        "cam_height_mm": _safe_float(calib.get("cam_height_mm", 1270.0), 1270.0),
        "cam_pitch_deg": _safe_float(calib.get("cam_pitch_deg", -10.0), -10.0),
    }


# 传进来一组图像坐标系下的点（矩阵），返回去畸变后的图像坐标系下的点（矩阵）
def _undistorted_points_to_pixel(points_xy: np.ndarray, k: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    points_xy: (N,2) distorted pixel
    return: (N,2) undistorted pixel in same pixel coordinate system (P=K)
    """
    pts = points_xy.reshape(-1, 1, 2).astype(np.float64)
    if d.size == 4:
        und = cv2.fisheye.undistortPoints(pts, k, d.reshape(4, 1), P=k)
    else:
        und = cv2.undistortPoints(pts, k, d.reshape(-1, 1), P=k)
    return und.reshape(-1, 2)


# 这里是在计算畸变体系下的相邻两帧之间的相机位移
def build_camera_displacements(
        vehicle_annotations: Dict[int, Dict[str, float]],
        calibration_json_path: str,
        max_speed_mps: float = 35.0,
        smooth_window: int = 5) -> Dict[int, Tuple[float, float]]:
    """
    Build per-frame camera displacement in pixel space from PIE vehicle annotations.

    Input:
        vehicle_annotations:
            dict keyed by frame id (string/int), each value with possible keys:
            GPS_speed, OBD_speed, heading_angle, latitude, longitude.
        video_path: to estimate FPS and dt.
        calibration_json_path: path to calibration_data.json.
    Output:
        {frame_idx: (dx_px, dy_px)}  where frame_idx is current frame id,
        and displacement corresponds roughly to prev_frame -> current_frame motion.

    Notes:
        - Robust to missing tail frames: outputs only available IDs; callers can forward-fill zeros.
        - Robust to missing lat/lon: fallback to speed+heading integration.
        - Uses K/fx/fy + camera height/pitch for a coarse ground-plane scale (annotation-grade accuracy).
    输出: {frame_id: (dx_u, dy_u)}，北向(x)和东向(y)的物理位移。
    """
    if not vehicle_annotations:
        raise ValueError("Missing vehicle annotations")

    # Sort by frame id (id is frame index per your note)
    frame_ids = sorted(int(x) for x in vehicle_annotations.keys())
    if len(frame_ids) < 2:
        return {}

    # Pre-read rows，获取用来计算相机运动的参数
    rows = []
    for fid in frame_ids:
        # 从 vehicle_annotations 里取当前帧 fid 对应的标注行
        r = vehicle_annotations.get(fid, {})
        gps_speed = _safe_float(r.get("GPS_speed", 0.0), 0.0)
        obd_speed = _safe_float(r.get("OBD_speed", 0.0), 0.0)
        speed = gps_speed if gps_speed > 1e-3 else obd_speed
        speed = min(max(speed, 0.0), max_speed_mps)

        heading = _normalize_heading_deg(_safe_float(r.get("heading_angle", 0.0), 0.0))
        lat = r.get("latitude", None)
        lon = r.get("longitude", None)
        lat = None if lat is None else _safe_float(lat, None)
        lon = None if lon is None else _safe_float(lon, None)

        rows.append({"fid": fid, "speed": speed, "heading": heading, "lat": lat, "lon": lon})

    # Reference for local XY，专门把经纬度提取出来
    valid_geo = [(r["lat"], r["lon"]) for r in rows if r["lat"] is not None and r["lon"] is not None]
    lat0, lon0 = valid_geo[0] if valid_geo else (None, None)

    prev = rows[0]
    prev_xy = None
    if lat0 is not None and prev["lat"] is not None and prev["lon"] is not None:
        prev_xy = _latlon_to_local_xy_m(prev["lat"], prev["lon"], lat0, lon0)

    # Build meter displacement in vehicle-forward/lateral approx
    meter_disp = {}  # fid -> (d_right_m, d_forward_m), from prev->curr
    for i in range(1, len(rows)):
        cur = rows[i]
        # d_forward, d_right = 0.0, 0.0
        dx_n, dy_e = 0.0, 0.0

        # 1) Try geo delta first
        if (lat0 is not None and prev["lat"] is not None and prev["lon"] is not None
                and cur["lat"] is not None and cur["lon"] is not None):
            cur_xy = _latlon_to_local_xy_m(cur["lat"], cur["lon"], lat0, lon0)
            dy_e = cur_xy[0] - prev_xy[0]  # east
            dx_n = cur_xy[1] - prev_xy[1]  # north

            '''
            # Rotate world EN to vehicle frame using previous heading，坐标系变换
            # heading: 0=north, 90=east
            hd = np.deg2rad(prev["heading"])
            fwd_x_e, fwd_y_n = np.sin(hd), np.cos(hd)
            right_x_e, right_y_n = np.cos(hd), -np.sin(hd)

            d_forward = dx_e * fwd_x_e + dy_n * fwd_y_n
            d_right = dx_e * right_x_e + dy_n * right_y_n'''
            prev_xy = cur_xy
        else:
            # 2) Fallback to speed * dt and heading change ignored for translation decomposition
            # 速度单位是 km/h，转换成 m/s 需要除以 3.6；再除以帧率（30）得到每帧的位移
            d_forward = prev["speed"] / 3.6 / 30.0
            d_right = 0.0

        meter_disp[cur["fid"]] = (float(dx_n), float(dy_e))
        prev = cur

    # Smooth meter displacement (simple moving average)，数据平滑，避免单帧异常值对后续像素位移计算的过大影响；
    # 平滑窗口过大会过度模糊，过小则无法有效抑制噪声
    if smooth_window > 1 and len(meter_disp) >= 3:
        keys = sorted(meter_disp.keys())
        arr = np.array([meter_disp[k] for k in keys], dtype=np.float64)
        pad = smooth_window // 2
        arr_pad = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        sm = np.array([np.mean(arr_pad[i:i + smooth_window], axis=0) for i in range(len(arr))], dtype=np.float64)
        for k_id, v in zip(keys, sm):
            meter_disp[k_id] = (float(v[0]), float(v[1]))

    '''
    # Convert meter displacement -> pixel displacement
    # Coarse scale:
    # - lateral: approximately fx * dx / Zref
    # - forward: approximately fy * dy / Zref (sign adjusted to image y-down convention)
    # Use Zref from camera height and pitch; keep conservative.
    calib = load_camera_calibration(calibration_json_path)
    # 相机的内参矩阵 intrinsic matrix
    k = calib["K"]
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    h_cam_m = calib["cam_height_mm"] / 1000.0
    pitch_deg = calib["cam_pitch_deg"]
    pitch_rad = np.deg2rad(abs(pitch_deg))
    z_ref = max(8.0, h_cam_m / max(np.tan(max(pitch_rad, np.deg2rad(2.0))), 1e-3))

    undisp = {}
    for fid, (d_right_m, d_forward_m) in meter_disp.items():
        dx_u = fx * (d_right_m / z_ref)
        dy_u = fy * (d_forward_m / z_ref)
        undisp[int(fid)] = (float(dx_u), float(dy_u))
    '''

    return meter_disp


def compute_pixel_displacement_pie(
        t_veh: np.ndarray,  # [dx_n, dy_n, 0] 车辆坐标系位移（米），前一帧和当前帧之间的位移
        heading: float,  # 前一帧 heading_angle（度）
        delta_heading: float,  # 当前帧与上一帧的 heading_angle 变化（度）
        cam_pitch_deg: float,  # 相机 pitch_angle（度）
        K: np.ndarray,
        Ki: np.ndarray,
        cam_height_m: float,
        ref_point_uv: tuple = None
) -> tuple:
    cx = K[0, 2]
    cy = K[1, 2]
    if ref_point_uv is None:
        u0, v0 = cx, cy
    else:
        u0, v0 = ref_point_uv

    # 从世界坐标系（北东地）转换到第一帧的相机坐标系的旋转矩阵
    R1 = np.array([[1, 0, 0],
                   [0, np.cos(th_x := np.deg2rad(90 - cam_pitch_deg)), -np.sin(th_x)],
                   [0, np.sin(th_x), np.cos(th_x)]])
    R2 = np.array([[np.cos(th_y := np.deg2rad(90 + heading)), 0, np.sin(th_y)],
                   [0, 1, 0],
                   [-np.sin(th_y), 0, np.cos(th_y)]])
    R3 = np.array([[np.cos(th_y := np.deg2rad(delta_heading)), 0, np.sin(th_y)],
                   [0, 1, 0],
                   [-np.sin(th_y), 0, np.cos(th_y)]])
    R_wc1 = R1 @ R2
    R_rel = R_wc1 @ R3 @ R_wc1.T  # 从第一帧相机坐标系到当前帧相机坐标系的旋转矩阵

    t = np.array([-t_veh[0], -t_veh[1], 0]).reshape(-1, 1)
    T_rel = R_wc1 @ R3 @ t

    p1 = np.array([u0, v0, 1]).reshape(-1, 1)
    n_w = np.array([0, 0, 1]).reshape(-1, 1)
    n_c1 = R_wc1 @ n_w
    s1 = cam_height_m / (n_c1.T @ Ki @ p1)

    p2 = K @ (R_rel @ (s1 * Ki @ p1) + T_rel)
    u1 = p2[0, 0] / p2[2, 0]
    v1 = p2[1, 0] / p2[2, 0]
    return float(u1 - u0), float(v1 - v0)


'''
    # 1. 构造世界到相机坐标系的旋转矩阵（使用 heading, pitch, roll）
    # R_wc = euler_to_rot_matrix_from_imu(pitch_deg, roll_deg, heading_deg)
    # TODO: 从世界坐标系（车辆）转换到相机坐标系的旋转矩阵，旋转角为 90 + camera_pitch_deg，绕 x 轴旋转，这里直接硬编码了
    R_wc = np.array([[1, 0, 0],
                    [0, np.cos(np.deg2rad(90-cam_pitch_deg)), -np.sin(np.deg2rad(90-cam_pitch_deg))],
                    [0, np.sin(np.deg2rad(90-cam_pitch_deg)), np.cos(np.deg2rad(90-cam_pitch_deg))]])

    # 2. 地平面在世界坐标系中为 z=0，法向量 n_w = [0,0,1]^T，距离 d = cam_height_m
    n_world = np.array([0, 0, 1])
    n_cam = R_wc @ n_world
    d_cam = cam_height_m

    # 3. 车辆平移转换到相机坐标系
    t_cam = R_wc @ np.array([t_veh[0], t_veh[1], t_veh[2]])

    # 4. 帧间相对旋转：假设相邻帧姿态变化很小，忽略旋转（R_rel = I）
    R_rel = np.eye(3)

    # 5. 单应矩阵
    M = R_rel - (t_cam[:, None] @ n_cam[None, :]) / -d_cam
    H = K @ M @ Ki

    # 6. 计算参考点位移
    p0 = np.array([u0, v0, 1.0])
    p1 = H @ p0
    u1 = p1[0] / p1[2]
    v1 = p1[1] / p1[2]
    return float(u1 - u0), float(v1 - v0)
'''


def build_camera_displacements_corrected(
        vehicle_annotations: Dict[int, Dict[str, float]],
        calibration_json_path: str,
        max_speed_mps: float = 35.0,
        smooth_window: int = 5
) -> Dict[int, Tuple[float, float]]:
    """
    修正版本：正确使用物理位移 + 姿态角计算像素位移。
    输出格式与原函数一致：{frame_id: (dx_px, dy_px)}
    """
    # ---- 数据解析部分（与原函数相同） ----
    if not vehicle_annotations:
        raise ValueError("Missing vehicle annotations")

    frame_ids = sorted(int(x) for x in vehicle_annotations.keys())
    if len(frame_ids) < 2:
        return {}

    rows = []
    for fid in frame_ids:
        r = vehicle_annotations.get(fid, {})
        gps_speed = _safe_float(r.get("GPS_speed", 0.0), 0.0)
        obd_speed = _safe_float(r.get("OBD_speed", 0.0), 0.0)
        speed = gps_speed if gps_speed > 1e-3 else obd_speed
        speed = min(max(speed, 0.0), max_speed_mps)

        heading = _normalize_heading_deg(_safe_float(r.get("heading_angle", 0.0), 0.0))
        # 假设标注中也有 pitch, roll
        pitch = _safe_float(r.get("pitch", 0.0), 0.0)
        roll = _safe_float(r.get("roll", 0.0), 0.0)

        lat = r.get("latitude", None)
        lon = r.get("longitude", None)
        lat = None if lat is None else _safe_float(lat, None)
        lon = None if lon is None else _safe_float(lon, None)

        rows.append({
            "fid": fid,
            "speed": speed,
            "heading": heading,
            "pitch": pitch,
            "roll": roll,
            "lat": lat,
            "lon": lon
        })

    # ---- 计算相邻帧间的车辆坐标系位移 (d_right, d_forward) ----
    # 原函数中这部分逻辑保留，但要注意它已经用了 heading 旋转世界位移到车辆系。
    valid_geo = [(r["lat"], r["lon"]) for r in rows if r["lat"] is not None and r["lon"] is not None]
    lat0, lon0 = valid_geo[0] if valid_geo else (None, None)

    prev = rows[0]
    prev_xy = None
    if lat0 is not None and prev["lat"] is not None and prev["lon"] is not None:
        prev_xy = _latlon_to_local_xy_m(prev["lat"], prev["lon"], lat0, lon0)

    meter_disp = {}  # fid -> (d_right_m, d_forward_m)
    for i in range(1, len(rows)):
        cur = rows[i]
        # d_forward, d_right = 0.0, 0.0
        dx_n, dy_e = 0.0, 0.0

        # 1) 尝试用 GPS 计算位移（世界坐标系 ENU）
        if (lat0 is not None and prev["lat"] is not None and prev["lon"] is not None
                and cur["lat"] is not None and cur["lon"] is not None):
            cur_xy = _latlon_to_local_xy_m(cur["lat"], cur["lon"], lat0, lon0)
            dy_e = cur_xy[0] - prev_xy[0]  # 东
            dx_n = cur_xy[1] - prev_xy[1]  # 北

            # 用前一帧 heading 将 ENU 位移旋转到车辆坐标系
            '''hd = np.deg2rad(prev["heading"])
            fwd_x_e, fwd_y_n = np.sin(hd), np.cos(hd)   # 北为0°，顺时针增加
            right_x_e, right_y_n = np.cos(hd), -np.sin(hd)

            d_forward = dx_e * fwd_x_e + dy_n * fwd_y_n
            d_right = dx_e * right_x_e + dy_n * right_y_n'''
            prev_xy = cur_xy
        else:
            # 2) 无 GPS，用速度积分
            d_forward = prev["speed"] / 3.6 / 30.0  # 假设 30fps
            d_right = 0.0

        # meter_disp[cur["fid"]] = (float(d_right), float(d_forward))
        meter_disp[cur["fid"]] = (float(dx_n), float(dy_e))
        prev = cur

    # ---- 平滑处理（可选） ----
    if smooth_window > 1 and len(meter_disp) >= 3:
        keys = sorted(meter_disp.keys())
        arr = np.array([meter_disp[k] for k in keys], dtype=np.float64)
        pad = smooth_window // 2
        arr_pad = np.pad(arr, ((pad, pad), (0, 0)), mode="edge")
        sm = np.array([np.mean(arr_pad[i:i + smooth_window], axis=0) for i in range(len(arr))])
        for k_id, v in zip(keys, sm):
            meter_disp[k_id] = (float(v[0]), float(v[1]))

    # ---- 加载相机参数 ----
    calib = load_camera_calibration(calibration_json_path)
    K = calib["K"]
    Ki = np.linalg.inv(K)
    cam_height_m = calib["cam_height_mm"] / 1000.0
    cam_pitch_deg = np.deg2rad(calib["cam_pitch_deg"])

    # ---- 计算每帧的像素位移（使用修正后的方法） ----
    undisp = {}
    for fid, (d_north, d_east) in meter_disp.items():
        # 获取当前帧的姿态（如果每帧都有）
        cur_row = next((r for r in rows if r["fid"] == fid), None)
        if cur_row is None:
            continue

        # 车辆坐标系平移（无垂直位移）
        # t_veh = np.array([d_right, d_forward, 0.0])
        t_veh = np.array([d_north, d_east, 0.0])

        du, dv = compute_pixel_displacement_pie(
            t_veh=t_veh,
            heading=rows[fid - 1]['heading'],
            delta_heading=rows[fid]['heading'] - rows[fid - 1]['heading'],
            cam_pitch_deg=cam_pitch_deg,
            K=K,
            Ki=Ki,
            cam_height_m=cam_height_m,  #
            ref_point_uv=(1755, 832)  #   # TODO: 修改为行人轨迹的初始边界框的右下坐标  1312, 823  1439,810  207,846
        )

        undisp[int(fid)] = (du, dv)

    return undisp


def get_pedestrian_real_displacement_endpoints(
        ped_track: Dict[str, Any],
        camera_displacements: Dict[int, Tuple[float, float]],
        calibration_json_path: str,
) -> Dict[str, Any]:
    """
    输入：ped_track 一个行人的轨迹
         camera_displacements 相机的所有逐帧物理位移
         calibration_json_path 相机参数文件路径
    仅首尾点（省算力）:
      1) 首尾行人质心去畸变
      2) 累积同时间段相机逐帧位移（无畸变域）
      3) 相减得到相对道路位移
    """
    # TODO: 后续打算给相机位移也增加首尾可选项，但目前先默认从第一帧开始累积，直到轨迹结束（如果中间有缺帧，则补0），以保证相机位移的完整性
    frame_nums = [int(f) for f in ped_track.get("frame_nums", [])]
    cents = ped_track.get("centroids", [])
    if len(frame_nums) != len(cents):
        raise ValueError(f"轨迹帧数和位置数不一致: frames={len(frame_nums)}, centroids={len(cents)}")

    s_f, e_f = frame_nums[0], frame_nums[-1]
    s_p = np.array(cents[0], dtype=np.float64).reshape(1, 2)
    e_p = np.array(cents[-1], dtype=np.float64).reshape(1, 2)

    calib = load_camera_calibration(calibration_json_path)
    k, d = calib["K"], calib["D"]

    s_u = _undistorted_points_to_pixel(s_p, k, d)[0]
    e_u = _undistorted_points_to_pixel(e_p, k, d)[0]

    # 相机按帧累积（支持轨迹从任意帧开始；有时候车辆标注信息可能不完整，缺帧补0）
    low, high = min(s_f, e_f), max(s_f, e_f)
    cam_dx, cam_dy = 0.0, 0.0
    for f in range(low + 1, high + 1):
        dxy = camera_displacements.get(f, (0.0, 0.0))
        cam_dx += float(dxy[0])
        cam_dy += float(dxy[1])
    if e_f < s_f:
        cam_dx *= -1.0
        cam_dy *= -1.0

    ped_dx = float(e_u[0] - s_u[0])
    ped_dy = float(e_u[1] - s_u[1])

    rel_dx = ped_dx - cam_dx
    rel_dy = ped_dy - cam_dy

    return {
        "start_frame": s_f,
        "end_frame": e_f,
        "ped_start_und": [float(s_u[0]), float(s_u[1])],
        "ped_end_und": [float(e_u[0]), float(e_u[1])],
        "camera_delta_und": [float(cam_dx), float(cam_dy)],
        "relative_displacement_und": [float(rel_dx), float(rel_dy)],
    }


def determine_intent(
        ped_track: Dict[str, Any],
        camera_displacements_undistorted: Dict[int, Tuple[float, float]],
        calibration_json_path: str,
        motion_threshold: float = 0.15) -> Tuple[List[str], Dict[str, Any]]:
    """
    只用首尾点做意图判定。
    """
    rs = get_pedestrian_real_displacement_endpoints(
        ped_track=ped_track,
        camera_displacements=camera_displacements_undistorted,
        calibration_json_path=calibration_json_path,
    )
    rel_dx, rel_dy = rs["relative_displacement_und"]

    lateral = "stationary"
    if abs(rel_dx) > motion_threshold:
        lateral = "goes to the right" if rel_dx > 0 else "goes to the left"

    vertical = "stationary"
    if abs(rel_dy) > motion_threshold:
        vertical = "moves away from ego vehicle" if rel_dy < 0 else "moves towards ego vehicle"

    return [lateral, vertical], rs


# --------------------------------- 可视化模块 -------------------------------------
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


# ------------------------------------ 主函数 -------------------------------------
# 这个函数应该是完整的数据处理流程
def process_dataset(video_root: str, anno_root: str, vehicle_root: str, camera_param_path: str, output_dir: str):
    """Process entire dataset and save results"""

    # 加载路径
    dataset_path = build_dataset(video_root, anno_root, vehicle_root)
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
        output_data = parse_pedestrians(anno_path)

        # Get video dimensions from first frame
        # animate_optical_flow(video_path)

        # Track objects in video and get their histories，这里应该是相当于获取行人的轨迹，行人轨迹已经合并了，并且生成的是新的id
        track_histories, camera = process_video(video_path)

        # 如果出错了，就使用这个函数加载
        # track_histories = load_json(r"filtered_tracks.json")

        # Plot 3D tracks for this sample，这是在 for 循环里的啊
        # plot_3d_tracks(track_histories, f"Object Tracks - Sample {sample_id}")

        # Match objects using Hungarian algorithm
        matches = match_objects(track_histories, output_data)

        # 获取相机帧间位移
        vehicle = parse_vehicle(sample_data['vehicle_path'])
        camera = build_camera_displacements(vehicle, camera_param_path)
        save_results_to_json(camera, r"camera_displacements.json")
        print(f"相机位移已计算完成")

        # Process direction intention
        motion = {}
        for track_id, track_data in track_histories.items():
            # 这里输出的是 [lateral intent, vertical intent]
            track_histories[track_id]['dir_intent'], check_point = determine_intent(track_data, camera,
                                                                                    camera_param_path,
                                                                                    motion_threshold=0.15)
            motion[track_id] = check_point

        # TODO: 匹配结束之后，GT可能有些没匹配上，这时候就需要对他们进行意图分析

        save_results_to_json(motion, r"motion_check.json")

        save_xml_annotation(anno_path, output_dir, track_histories, matches)
        count += 1


def main():
    """
    video_path = r"PIE/video"
    anno_path = r"PIE/annotation"
    output_dir = r"PIE/new_annotation"
    process_dataset(video_path, anno_path, output_dir)
    """
    # test_match(r"filtered_tracks.json", r"PIE/annotation/set02/video_0003_annt.xml")


# ------------------------------------- 测试 ----------------------------------------
# 用来测试 轨迹匹配
def test_match(json_path, gt_path):
    track_histories = load_json(json_path)
    inputs = parse_pedestrians(gt_path)
    matches, pairs, cost = match_objects(track_histories, inputs)
    print(pairs)
    print([f"{num:.2f}" for num in cost])


def test_camera_motion(json_path):
    all_tracks = load_json(json_path)
    obj = all_tracks['5420']
    camera = all_tracks.get('camera', {})
    new_camera = camera
    obj_frames = obj['frame_nums']
    obj_track = obj['centroids']
    camera_track = [new_camera[frame] for frame in obj_frames]
    print(obj_track[0])
    print(obj_track[1])
    print(obj_track[-1])
    print(camera_track[0])
    print(camera_track[1])
    print(camera_track[-1])


def test_intent_from_track(json_path, vehicle_path, calibration_json_path):
    # 先尝试只对一个行人的意图进行分析
    all_tracks = load_json(json_path)
    obj = all_tracks['5489']

    vehicle = parse_vehicle(vehicle_path)
    camera = build_camera_displacements_corrected(vehicle, calibration_json_path)
    # save_results_to_json(camera, r"camera_displacements.json")

    intent, checkpoint = determine_intent(obj, camera, calibration_json_path, motion_threshold=0.15)
    print(intent)
    print(checkpoint)


if __name__ == "__main__":
    # main()
    # test_camera_motion(r"filtered_tracks.json")
    test_intent_from_track(r"filtered_tracks.json",
                           r"PIE/annotation_vehicle/set02/video_0003_obd.xml",
                           r"PIE/camera_params/calibration_data.json")
