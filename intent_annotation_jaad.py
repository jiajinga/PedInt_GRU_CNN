"""
Person Intent Tracker

This script processes videos to annotate object intents in frames.
Key features:
- Handles videos from moving vehicle perspective
- Scales bounding boxes (video frames are half size of reference)
- Annotates vertical intent (along road) and lateral intent (across frame)  意图
- Considers motion relative to road, not the moving vehicle  运动是相对于道路的，而不是相对于移动的车辆
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import argparse
import json
import requests
import tempfile
import random
from collections import deque
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torchvision
from torchvision.transforms import functional as F
from matplotlib import animation
import os
import cv2
from multiprocessing import Pool, cpu_count
from itertools import islice
from tqdm import tqdm

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

###########################
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
        self.scale_factor = 0.5  # Video frames are half size of reference
        self.camera_motion = []   # Store camera centroids for each frame

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

    def determine_position(self, centroid: Tuple[int, int]) -> str:
        """Determine object position relative to ego vehicle"""
        if centroid[0] < self.position_threshold:
            return "Left of ego vehicle"
        if centroid[0] > self.frame_size[0] - self.position_threshold:
            return "Right of ego vehicle"
        return "Front of ego vehicle"

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
        windows = [(0, len(rel_history)//2), (len(rel_history)//2, len(rel_history))]
        dx_values = []
        dy_values = []
        for start, end in windows:
            if end - start < 2:
                continue
            window = rel_history[start:end]
            dx = [window[i+1][0] - window[i][0] for i in range(len(window)-1)]
            dy = [window[i+1][1] - window[i][1] for i in range(len(window)-1)]
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


def load_json_data(file_path: str) -> dict:
    """Load JSON data from file"""
    print(f"Loading JSON data from {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_data(data: dict, file_path: str):
    """Save JSON data to file"""
    print(f"Saving JSON data to {file_path}")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def get_centroid(box: List[int]) -> Tuple[int, int]:
    """Calculate centroid from bounding box coordinates"""
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

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
        good_new = p1[st==1]
        good_old = p0[st==1]

        # Calculate motion vectors
        motion_vectors = good_new - good_old
        # Get median motion to remove outliers
        if len(motion_vectors) > 0:
            median_motion = np.median(motion_vectors, axis=0)
            return median_motion[0], median_motion[1]

    return 0, 0

# 计算从开始到最后的位移
def calculate_displacement(points: List[Tuple[float, float]]) -> Tuple[float, Tuple[float, float], Tuple[float, float], float, float]:
    """Calculate net displacement between first and last point"""
    if len(points) < 2:
        return 0.0, (0.0, 0.0), (0.0, 0.0), 0.0, 0.0

    start = np.array(points[0], dtype=np.float32)
    end = np.array(points[-1], dtype=np.float32)

    displacement = float(np.linalg.norm(end - start))
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])

    return displacement, tuple(start), tuple(end), dx, dy

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


# 这个函数是整个流程的核心，处理视频帧，跟踪对象，分析意图，并返回跟踪历史
def process_video(video_url: str, intent_analyzer: IntentAnalyzer,
                 conf_threshold: float = 0.25) -> Dict[int, Dict]:
    """
    Process video to track objects and analyze their intent
    Returns dictionary mapping object IDs to their tracking histories
    """
    # Download and open video
    video_path = download_video(video_url)
    cap = cv2.VideoCapture(video_path)

    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = width * height

    track_histories = {}  # Store tracking histories
    camera_motion = []   # Store camera motion vectors
    prev_frame = None
    frame_count = 0
    cyclist_memory = {}  # Store cyclist detection history

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Estimate camera motion if we have a previous frame
        if prev_frame is not None:
            dx, dy = estimate_camera_motion(prev_frame, frame)
            camera_motion.append((dx, dy))
        prev_frame = frame.copy()

        # Run YOLOv8 tracking for persons
        results = yolo_model.track(frame, persist=True, conf=conf_threshold, classes=[0], verbose=False, tracker="botsort.yaml")  # person class only

        # Get cycle detections
        cycle_boxes = detect_cycles(frame)

        person_detections = []
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Convert boxes to xyxy format for person detections
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                xyxy = [x-w/2, y-h/2, x+w/2, y+h/2]
                person_detections.append((xyxy, track_id))

                # Filter out unrealistic detections
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
                track_histories[track_id]['centroids'].append((float(x), float(y)))
                track_histories[track_id]['frame_nums'].append(frame_count)

                # Draw centroid on frame
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.putText(frame, f'ID: {track_id}', (int(x), int(y) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update cyclist tracks
        track_histories = update_cyclist_tracks(track_histories, frame_count,
                                              person_detections, cycle_boxes,
                                              cyclist_memory)

        # Draw cycle detections
        for box in cycle_boxes:
            x1, y1, x2, y2, conf = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

        # Display frame
        # display_frame_with_grid(frame)
        frame_count += 1

    cap.release()
    os.remove(video_path)

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
    displacements = {}
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
        }

    

    yolo_model.predictor.trackers[0].reset()
    return track_histories


def convert_bbox_format(bbox):
    """Convert bounding box format if needed"""
    x1, y1 = bbox[0]
    x2, y2 = bbox[2]
    return [x1, y1, x2, y2]


# 这个函数使用光流法计算视频帧之间的运动，并创建一个动画来可视化这些运动。它可以帮助我们理解视频中对象的动态行为。
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
def get_median_optical_flow_multiple_margins(video_path, point, box_size=(50, 150), margins=[50, 70,  100, 120, 150, 170, 200, 50], max_frames=100, y_shift = False, margin_add = 0):
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
            flow_box = flow[box_y:box_y+bh, box_x:box_x+bw]
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
def get_median_optical_flow(video_path, point, box_h = 150,  max_frames=100, y_shift = False):
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
        flow_box = flow[box_y:box_y+bh, box_x:box_x+bw]
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


# 这个函数应该是完整的数据处理流程，参数列表中的 str 只是类型提示，并非强制
def process_dataset(dataset_filepath: str, original_dataset_filepath:str, output_dir: str, output_json: str, diff_keys = None):
    """Process entire dataset and save results"""

    # Load dataset, 换成自己的
    dataset = load_json_data(dataset_filepath)
    original_dataset = load_json_data(original_dataset_filepath)
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process each sample
    output_data = {}
    flag = 1
    count = 0
    for sample_id, sample_data in tqdm(islice(dataset.items(), 50), desc="Processing samples", total=50):  # tqdm是个啥，看不懂，先放这
        
        # Copy original data
        output_data[sample_id] = sample_data.copy()

        # Find corresponding annotation
        annotation_index = next(
            (i for i, ann in enumerate(original_dataset)
             if ann.get('s3_fileUrl') == sample_data['image_path']),
            None
        )

        if annotation_index is None:
            print(f"No annotation found for frame {sample_id}")
            continue

        # Process pedestrian annotations，这里好像是提取行人的标签，如果是行人且运动方向不为N/A或空，就把它的边界框和意图添加到输出数据中
        # 但是一开始的数据集不是本来就没有这些东西吗？
        if (original_dataset[annotation_index].get('Agent-classifier') == 'Pedestrian' and
            original_dataset[annotation_index].get('pedestrian_motion_direction') not in ["N/A", []]):
            output_data[sample_id]['Pedestrians'][str(len(sample_data['Pedestrians']) + 1)] = {
                "Box": convert_bbox_format(original_dataset[annotation_index].get('geometry')),
                "Intent": original_dataset[annotation_index].get('pedestrian_motion_direction')[0]
            }

        # Process cyclist annotations，类似地处理自行车手
        if (original_dataset[annotation_index].get('Agent-classifier') == 'Cyclist' and
            original_dataset[annotation_index].get('pedestrian_motion_direction') not in ["N/A", []]):
            output_data[sample_id]['Cyclists'][str(len(sample_data['Cyclists']) + 1)] = {
                "Box": convert_bbox_format(original_dataset[annotation_index].get('geometry')),
                "Intent": original_dataset[annotation_index].get('pedestrian_motion_direction')[0]
            }

        try:
            # Get video dimensions from first frame
            video_path = "JAAD/JAAD_clips"
            # animate_optical_flow(video_path)
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            # Initialize intent analyzer with more sensitive threshold
            intent_analyzer = IntentAnalyzer(frame_size=(width, height), motion_threshold=0.15)

            # Track objects in video and get their histories
            track_histories = process_video(output_data[sample_id]['video_path'], intent_analyzer)

            # Set camera motion in intent analyzer
            if 'camera' in track_histories:
                intent_analyzer.set_camera_motion(track_histories['camera']['centroids'])

            # Plot 3D tracks for this sample
            plot_3d_tracks(track_histories, f"Object Tracks - Sample {sample_id}")
            matched_tracks = {}
            # Process each object type separately
            for object_type in ['Pedestrians', 'Cyclists']:
                if object_type not in output_data[sample_id]:
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
                filtered_input_boxes = remove_duplicate_boxes(output_data[sample_id][object_type])

                # Match objects using Hungarian algorithm
                matches = match_objects(type_tracks, filtered_input_boxes)
                # Process matches
                for track_idx, obj_id in matches.items():
                    track_id = track_idx
                    track_data = type_tracks[track_id]
                    matched_tracks[track_id] = track_data

                    # Update track history for intent analysis
                    for i, centroid in enumerate(track_data['centroids']):
                        if i == 0:
                            cam_dx, cam_dy = get_median_optical_flow(video_path, (centroid[0], centroid[1]))

                            iter = 0
                            while cam_dy < 0:
                              cam_dx, cam_dy = get_median_optical_flow_multiple_margins(video_path, (centroid[0], centroid[1]), y_shift=True, margin_add=iter*5)
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
                    output_data[sample_id][object_type][obj_id].update({
                        'Intent': intent,
                        'Position': position,
                        'Description': intent_analyzer.generate_description(intent)
                    })
            count += 1
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}")
            continue

        if count % 100 == 0:
          print(f"Processed {count} samples")
          save_json_data(output_data, output_json)

    # Save results
    save_json_data(output_data, output_json)


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
        local_output = { sample_id: sample_data.copy() }

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
                                    margin_add=iter*5
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
    parser = argparse.ArgumentParser(description='Analyze pedestrians intent in videos-clips')
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
      process_dataset_parallel(args.dataset_filepath, args.original_dataset_filepath, args.output_dir, args.output_json)
    else:
      process_dataset(args.dataset_filepath, args.original_dataset_filepath, args.output_dir, args.output_json )

if __name__ == "__main__":
    main()