import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from scipy.optimize import linear_sum_assignment
# 为什么没有引入 pyopenpose ？下面进行了封装


def _import_openpose():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(dir_path)
    os.environ["PATH"] = os.environ["PATH"] + ";" + dir_path + ";"
    import pyopenpose as op  # type: ignore

    return op, project_root


def _build_openpose_params(project_root: str, unknown_args: List[str]) -> Dict[str, str]:
    params: Dict[str, str] = {
        "model_folder": os.path.join(project_root, "models"),
        "model_pose": "COCO",
        "net_resolution": "368x256",   # 若要保证16:9，则应该设置为 256*144
    }

    for i in range(0, len(unknown_args)):
        curr_item = unknown_args[i]
        next_item = unknown_args[i + 1] if i != len(unknown_args) - 1 else "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace("-", "")
            if key not in params:
                params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace("-", "")
            if key not in params:
                params[key] = next_item
    return params


def _list_images(image_dir: Path) -> List[Tuple[int, Path]]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in image_dir.iterdir() if p.suffix.lower() in exts]

    def _parse_frame_idx(path: Path) -> int:
        stem = path.stem
        if stem.isdigit():
            return int(stem)
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            return int(digits)
        return -1

    parsed = [(_parse_frame_idx(p), p) for p in image_paths]
    has_idx = [item for item in parsed if item[0] >= 0]
    if len(has_idx) == len(parsed):
        return sorted(parsed, key=lambda x: x[0])

    # Fallback to filename order when frame index cannot be parsed.
    sorted_paths = sorted(image_paths, key=lambda p: p.name)
    return list(enumerate(sorted_paths))


def _infer_set_id(image_set_dir: Path) -> str:
    if image_set_dir.name:
        return image_set_dir.name
    raise ValueError("无法从 image_set_dir 推断 set_id，请显式传入 --set_id")


def _discover_video_dirs(image_set_dir: Path) -> List[Path]:
    return sorted([p for p in image_set_dir.iterdir() if p.is_dir()])


def _discover_annotation_map(anno_set_dir: Path) -> Dict[str, Path]:
    """
    生成 video_id -> annotation_xml_path 的映射。
    约定标注文件名格式: <video_id>_annt.xml，例如 video_0003_annt.xml。
    """
    anno_map: Dict[str, Path] = {}
    for p in anno_set_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != ".xml":
            continue
        stem = p.stem
        video_id = stem[:-5] if stem.endswith("_annt") else stem
        anno_map[video_id] = p
    return anno_map


def _frame_key(frame_path: Optional[Path], fallback_frame_idx: int, ped_id: str) -> str:
    if frame_path is not None:
        return f"{frame_path.stem}_{ped_id}"
    return f"{fallback_frame_idx:05d}_{ped_id}"


def parse_pedestrians(anno_path: Path) -> Dict[str, Dict[str, Any]]:
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

        for i, box in enumerate(track.findall('box')):
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


def _load_tracks(track_path: Path) -> Dict[int, Dict[str, List[float]]]:
    data = parse_pedestrians(track_path)
    # 按帧组织: frame_idx -> {ped_id: [x1, y1, x2, y2]}
    frame_tracks: Dict[int, Dict[str, List[float]]] = {}
    for ped_id, track in data.items():
        boxes = track.get("boxes", [])
        frame_nums = track.get("frame_nums", [])
        if not boxes or not frame_nums or len(boxes) != len(frame_nums):
            continue

        for fr, box in zip(frame_nums, boxes):
            if box and len(box) >= 4:
                frame_idx = int(fr)
                if frame_idx not in frame_tracks:
                    frame_tracks[frame_idx] = {}
                frame_tracks[frame_idx][str(ped_id)] = [
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                ]

    return frame_tracks


def _box_center_xyxy(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _pose_hip_center(person_pose: np.ndarray, conf_thresh: float) -> Optional[Tuple[float, float]]:
    # COCO_18 hip indexes: RHip=8, LHip=11
    candidates = []
    for hip_idx in (8, 11):
        x, y, c = person_pose[hip_idx]
        if c >= conf_thresh:
            candidates.append((float(x), float(y)))

    if candidates:
        xs = [p[0] for p in candidates]
        ys = [p[1] for p in candidates]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    valid = person_pose[person_pose[:, 2] >= conf_thresh]
    if len(valid) == 0:
        return None
    return float(np.mean(valid[:, 0])), float(np.mean(valid[:, 1]))


def _flatten_pose_xy_36(person_pose: np.ndarray, conf_thresh: float) -> List[float]:
    # Keep COCO 18 keypoints order, output only x,y -> 36 dims.
    flat: List[float] = []
    for kp in person_pose[:18]:
        x, y, c = kp
        if float(c) >= conf_thresh:
            flat.extend([float(x), float(y)])
        else:
            flat.extend([0.0, 0.0])
    return flat


def _hungarian_match(
    ped_boxes: Dict[str, List[float]],
    pose_centers: List[Tuple[float, float]],
    cost_threshold: float,
    # frame_idx
) -> Dict[str, int]:
    """
    使用匈牙利算法分配行人框和姿态中心。
    cost = |dx|/(2w) + |dy|/(2h)
    仅保留 cost < cost_threshold 的匹配。
    """
    ped_ids = list(ped_boxes.keys())
    num_peds = len(ped_ids)
    num_poses = len(pose_centers)
    if num_peds == 0 or num_poses == 0:
        return {}

    # ped_center = []   # 后续注释掉
    cost_matrix = np.zeros((num_peds, num_poses), dtype=np.float32)
    for i, ped_id in enumerate(ped_ids):
        x1, y1, x2, y2 = ped_boxes[ped_id]
        bx, by = _box_center_xyxy(ped_boxes[ped_id])
        # ped_center.append((bx, by))   # 后续注释掉
        w = max(float(x2 - x1), 1e-6)
        h = max(float(y2 - y1), 1e-6)
        for j, (px, py) in enumerate(pose_centers):
            d_x = abs(px - bx)
            d_y = abs(py - by)
            cost_matrix[i, j] = float(d_x / (2.0 * w) + d_y / (2.0 * h))

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    assignments: Dict[str, int] = {}
    assigned_costs: List[float] = []
    for r, c in zip(row_idx.tolist(), col_idx.tolist()):
        cst = float(cost_matrix[r, c])
        assigned_costs.append(cst)
        if cst < cost_threshold:
            assignments[ped_ids[r]] = int(c)
    
    # print(frame_idx)
    # print(ped_center)  # 后续注释掉
    # print(pose_centers)
    # print(cost_matrix)

    return assignments


def _iter_image_frames(image_dir: Path) -> Iterable[Tuple[int, Path, np.ndarray]]:
    for frame_idx, image_path in _list_images(image_dir):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        yield frame_idx, image_path, frame


def main():
    parser = argparse.ArgumentParser(
        description="按 set 批量提取 OpenPose COCO18 姿态，并保存为 pose_{set_id}.pkl。"
    )
    parser.add_argument("--image_set_dir", type=str, default=r"PIE\images\set01",
                        help="输入图像 set 目录，目录下应为多个视频子目录")
    parser.add_argument("--anno_set_dir", type=str, default=r"PIE\annotations\new_annotations\set01",
                        help="输入标注 set 目录，目录下应为多个视频 XML")
    parser.add_argument("--output_dir", type=str, default=r"data\features\pie\new_poses",
                        help="输出目录（文件名固定为 pose_{set_id}.pkl）")
    parser.add_argument("--set_id", type=str, default=None, help="可选 set_id，不传则从 image_set_dir 推断")
    parser.add_argument("--kp_conf_thresh", type=float, default=0.05, help="关键点置信度阈值")
    parser.add_argument("--cost_threshold", type=float, default=0.3, help="匈牙利匹配后保留阈值，cost < threshold")
    parser.add_argument("--fill_missing_zero", action="store_true", help="未匹配到姿态的行人也写入全 0 pose")
    args, unknown_args = parser.parse_known_args()

    image_set_dir = Path(args.image_set_dir)
    anno_set_dir = Path(args.anno_set_dir)
    if not image_set_dir.is_dir():
        raise FileNotFoundError(f"图像 set 目录不存在: {image_set_dir}")
    if not anno_set_dir.is_dir():
        raise FileNotFoundError(f"标注 set 目录不存在: {anno_set_dir}")

    op, project_root = _import_openpose()
    params = _build_openpose_params(project_root, unknown_args)

    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()

    set_id = args.set_id if args.set_id else _infer_set_id(image_set_dir)
    video_dirs = _discover_video_dirs(image_set_dir)
    anno_map = _discover_annotation_map(anno_set_dir)

    if not video_dirs:
        raise ValueError(f"图像 set 目录下未发现视频子目录: {image_set_dir}")

    output: Dict[str, Dict[str, List[float]]] = {}
    total_frames = 0
    matched_count = 0

    for video_dir in video_dirs:
        video_id = video_dir.name
        anno_path = anno_map.get(video_id)
        if anno_path is None:
            print(f"[WARNING] 缺少标注文件，跳过视频: {video_id}")
            continue

        output[video_id] = {}
        frame_tracks = _load_tracks(anno_path)
        frame_iter = _iter_image_frames(video_dir)

        for frame_idx, frame_path, frame in frame_iter:
            total_frames += 1
            if total_frames > 4:
                break

            # 当前帧直接索引，无需遍历全部行人轨迹
            ped_boxes: Dict[str, List[float]] = frame_tracks.get(frame_idx, {})

            if not ped_boxes:
                continue

            datum = op.Datum()
            datum.cvInputData = frame
            op_wrapper.emplaceAndPop(op.VectorDatum([datum]))

            pose_keypoints = datum.poseKeypoints
            if pose_keypoints is None or len(pose_keypoints.shape) != 3:
                if args.fill_missing_zero:
                    for ped_id in ped_boxes:
                        key = _frame_key(frame_path, frame_idx, ped_id)
                        output[video_id][key] = [0.0] * 36
                continue

            # shape: [num_people, 18, 3] for COCO model
            pose_centers: List[Tuple[float, float]] = []
            pose_vecs: List[List[float]] = []
            for person_pose in pose_keypoints:
                center = _pose_hip_center(person_pose, args.kp_conf_thresh)
                if center is None:
                    continue
                pose_centers.append(center)
                pose_vecs.append(_flatten_pose_xy_36(person_pose, args.kp_conf_thresh))

            if not pose_centers:
                if args.fill_missing_zero:
                    for ped_id in ped_boxes:
                        key = _frame_key(frame_path, frame_idx, ped_id)
                        output[video_id][key] = [0.0] * 36
                continue

            matches = _hungarian_match(
                ped_boxes=ped_boxes,
                pose_centers=pose_centers,
                cost_threshold=args.cost_threshold, 
                # frame_idx=frame_idx   # 后续注释掉
            )

            for ped_id, pose_idx in matches.items():
                key = _frame_key(frame_path, frame_idx, ped_id)
                output[video_id][key] = pose_vecs[pose_idx]
                matched_count += 1

            if args.fill_missing_zero:
                for ped_id in ped_boxes:
                    key = _frame_key(frame_path, frame_idx, ped_id)
                    if key not in output[video_id]:
                        output[video_id][key] = [0.0] * 36

        print(f"[INFO] 视频处理完成: {video_id}, 写入样本数={len(output[video_id])}")

    output_arg_path = Path(args.output_dir)
    output_dir = output_arg_path.parent if output_arg_path.suffix.lower() == ".pkl" else output_arg_path
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"pose_{set_id}.pkl"

    with output_path.open("wb") as f:
        pickle.dump(output, f)

    print("OpenPose model_pose:", params.get("model_pose", "(default)"))
    print("set_id:", set_id)
    print("视频数量:", len(output))
    print("总处理帧数:", total_frames)
    print("总写入键值数量:", sum(len(v) for v in output.values()))
    print("成功匹配数量:", matched_count)
    print("输出文件:", str(output_path))


if __name__ == "__main__":
    main()

# python OpenPose_demo\bin\extract_pose_to_pkl.py