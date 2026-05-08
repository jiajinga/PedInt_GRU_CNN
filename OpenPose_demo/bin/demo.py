import os
import cv2
import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.dirname(dir_path)

os.environ["PATH"] = os.environ["PATH"] + ";" + dir_path + ";"
import pyopenpose as op

parser = argparse.ArgumentParser()
parser.add_argument(
    "--image_path",
    default=r"D:\PedInt_GRU_CNN\PIE\images\set02\video_0003\00002.png",  # 改成你实际图片
    help="Process an image."
)
args = parser.parse_known_args()

params = dict()
params["model_folder"] = os.path.join(project_root, "models")  # 绝对路径，避免 cwd 影响
params["model_pose"] = "COCO"  # 切换为 COCO_18（18个关键点）
params["net_resolution"] = "368x256"

for i in range(0, len(args[1])):
    curr_item = args[1][i]
    next_item = args[1][i + 1] if i != len(args[1]) - 1 else "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace("-", "")
        if key not in params:
            params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace("-", "")
        if key not in params:
            params[key] = next_item

imageToProcess = cv2.imread(args[0].image_path)
if imageToProcess is None:
    raise FileNotFoundError(f"无法读取图片: {args[0].image_path}")

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()
datum.cvInputData = imageToProcess
opWrapper.emplaceAndPop(op.VectorDatum([datum]))

if datum.cvOutputData is None:
    raise RuntimeError(f"OpenPose 处理失败，请检查模型目录: {params['model_folder']}")

print("Body keypoints:\n", datum.poseKeypoints)
cv2.imshow("OpenPose", datum.cvOutputData)
cv2.waitKey(0)
cv2.destroyAllWindows()