from openpose import pyopenpose as op
import argparse
import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
import pickle
import plotly.graph_objects as go
import shutil
import xml.etree.cElementTree as ET
import glob

mp_hands = mp.solutions.hands

# Flag
parser = argparse.ArgumentParser()
parser.add_argument("--root", default="./hand_views/", type=str)
args = parser.parse_known_args()

# Custom Params (refer to include/openpose/flags.hpp for more parameters)
params = dict()

# Add others in path?
for i in range(0, len(args[1])):
    curr_item = args[1][i]
    if i != len(args[1]) - 1:
        next_item = args[1][i + 1]
    else:
        next_item = "1"
    if "--" in curr_item and "--" in next_item:
        key = curr_item.replace("-", "")
        if key not in params:
            params[key] = "1"
    elif "--" in curr_item and "--" not in next_item:
        key = curr_item.replace("-", "")
        if key not in params:
            params[key] = next_item

params["model_folder"] = "./models/"
params["hand"] = True
params["hand_detector"] = 2
params["body"] = 0
params["3d"] = True
params["number_people_max"] = 1
params["camera_parameter_path"] = os.path.join(args[0].root, "param/")
params["image_dir"] = os.path.join(args[0].root, "output/")

# Starting OpenPose
opWrapper = op.WrapperPython(op.ThreadManagerMode.Asynchronous)
opWrapper.configure(params)
opWrapper.start()

# By how much to pad hand detection bounding box
padding = 0.3

# Write camera parameters according to OpenPose format
with open("./hand_views/cam_data.pkl", "rb") as f:
    cam_data = pickle.load(f)

image_files = ["front", "back", "top", "bottom", "right"]

bboxes = []
data = []
handRectangles = []
valid_files = []

with open(os.path.join(args[0].root, "cam_data.pkl"), "rb") as f:
    cam_data = pickle.load(f)

# Use mediapipe to get bounding boxes
with mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
) as hands:
    base_path = os.path.join(args[0].root, "render")
    for file in glob.glob("**/*.png", root_dir=base_path):
        file_path = os.path.join(base_path, file)
        bbox = []
        image = cv2.flip(cv2.imread(file_path), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_handedness:
            print("-- WARNING: Skipping:", file)
            continue

        handedness = results.multi_handedness[0].classification[0].label
        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        landmarks = results.multi_hand_landmarks[0].landmark
        for landmark in landmarks:
            bbox.append([landmark.x * image_width, landmark.y * image_height])

        # Calculate bounding box
        bbox = np.array(bbox)
        bbox_min = bbox.min(0)
        bbox_max = bbox.max(0)
        bbox_size = bbox_max - bbox_min

        # Pad hand bounding box
        bbox_min -= bbox_size * padding
        bbox_max += bbox_size * padding
        bbox_size = bbox_max - bbox_min

        # Convert bbox to square of length equal
        # to longer edge
        diff = bbox_size[0] - bbox_size[1]
        if diff > 0:
            bbox_min[1] -= diff / 2
            bbox_max[1] += diff / 2
            bbox_size[1] = bbox_size[0]
        else:
            bbox_min[0] -= -diff / 2
            bbox_max[0] += -diff / 2
            bbox_size[0] = bbox_size[1]

        # Flip
        tmp = bbox_min[0]
        bbox_min[0] = image_width - bbox_max[0]
        bbox_max[0] = image_width - tmp
        image = cv2.flip(image, 1)

        bboxes.append([*bbox_min, *bbox_size])

        handRectangles += [
            op.Rectangle(0, 0, 0, 0),
            op.Rectangle(*bboxes[-1]),
        ]
        valid_files += [file]

shutil.rmtree(params["image_dir"])
os.makedirs(params["image_dir"])
shutil.rmtree(params["camera_parameter_path"])
os.makedirs(params["camera_parameter_path"])

i = 1
print(f"-- INFO: Found bounding boxes for {len(valid_files)} images")
for file in valid_files:
    cam = cam_data[os.path.basename(file).split(".")[0]]

    image_path = os.path.join(args[0].root, "render", file)
    shutil.copyfile(image_path, os.path.join(params["image_dir"], f"{i}.png"))

    # Write camera parameter XML
    root = ET.Element("opencv_storage")
    ext_matrix = ET.SubElement(root, "CameraMatrix", type_id="opencv-matrix")
    ET.SubElement(ext_matrix, "rows").text = str(3)
    ET.SubElement(ext_matrix, "cols").text = str(4)
    ET.SubElement(ext_matrix, "dt").text = "d"
    ET.SubElement(ext_matrix, "data").text = " ".join(
        map(str, cam["extrinsics_opencv"].flatten().tolist())
    )

    int_matrix = ET.SubElement(root, "Intrinsics", type_id="opencv-matrix")
    k = np.eye(3)
    k[0, 0] = cam["K"][0]
    k[1, 1] = cam["K"][1]
    k[0, 2] = cam["K"][2]
    k[1, 2] = cam["K"][3]
    ET.SubElement(int_matrix, "rows").text = str(3)
    ET.SubElement(int_matrix, "cols").text = str(3)
    ET.SubElement(int_matrix, "dt").text = "d"
    ET.SubElement(int_matrix, "data").text = " ".join(map(str, k.flatten().tolist()))

    dist_matrix = ET.SubElement(root, "Distortion", type_id="opencv-matrix")
    ET.SubElement(dist_matrix, "rows").text = str(8)
    ET.SubElement(dist_matrix, "cols").text = str(1)
    ET.SubElement(dist_matrix, "dt").text = "d"
    ET.SubElement(dist_matrix, "data").text = " ".join(
        map(str, np.zeros(8).flatten().tolist())
    )

    tree = ET.ElementTree(root)
    text = ET.tostring(root, xml_declaration=True).decode("utf-8")
    text = str.replace(text, "'", '"')
    with open(os.path.join(params["camera_parameter_path"], f"{i}.xml"), "w") as f:
        f.write(text)

    i += 1

# Run openpose 3D hand detection module
datums = op.VectorDatum()
result = opWrapper.detectHandKeypoints3D(datums, handRectangles)
if result:
    coords = datums[0].handKeypoints3D[1][0]
else:
    print("-- ERROR: Pose estimation failed")
    exit(1)

fig = go.Figure(
    go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode="markers"
        # mode='lines',
        # line_width=2,
        # line_color='blue'
    )
)
fig.write_html("keypoints.html", auto_open=True)
print("-- INFO: Saved 3D visualization in 'keypoints.html'")
