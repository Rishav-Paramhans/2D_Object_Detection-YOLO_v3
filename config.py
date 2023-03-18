import os
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils import seed_everything

file_dir= os.path.dirname(__file__)
DATASET = r"E:\Thesis_Final\A2D2_dataset"
DEVICE = "cuda" if torch.cuda.is_available() else print("using cpu")  # using GPU3 on Thor
print('device', DEVICE)
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 0
BATCH_SIZE = 1
WRITER= True
IMAGE_SIZE = 608
NUM_CLASSES = 10
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.6
NMS_IOU_THRESH = 0.35
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "Saved_model/checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/YOLO_Bbox_2D_label/"

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]


scale = 1
train_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),

        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)
test_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),

        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)
A2D2_CLASSES = [
    "Car",
    "Pedestrian",
    "Truck",
    "VanSUV",
    "Cyclist",
    "Bus",
    "MotorBiker",
    "Bicycle",
    "UtilityVehicle",
    "Motorcycle"
]