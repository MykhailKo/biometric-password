import os
import csv
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import draw_landmarks_on_image


MODEL_PATH = '/models/face_landmarker.task'
DATASET_PATH = 'dataset/'
RESULT_IMG_PATH = 'results/landmark_img/'
RESULT_DATA_PATH = 'results/landmark_data/'


base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

class_dirs = os.listdir('dataset/')
skip_classes = ['neutral', 'smile', 'blink', 'brows_lifted', 'mouth_open']

for class_dir in class_dirs:
  if class_dir in skip_classes: continue
  os.makedirs(os.path.join(RESULT_IMG_PATH, class_dir), exist_ok=True)
  os.makedirs(os.path.join(RESULT_DATA_PATH, class_dir), exist_ok=True)

  images = os.listdir(os.path.join(DATASET_PATH, class_dir))
  for file in images:
    file_name = file.split('.')[0]

    image = mp.Image.create_from_file(os.path.join(DATASET_PATH, class_dir, file))
    detection_result = detector.detect(image)
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    landmark_coords = [[landmark.x, landmark.y, landmark.z] for landmark in detection_result.face_landmarks[0]] 
    with open(os.path.join(RESULT_DATA_PATH, class_dir, f'{file_name}.csv'), 'w', newline='') as f:
        csv.writer(f).writerows(landmark_coords)

    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(RESULT_IMG_PATH, class_dir, f'{file_name}.jpg'), annotated_image)
    cv2.imshow('Manual landmark validation', annotated_image)
    cv2.waitKey(0)