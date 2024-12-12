import os
from typing import List, Tuple
from datetime import datetime as dt
from math import ceil

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as mplt

import anomaly_detection as ad
from face_landmarks_subset import gesture_landmark_indexes
from config import LANDMARKER_MODEL_PATH, GESTURE_MODEL_PATH, PASSWORD_PATH, VIDEO_REC_PATH, VID_HEIGHT, VID_WIDTH, VID_FPS


gesture2class_dict = {
    'neutral': 0,
    'blink': 1,
    'brows_lifted': 2,
    'mouth_open': 3,
    'smile': 4,
    'wink_left': 5,
    'wink_right': 6
}

class2gesture_dict = {c: g for g, c in gesture2class_dict.items()}

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])
    
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image


# CAMERA DEMO UTILS

def prepare_video():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, VID_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, VID_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, VID_FPS)

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    output_file = os.path.join(VIDEO_REC_PATH, f'{str(dt.now()).replace(':', '-')}.mp4')
    video_out = cv2.VideoWriter(output_file, codec, VID_FPS, (VID_WIDTH, VID_HEIGHT), True)

    return camera, video_out


def load_models():
    model_file = open(LANDMARKER_MODEL_PATH, 'rb')
    model_data = model_file.read()
    model_file.close()
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    landmark_detector = vision.FaceLandmarker.create_from_options(options)

    gesture_recognition = tf.saved_model.load(GESTURE_MODEL_PATH)

    return landmark_detector, gesture_recognition


def get_gesture_landmarks(detection_result):
    landmarks = detection_result.face_landmarks[0]
    valuable_landmarks = [landmarks[i] for i in gesture_landmark_indexes]
    landmark_coords = [[landmark.x, landmark.y, landmark.z] for landmark in valuable_landmarks]
    return landmark_coords


def resolve_gesture_delta(cur_probs, new_probs):
    DIV_NORM = 10**6
    ACT_THRESHOLDS = [3, 5, 3, 3, 3, 3, 3] 
    prob_deltas = [(n*DIV_NORM)/(c*DIV_NORM) for c, n in zip(cur_probs, new_probs)]
    max_delta = max(prob_deltas)
    gesture = prob_deltas.index(max(prob_deltas))
    if max_delta < ACT_THRESHOLDS[gesture]: return None
    return gesture


def get_ad_landmarks(detection_result) -> ad.LandmarksFrame:
    landmarks = detection_result.face_landmarks[0]
    ad_landmarks = {}
    for l in ad.AnomalyLandmark:
        l_index = ad.landmark2i_dict[l]
        l_sig_axis = ad.landmark2axis_dict[l]
        ad_landmarks[l] = landmarks[l_index].__dict__[l_sig_axis]
    return ad_landmarks


def get_ad_paths(ad_cache: List[Tuple[float, ad.LandmarksFrame]]) -> ad.LandmarkPaths:
    t0 = ad_cache[0][0]
    landmark_paths = {}
    for l in ad.AnomalyLandmark:
        path = []
        for frame in ad_cache:
            ts = frame[0] - t0          
            val = frame[1][l]
            path.append([ts, val])
        landmark_paths[l] = path
    return landmark_paths


def save_password(sequence):
    password = ''.join(map(str, sequence))
    with open(PASSWORD_PATH, 'w') as file:
        file.write(password)
        file.close()


def check_password(sequence):
    password = ''.join(map(str, sequence))
    correct_password = open(PASSWORD_PATH, 'r').read()
    return password == correct_password


def visualize_ad_paths(gesture, paths: ad.LandmarkPaths):
    rows, cols = 3, 4
    figure, axis = mplt.subplots(rows, cols)
    i = 0
    for l, p in paths.items():
        row = ceil(i/cols) - 1
        col = i - row*cols - 1
        axis[row, col].scatter([s[0] for s in p], [s[1] for s in p])
        axis[row, col].set_title(f'{gesture} - {l.value}')
        i += 1
    mplt.show()

