import time
from typing import List, Dict, Tuple

import cv2
import mediapipe as mp

import utils as u
from config import AD_CACHE_SIZE, INPUT_TIMEOUT, PASSWORD_LEN, NUM_CONFIRMS
import anomaly_detection as ad
import ui
    

camera, video_out = u.prepare_video()
landmark_detector, gesture_recognition = u.load_models()

repetitions: List[List[int]] = []
ad_gesture_train_data: Dict[str, ad.LandmarkPaths] = {}

def confirm_rep(sequence, ad_sequence: List[ad.LandmarkPaths]):
    repetitions.append(sequence)
    for i in range(len(sequence)):
        g = u.class2gesture_dict[sequence[i]]
        paths = ad_sequence[i]
        if not g in ad_gesture_train_data: ad_gesture_train_data[g] = {}
        for l, path in paths.items():
            train_data = ad_gesture_train_data[g][l] if l in ad_gesture_train_data[g] else []
            train_data.extend(path)
            ad_gesture_train_data[g][l] = train_data


ad_cache: List[Tuple[float, ad.LandmarksFrame]] = []

started = False
status = 'Biometric password Setup. Press S to start'
cur_rep = 1
confirmation_status = lambda: 'Create password' if cur_rep == 1 else f'Password confirmation {cur_rep-1} of {NUM_CONFIRMS}'
sequence = []
ad_sequence: List[ad.LandmarkPaths] = []
input_filled = False
input_unlocked = True
current_gesture_probs = []
input_ts = 0

while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    if not started:
        ui_image = ui.status(frame, status)
        video_out.write(ui_image)
        cv2.imshow('Camera', ui_image)
        key = cv2.waitKey(1)
        if key == ord('s'): started = started or True
        elif key == ord('q'): break
        continue
    
    input_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    detection_result = landmark_detector.detect(input_image)

    gesture_input = u.get_gesture_landmarks(detection_result)
    recognition_result = gesture_recognition.serve([gesture_input])
    probs = recognition_result.numpy().tolist()[0]

    frame_ts = time.time()
    ad_landmarks = u.get_ad_landmarks(detection_result)
    ad_cache.append([frame_ts, ad_landmarks])
    ad_cache = ad_cache[-AD_CACHE_SIZE:]

    if frame_ts - input_ts > INPUT_TIMEOUT: input_unlocked = True
    if input_unlocked and not input_filled:
        if not len(current_gesture_probs): 
            current_gesture_probs = probs
            continue
        gesture = u.resolve_gesture_delta(current_gesture_probs, probs)
        if gesture == u.gesture2class_dict['neutral']: continue    
        if gesture:
            input_unlocked = False
            input_ts = time.time()
            current_gesture_probs = []
            sequence.append(gesture)
            if len(sequence) == PASSWORD_LEN: input_filled = True
            print(u.class2gesture_dict[gesture], probs)
            ad_landmark_paths = u.get_ad_paths(ad_cache)
            ad_sequence.append(ad_landmark_paths)
            status = confirmation_status()

    annotated_image = u.draw_landmarks_on_image(input_image.numpy_view(), detection_result) 
    ui_image = ui.status(ui.sequence(annotated_image, sequence), status)

    video_out.write(ui_image)
    cv2.imshow('Camera', ui_image)

    key = cv2.waitKey(1)
    if key == ord('s'): started = started or True
    elif key == ord('q'): break
    elif key == ord('d'):
        if len(sequence): sequence.pop()
        if len(ad_sequence): ad_sequence.pop()
        input_filled = False
        status = confirmation_status()
    elif key == ord('e'):
        if len(sequence) != PASSWORD_LEN: continue
        if cur_rep != 1 and repetitions[0] != sequence: 
            status = 'Password does not match'
            continue
        confirm_rep(sequence, ad_sequence)
        sequence = []
        ad_sequence = []
        if cur_rep == NUM_CONFIRMS + 1:
            for g, p in ad_gesture_train_data.items(): ad.train_gesture(g, p)
            print('Anomaly detection models training completed')
            u.save_password(repetitions[0])
            break
        cur_rep += 1
        input_filled = False
        status = confirmation_status()


camera.release()
video_out.release()
cv2.destroyAllWindows()

for g, lp in ad_gesture_train_data.items(): u.visualize_ad_paths(g, lp)
