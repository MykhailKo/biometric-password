import time
from typing import List, Dict, Tuple

import cv2
import mediapipe as mp

import utils as u
from config import AD_CACHE_SIZE, INPUT_TIMEOUT, PASSWORD_LEN
import anomaly_detection as ad
import ui


camera, video_out = u.prepare_video()
landmark_detector, gesture_recognition = u.load_models()
ad.load_models()

ad_cache: List[Tuple[float, ad.LandmarksFrame]] = []

started = False
status = 'Biometric password Verification. Press S to start'
sequence = []
ad_score_sequence = []
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
            max_ad_score, ad_scores = ad.eval_gesture(u.class2gesture_dict[gesture], ad_landmark_paths)
            ad_score_sequence.append(max_ad_score)

    annotated_image = u.draw_landmarks_on_image(input_image.numpy_view(), detection_result) 
    ui_image = ui.status(ui.ascore(ui.sequence(annotated_image, sequence), ad_score_sequence), status)

    video_out.write(ui_image)
    cv2.imshow('Camera', ui_image)

    key = cv2.waitKey(1)
    if key == ord('s'): started = started or True
    if key == ord('q'): break
    if key == ord('d'):
        if len(sequence): sequence.pop()
        if len(ad_score_sequence): ad_score_sequence.pop()
        input_filled = False
        status = ''
    if key == ord('e'):
        if len(sequence) != PASSWORD_LEN: continue
        password_correct = u.check_password(sequence)
        if password_correct: break
        else: status = 'Incorrect password'

camera.release()
video_out.release()
cv2.destroyAllWindows()
