import os
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
from enum import Enum
from typing import List, Dict, NewType

from pyod.models.lof import LOF
from joblib import dump, load
import numpy as np

import utils as u
from config import AD_MODELS_PATH, AD_LANDMARK_SIG_MAX_RATIO, AD_MAX_EVAL_WORKERS, AD_PATH_SIG_MAX_RATIO

class AnomalyLandmark(Enum):
    LIPS_LEFT    = 'lips_left'
    LIPS_RIGT    = 'lips_rigt'
    LIPS_TOP     = 'lips_top'
    LIPS_BOTTOM  = 'lips_bottom'
    LEYE_TOP     = 'leye_top'
    LEYE_BOTTOM  = 'leye_bottom'
    REYE_TOP     = 'reye_top'
    REYE_BOTTOM  = 'reye_bottom'
    LBROW_INNER  = 'lbrow_inner'
    LBROW_CENTER = 'lbrow_center'
    RBROW_INNER  = 'rbrow_inner'
    RBROW_CENTER = 'rbrow_cente'

LandmarkPaths = NewType('LandmarkPaths', Dict[AnomalyLandmark, List[List[float]]])
LandmarksFrame = NewType('LandmarksFrame', Dict[AnomalyLandmark, float])

landmark2i_dict = {
    AnomalyLandmark.LIPS_LEFT: 61,  # lips left corner
    AnomalyLandmark.LIPS_RIGT: 291, # lips rigt corner
    AnomalyLandmark.LIPS_TOP: 0,   # lips top
    AnomalyLandmark.LIPS_BOTTOM: 17,  # lips bottom
    AnomalyLandmark.LEYE_TOP: 159, # left eye top
    AnomalyLandmark.LEYE_BOTTOM: 145, # left eye bottom
    AnomalyLandmark.REYE_TOP: 386, # right eye top
    AnomalyLandmark.REYE_BOTTOM: 374, # right eye bottom
    AnomalyLandmark.LBROW_INNER: 55,  # left brow inner
    AnomalyLandmark.LBROW_CENTER: 52,  # left brow center
    AnomalyLandmark.RBROW_INNER: 285, # right brow inner
    AnomalyLandmark.RBROW_CENTER: 282  # right brow center
}

landmark2axis_dict = {
    AnomalyLandmark.LIPS_LEFT: 'x',
    AnomalyLandmark.LIPS_RIGT: 'x',
    AnomalyLandmark.LIPS_TOP: 'x',
    AnomalyLandmark.LIPS_BOTTOM: 'x',
    AnomalyLandmark.LEYE_TOP: 'y',
    AnomalyLandmark.LEYE_BOTTOM: 'y',
    AnomalyLandmark.REYE_TOP: 'y',
    AnomalyLandmark.REYE_BOTTOM: 'y',
    AnomalyLandmark.LBROW_INNER: 'y',
    AnomalyLandmark.LBROW_CENTER: 'y',
    AnomalyLandmark.RBROW_INNER: 'y',
    AnomalyLandmark.RBROW_CENTER: 'y'
}

models: Dict[str, Dict[AnomalyLandmark, LOF]] = {}

def load_models():
    for gesture in u.class2gesture_dict.values():
        gesture_path = os.path.join(AD_MODELS_PATH, gesture)
        if not os.path.exists(gesture_path): continue
        models[gesture] = {}
        for landmark in AnomalyLandmark:
            model = load(os.path.join(gesture_path, f'{landmark.value}.joblib'))
            models[gesture][landmark] = model 


def train_gesture(gesture: str, landmarks_data: LandmarkPaths):
    gesture_path = os.path.join(AD_MODELS_PATH, gesture)
    os.makedirs(gesture_path, exist_ok=True)
    for landmark, samples in landmarks_data.items():
        model = train_landmark_model(samples)
        dump(model, os.path.join(gesture_path, f'{landmark.value}.joblib'))


def train_landmark_model(samples: List[List[float]]):
    model = LOF(algorithm='auto', contamination=0.01)
    model.fit(samples)
    return model


def eval_gesture(gesture: str, landmark_paths: LandmarkPaths):
    if not gesture in models: return 0., []
    gesture_models = models[gesture]
    landmark_scores = [] 
    with ThreadPoolExecutor(max_workers=min(AD_MAX_EVAL_WORKERS, len(landmark_paths))) as executor:
        futures = [
            executor.submit(eval_landmark_path, gesture_models[landmark], path, landmark) 
            for landmark, path 
            in landmark_paths.items()
        ] 
        res, errors = wait(futures, return_when=FIRST_EXCEPTION)
        if len(errors): print(errors)
        landmark_scores = [r.result() for r in res]
        max_score = max(landmark_scores)
        significant_scores = list(filter(lambda s: s >= max_score*AD_LANDMARK_SIG_MAX_RATIO, landmark_scores))
        uniscore = sum(significant_scores) / len(significant_scores)
    
    return uniscore, landmark_scores


def eval_landmark_path(model: LOF, path: List[List[float]], l):
    scores = model.predict_proba(np.array(path, np.float16))
    outlier_scores =  [s[1] for s in scores.tolist()]
    max_score = max(outlier_scores)
    significant_scores = list(filter(lambda s: s >= max_score*AD_PATH_SIG_MAX_RATIO, outlier_scores))
    uniscore = sum(significant_scores) / len(significant_scores)
    print(l, uniscore, outlier_scores)
    return uniscore
