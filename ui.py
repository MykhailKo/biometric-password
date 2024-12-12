import cv2
from utils import class2gesture_dict
from utils import VID_HEIGHT

def sequence(image, sequence):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_stroke = cv2.LINE_AA 
    font_color = (255, 0, 50)
    font_size = 0.6
    font_weight = 2
    position = (30, 30)

    delimiter = '   '
    text = delimiter.join([class2gesture_dict[g] for g in sequence])
    cv2.putText(image, text, position, font, font_size, font_color, font_weight, font_stroke)
    return image


def ascore(image, scores):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_stroke = cv2.LINE_AA 
    font_color = (46, 28, 206)
    font_size = 0.6
    font_weight = 2
    position = (30, 70)

    delimiter = '   '
    text = delimiter.join([str(round(s, 2)) for s in scores])
    cv2.putText(image, text, position, font, font_size, font_color, font_weight, font_stroke)
    return image


def status(image, status):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_stroke = cv2.LINE_AA 
    font_color = (151, 198, 57)
    font_size = 0.6
    font_weight = 2
    position = (30, 110)

    image = manual(image)
    cv2.putText(image, status, position, font, font_size, font_color, font_weight, font_stroke)
    return image


def manual(image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_stroke = cv2.LINE_AA 
    font_color = (151, 198, 57)
    font_size = 0.7
    font_weight = 2
    position = (30, VID_HEIGHT - 30)

    manual_text = 'Press S to start scenario. Press Q to quit. Press E to confirm password. Press D to delete last input'
    cv2.putText(image, manual_text, position, font, font_size, font_color, font_weight, font_stroke)
    return image