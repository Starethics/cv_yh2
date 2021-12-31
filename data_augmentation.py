
import cv2, json, os, numpy as np

def flip(img, axis = 1):
#     1    水平翻转
#     0    垂直翻转
#     -1    水平垂直翻转
    return cv2.flip(img, 1)

def rotate(img, degree = 0):
    if degree == 1:
        img = np.rot90(img, 1)
    elif degree == 2:
        img = np.rot90(img, 2)
    elif degree == 3:
        img = np.rot90(img, -1)
    return img

def gamma(img):
    gm = np.random.uniform(0.7,1.4)
    img = np.power(img/float(np.max(img)), gm)*np.max(img)
    return img.astype(np.uint8)

def blur(img):
    img  = cv2.GaussianBlur(img, (5,5),3)
    return img.astype(np.uint8)

def hsv(img):
    fraction = 0.30
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    S = img_hsv[:, :, 1].astype(np.float32)
    V = img_hsv[:, :, 2].astype(np.float32)

    a = (np.random.random() * 2 - 1) * fraction + 1
    S *= a
    if a > 1:
        np.clip(S, a_min=0, a_max=255, out=S)

    a = (np.random.random() * 2 - 1) * fraction + 1
    V *= a
    if a > 1:
        np.clip(V, a_min=0, a_max=255, out=V)

    img_hsv[:, :, 1] = S.astype(np.uint8)
    img_hsv[:, :, 2] = V.astype(np.uint8)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img
