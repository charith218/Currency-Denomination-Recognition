import cv2
import numpy as np

def resize_keep_aspect(img, max_side=640):
    h, w = img.shape[:2]
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    nh, nw = int(h * scale), int(w * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), scale

def deskew(img):
    # Simple skew correction via largest contour minAreaRect
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    c = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def make_mask(img):
    # Rough background removal
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 10)
    kernel = np.ones((5,5), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    return thr

def preprocess_for_matching(img_bgr):
    img, _ = resize_keep_aspect(img_bgr, 640)
    img = deskew(img)
    mask = make_mask(img)
    return img, mask
