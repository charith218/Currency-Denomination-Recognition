import cv2
import glob
import os
import numpy as np
from .preprocess import preprocess_for_matching

class Predictor:
    """
    Baseline ORB feature matcher. For production, replace with your CNN in __init__/predict.
    """
    def __init__(self, references_dir="data/references"):
        self.references_dir = references_dir
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.refs = []  # list of (label, image, keypoints, descriptors)

        os.makedirs(self.references_dir, exist_ok=True)
        self._load_references()

    def _load_references(self):
        paths = sorted(glob.glob(os.path.join(self.references_dir, "*.*")))
        for p in paths:
            label = os.path.splitext(os.path.basename(p))[0]
            img = cv2.imread(p)
            if img is None:
                continue
            proc, mask = preprocess_for_matching(img)
            kp, des = self.orb.detectAndCompute(proc, mask)
            if des is not None and len(kp) >= 10:
                self.refs.append((label, proc, kp, des))

        if not self.refs:
            # No references found; create synthetic references to keep demo runnable
            self._create_synthetic_refs()

    def _create_synthetic_refs(self):
        for label in ["INR_10", "INR_50", "INR_100", "INR_500"]:
            img = np.full((300, 600, 3), 250, dtype=np.uint8)
            cv2.rectangle(img, (20, 20), (580, 280), (220, 220, 220), 2)
            cv2.putText(img, label, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,0), 4, cv2.LINE_AA)
            proc, mask = preprocess_for_matching(img)
            kp, des = self.orb.detectAndCompute(proc, mask)
            self.refs.append((label, proc, kp, des))

    def predict(self, img_bgr, topk=3):
        img, mask = preprocess_for_matching(img_bgr)
        kp, des = self.orb.detectAndCompute(img, mask)
        if des is None or len(kp) < 10 or not self.refs:
            return [("Unknown", 0.0)]

        scores = []
        for (label, ref_img, ref_kp, ref_des) in self.refs:
            matches = self.matcher.match(des, ref_des)
            if not matches:
                scores.append((label, 0))
                continue
            # Lower distance = better; convert to similarity-like score
            dists = [m.distance for m in matches]
            score = 1.0 / (np.mean(dists) + 1e-6)
            scores.append((label, score))

        # Normalize to pseudo-confidence
        raw = np.array([s for _, s in scores], dtype=np.float32)
        if raw.sum() > 0:
            confs = (raw / raw.sum()).tolist()
        else:
            confs = [0.0] * len(scores)

        ranked = sorted([(lbl, c) for (lbl, _), c in zip(scores, confs)], key=lambda x: x[1], reverse=True)
        return ranked[:topk]
