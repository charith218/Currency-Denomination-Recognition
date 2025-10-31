import argparse
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from .classifier import Predictor

SAMPLES = [
    ("samples/sample_10.png", "INR_10"),
    ("samples/sample_50.png", "INR_50"),
    ("samples/sample_100.png", "INR_100"),
    ("samples/sample_500.png", "INR_500"),
]

def _ensure_samples():
    os.makedirs("samples", exist_ok=True)
    for path, label in SAMPLES:
        if not os.path.exists(path):
            img = Image.new("RGB", (900, 450), (245, 245, 245))
            d = ImageDraw.Draw(img)
            d.rectangle([20, 20, 880, 430], outline=(200,200,200), width=6)
            d.text((60, 160), label, fill=(0,0,0))
            img.save(path)

def run_demo(image_path: str):
    pred = Predictor()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    results = pred.predict(img, topk=3)
    print("Top predictions:")
    for label, conf in results:
        print(f"  {label:>10}  {conf*100:.1f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", type=str, help="Path to image to test")
    args = parser.parse_args()

    _ensure_samples()

    if args.demo:
        run_demo(args.demo)
    else:
        print("Nothing to do. Try: python -m scanner.utils --demo samples/sample_100.png")

if __name__ == "__main__":
    main()
