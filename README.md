# Currency Denomination Recognition – Photo Scanner (Starter Pack)

A **ready-to-run** Streamlit app you can open in **PyCharm**. It provides:
- An upload UI for banknote photos
- Preprocessing (resize, deskew, background mask)
- A **baseline matcher** using ORB features against reference images
- A scaffold to plug-in your own deep learning model (TensorFlow/PyTorch not required to run the demo)
- A simple **training stub** (`train_tf.py`) if you want to train a CNN later

> ⚠️ The baseline matcher is only a starter; for real accuracy, replace `data/references/` with real banknote images for each denomination and/or train a CNN and call it from `scanner/classifier.py`.

---

## 1) Quick Start

### Option A — Run app
```bash
# (Recommended) Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app_streamlit.py
```
The app opens in your browser. Upload a banknote photo to get a predicted denomination and confidence score.

### Option B — Run as a module (CLI demo)
```bash
python -m scanner.utils --demo samples/sample_100.png
```

---

## 2) Project Structure

```
currency_denom_scanner/
├─ app_streamlit.py           # Streamlit UI
├─ scanner/
│  ├─ __init__.py
│  ├─ preprocess.py           # image cleanup, deskew, ROI
│  ├─ classifier.py           # baseline ORB matcher + hooks for DL model
│  └─ utils.py                # I/O helpers and reference bootstrap
├─ data/
│  └─ references/             # reference images per denomination (auto-created on first run)
├─ samples/
│  ├─ sample_10.png           # synthetic samples for demo
│  ├─ sample_50.png
│  ├─ sample_100.png
│  └─ sample_500.png
├─ requirements.txt
├─ train_tf.py                # OPTIONAL: CNN training scaffold
└─ README.md
```

---

## 3) Add Real References (no ML needed)

1. Collect **clear, front-side** images for each denomination (same currency).
2. Place them under `data/references/` with file names like:
   - `INR_10.jpg`, `INR_20.jpg`, `INR_50.jpg`, `INR_100.jpg`, `INR_200.jpg`, `INR_500.jpg`, `INR_2000.jpg`
3. The baseline ORB matcher will use whichever files it finds. More images per class = better robustness.

---

## 4) Plug-in a Deep Learning Model (optional)

- Train any CNN and export a model (e.g., TensorFlow SavedModel or ONNX).
- Update `scanner/classifier.py` in the `Predictor` class:
  - Load your model once in `__init__`
  - Use it in `predict()` to return `[(label, confidence), ...]`

A starting point is in `train_tf.py` (not required for the demo).

---

## 5) Notes & Tips

- Good lighting and minimal glare improve results a lot.
- For speed on low-power devices, crop to the note area before predicting.
- If you plan to support **multiple currencies**, prefix labels (e.g., `INR_100`, `USD_1`, `EUR_10`).

---

## 6) License

MIT. Use at your own risk.
