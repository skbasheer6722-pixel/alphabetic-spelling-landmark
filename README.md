# ASL Alphabet Fingerspelling Recognition using CNN on Hand-Landmark Images

A complete Python prototype that captures webcam hand gestures, detects 21 hand landmarks with MediaPipe, renders those landmarks as a 128×128 image, and classifies the image using a custom CNN.

## Important scope and limitation

This prototype recognizes **static handshapes**. ASL **J** and **Z** involve motion, so a single-frame static CNN cannot recognize them reliably. For a complete alphabet system, collect short landmark sequences and add a temporal model (for example, an LSTM/GRU) for J and Z. Do not report those two as supported unless you implement and test that sequence model.

This is a student project starter, not a pre-trained ASL model. You must collect labeled samples and train it on your own camera, hand, lighting and background conditions. Accuracy depends on dataset quality and testing with people not represented in the training set.

## 1. Requirements

Recommended environment: **64-bit Windows, Python 3.10**. Python 3.10 is recommended because TensorFlow 2.15 and MediaPipe 0.10.14 have compatible prebuilt packages for it.

Install Python 3.10, then open Command Prompt in this project folder:

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`. Make sure your webcam is connected and allowed by the operating system.

## 2. Collect training images

Collect each letter separately. The command opens your webcam. Show the requested ASL handshape and press **C** to save a sample; press **Q** to stop.

```bash
python collect_data.py --label A --samples 250
python collect_data.py --label B --samples 250
python collect_data.py --label C --samples 250
```

Repeat for each static letter you intend to classify. Use varied positions, distances and lighting. Keep the hand fully visible. Collect at least 150–300 examples per class as a starting point, preferably across multiple sessions. Avoid near-duplicate frames. If collecting J/Z, note that this static-image pipeline does not capture their movement; leave them out of the static model.

The folder structure is created automatically:

```text
dataset/
  A/
    A_....png
  B/
    B_....png
```

Each image is a rendered hand skeleton generated from the normalized MediaPipe landmarks. This same renderer is used at inference time, preventing preprocessing mismatch.

## 3. Train the CNN

```bash
python train_model.py --data dataset --epochs 35 --batch-size 32
```

The script creates `models/asl_cnn.keras` and `models/labels.json`, prints test accuracy, precision, recall, F1-score and a confusion matrix. The test split is stratified and reproducible. Do not treat this random image-level split as a strong generalization test if adjacent frames or the same signer occur in both splits; for a report, also test on a separate recording session or signer.

## 4. Run live recognition

```bash
python realtime_recognition.py
```

Press **Q** to close. If the camera is not the default camera, pass `--camera 1`. Adjust the confidence cutoff with `--threshold 0.70`.

## 5. Create an evaluation confusion-matrix figure

```bash
python evaluate_model.py
```

The figure is saved to `evaluation/confusion_matrix.png`.

## Project pipeline

1. Webcam capture using OpenCV.
2. Hand detection and 21-point landmark tracking using MediaPipe Hands.
3. Translation/scale normalization of x/y coordinates.
4. Rendering landmarks and hand connections into a 128×128 three-channel image.
5. CNN feature extraction and alphabet-class prediction.
6. Confidence-thresholded output overlaid on the live camera feed.
7. Evaluation using accuracy, precision, recall, F1-score and confusion matrix.

## Files

- `hand_image.py` — shared landmark-to-image preprocessing.
- `collect_data.py` — labeled webcam sample collection.
- `train_model.py` — CNN training, held-out evaluation and model export.
- `realtime_recognition.py` — real-time webcam prediction.
- `evaluate_model.py` — classification report and confusion-matrix plot.
- `requirements.txt` — pinned dependencies.

## Troubleshooting

- **No webcam:** close other camera applications; try `--camera 1`.
- **MediaPipe/TensorFlow installation errors:** use 64-bit Python 3.10 in a clean virtual environment; do not install NumPy 2.x with this pinned stack.
- **Poor accuracy:** collect more balanced samples, include different hand positions and lighting, ensure each class is correctly labeled, and test with a separate signer/session.
- **Wrong predictions:** increase `--threshold`, improve data quality, and ensure the same hand visibility and preprocessing are used during collection and inference.

## Academic reporting note

Report measured results from your own experiment; do not invent accuracy values. Include the number of samples per class, signer/session split, hardware, Python/library versions, test-set metrics and confusion matrix. This prototype is an alphabet fingerspelling aid and is not a replacement for a qualified interpreter.
