"""Generate evaluation metrics and a confusion-matrix image for a trained model."""
import argparse
import json
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from train_model import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--model", default="models/asl_cnn.keras")
    parser.add_argument("--labels", default="models/labels.json")
    parser.add_argument("--output", default="evaluation")
    args = parser.parse_args()
    X, raw_labels = load_dataset(args.data)
    with open(args.labels, encoding="utf-8") as f:
        classes = json.load(f)
    mapping = {label: i for i, label in enumerate(classes)}
    y = np.array([mapping[label] for label in raw_labels])
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = tf.keras.models.load_model(args.model)
    pred = model.predict(X_test, verbose=0).argmax(axis=1)
    print(classification_report(y_test, pred, target_names=classes, zero_division=0))
    os.makedirs(args.output, exist_ok=True)
    cm = confusion_matrix(y_test, pred, labels=np.arange(len(classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "confusion_matrix.png"), dpi=180)
    print("Saved evaluation/confusion_matrix.png")

if __name__ == "__main__":
    main()
