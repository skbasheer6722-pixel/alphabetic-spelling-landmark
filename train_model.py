"""Train a CNN on rendered hand-landmark images stored in dataset/<LABEL>/."""
import argparse
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models

def load_dataset(data_dir, image_size=128):
    images, labels = [], []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")
    for label in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for name in os.listdir(class_dir):
            if not name.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            path = os.path.join(class_dir, name)
            img = tf.keras.utils.load_img(path, target_size=(image_size, image_size))
            arr = tf.keras.utils.img_to_array(img) / 255.0
            images.append(arr)
            labels.append(label.upper())
    if not images:
        raise ValueError("No images found. Collect data first with collect_data.py.")
    return np.asarray(images, dtype=np.float32), np.asarray(labels)

def build_model(num_classes, image_size=128):
    model = models.Sequential([
        layers.Input(shape=(image_size, image_size, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(192, 3, padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.35),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset")
    parser.add_argument("--output", default="models")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    X, raw_labels = load_dataset(args.data)
    encoder = LabelEncoder()
    y = encoder.fit_transform(raw_labels)
    counts = np.bincount(y)
    if len(encoder.classes_) < 2:
        raise SystemExit("At least two different letter classes are required.")
    if counts.min() < 2:
        raise SystemExit("Each class needs at least 2 images for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    os.makedirs(args.output, exist_ok=True)
    model = build_model(len(encoder.classes_))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.output, "asl_cnn.keras"),
            monitor="val_accuracy", save_best_only=True
        ),
    ]
    model.fit(
        X_train, y_train, validation_split=0.15,
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=callbacks, shuffle=True
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    probabilities = model.predict(X_test, verbose=0)
    predictions = probabilities.argmax(axis=1)
    print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions,
                                target_names=encoder.classes_, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predictions))

    model.save(os.path.join(args.output, "asl_cnn.keras"))
    with open(os.path.join(args.output, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f, indent=2)
    print(f"\nSaved model and labels to: {args.output}")

if __name__ == "__main__":
    main()
