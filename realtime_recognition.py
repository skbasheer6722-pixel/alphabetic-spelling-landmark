"""Real-time ASL static-letter recognition with webcam + MediaPipe + CNN."""
import argparse
import json
import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from hand_image import landmarks_to_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/asl_cnn.keras")
    parser.add_argument("--labels", default="models/labels.json")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()

    if not os.path.isfile(args.model) or not os.path.isfile(args.labels):
        raise SystemExit("Model files missing. Run train_model.py after collecting data.")
    model = tf.keras.models.load_model(args.model)
    with open(args.labels, "r", encoding="utf-8") as f:
        labels = json.load(f)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera index/permissions.")

    displayed_label, confidence = "No hand", 0.0
    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.65, min_tracking_confidence=0.6
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                input_image = landmarks_to_image(hand).astype(np.float32) / 255.0
                prediction = model.predict(input_image[None, ...], verbose=0)[0]
                index = int(np.argmax(prediction))
                confidence = float(prediction[index])
                displayed_label = labels[index] if confidence >= args.threshold else "Uncertain"
                preview = (input_image * 255).astype(np.uint8)
                cv2.imshow("CNN landmark input", preview)
            else:
                displayed_label, confidence = "No hand", 0.0

            text = f"{displayed_label}  ({confidence*100:.1f}%)"
            cv2.rectangle(frame, (8, 8), (430, 58), (25, 25, 25), -1)
            cv2.putText(frame, text, (18, 43), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Q: quit", (18, 86), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            cv2.imshow("ASL alphabet recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
