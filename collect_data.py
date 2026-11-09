"""Collect labeled landmark-render images using a webcam.

Example:
    python collect_data.py --label A --samples 250
Repeat for each desired letter (A through Z). Press C to save, Q to quit.
"""
import argparse
import os
import time
import cv2
import mediapipe as mp
from hand_image import landmarks_to_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="ASL class label, e.g. A")
    parser.add_argument("--samples", type=int, default=250)
    parser.add_argument("--output", default="dataset")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    label = args.label.strip().upper()
    if len(label) != 1 or not label.isalpha():
        raise SystemExit("Label must be one alphabet letter A-Z.")
    save_dir = os.path.join(args.output, label)
    os.makedirs(save_dir, exist_ok=True)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open webcam. Check camera index/permissions.")

    saved = len([f for f in os.listdir(save_dir) if f.lower().endswith(".png")])
    print("Show the sign for", label, "| press C to capture, Q to quit.")
    print("Vary hand position, distance and lighting while keeping the same sign.")
    with mp_hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=0.65, min_tracking_confidence=0.6
    ) as hands:
        while saved < args.samples:
            ok, frame = cap.read()
            if not ok:
                print("Unable to read from webcam.")
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            preview = None
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                preview = landmarks_to_image(hand)

            cv2.putText(frame, f"Label: {label}  Saved: {saved}/{args.samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "C: capture | Q: quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("ASL data collection", frame)
            if preview is not None:
                cv2.imshow("CNN input preview", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and preview is not None:
                filename = os.path.join(save_dir, f"{label}_{int(time.time()*1000)}_{saved:04d}.png")
                cv2.imwrite(filename, preview)
                saved += 1
                print(f"Saved {saved}/{args.samples}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Finished. {saved} images in {save_dir}")

if __name__ == "__main__":
    main()
