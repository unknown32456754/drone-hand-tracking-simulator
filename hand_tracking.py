"""
Hand tracking + finger counter (laptop webcam only)

SETUP (run once in a terminal):
    pip install opencv-python mediapipe==0.10.14

Needs Python 3.9 - 3.12.

RUN:
    python hand_tracking.py

Press 'q' to quit.
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Landmark ids from MediaPipe: 4=thumb tip, 8=index tip, 12=middle tip,
# 16=ring tip, 20=pinky tip. The joint two steps below each tip is used
# to decide if the finger is raised.
FINGER_TIPS = [8, 12, 16, 20]


def count_fingers(hand_landmarks, label):
    """Return how many fingers are raised (0-5)."""
    lm = hand_landmarks.landmark
    count = 0

    # Four fingers: raised if the tip is higher on screen (smaller y)
    # than the joint below it.
    for tip in FINGER_TIPS:
        if lm[tip].y < lm[tip - 2].y:
            count += 1

    # Thumb moves sideways, so compare x instead of y.
    # The frame is mirrored, so the direction depends on which hand it is.
    if label == "Right":
        if lm[4].x < lm[3].x:
            count += 1
    else:
        if lm[4].x > lm[3].x:
            count += 1

    return count


def main():
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror view, feels natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = handedness.classification[0].label  # "Left"/"Right"

                    # Draw the 21 points and the lines connecting them
                    mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                    fingers = count_fingers(hand, label)

                    # Put the number near the wrist
                    h, w, _ = frame.shape
                    wrist = hand.landmark[0]
                    x, y = int(wrist.x * w), int(wrist.y * h)
                    cv2.putText(
                        frame,
                        f"{label}: {fingers}",
                        (x - 40, y + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

            cv2.imshow("Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

