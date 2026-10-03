"""
Real-time emotion recognition from the webcam.

  python detect_live.py                 # just view
  python detect_live.py --user vismaya  # also log your moods to the dashboard
Press Q to stop.
"""
import argparse
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

import db
from emotions import COLORS
from model import EmotionModel, default_weights, draw_results

p = argparse.ArgumentParser()
p.add_argument("--weights", default=str(default_weights()))
p.add_argument("--camera", type=int, default=0)
p.add_argument("--user", help="username to log this session for")
args = p.parse_args()

if not Path(args.weights).exists():
    raise SystemExit(f"Model not found: {args.weights}. Train it first (see README).")

model = EmotionModel(args.weights)
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit("Could not open webcam.")

session_id = datetime.now().strftime("%Y%m%d_%H%M%S") if args.user else None
if args.user:
    db.init()
start, last_log, counts = time.time(), 0.0, Counter()


def draw_panel(frame, probs):
    """Probability bars for the main face in the top-left corner."""
    cv2.rectangle(frame, (8, 8), (228, 30 + 22 * len(probs)), (25, 25, 25), -1)
    for i, (emo, p) in enumerate(sorted(probs.items(), key=lambda kv: -kv[1])):
        y = 30 + 22 * i
        cv2.putText(frame, emo[:8], (14, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (90, y - 8), (90 + int(130 * p), y + 6), COLORS[emo], -1)


while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)   # mirror view feels natural
    results = model.predict_faces(frame)
    draw_results(frame, results)

    if results:
        main = results[0]          # largest face
        counts[main["emotion"]] += 1
        draw_panel(frame, main["probs"])
        if args.user and time.time() - last_log >= 1:
            db.add_detection(args.user, "live", main["emotion"], main["confidence"], session_id)
            last_log = time.time()

    elapsed = int(time.time() - start)
    status = f"{elapsed // 60:02d}:{elapsed % 60:02d}  " + ("REC  " if args.user else "") + "Q = stop"
    h = frame.shape[0]
    cv2.putText(frame, status, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255) if args.user else (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("MoodLens - live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
if counts:
    total = sum(counts.values())
    print("\nSession summary:")
    for emo, n in counts.most_common():
        print(f"  {emo:9s} {n / total:.0%}")
