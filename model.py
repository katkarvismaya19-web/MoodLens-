"""CNN architecture, face detection and inference helpers (shared by training and the app)."""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from emotions import COLORS, EMOTIONS

IMG_SIZE = 48
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def conv_block(c_in, c_out):
    """Two 3x3 convolutions + BatchNorm + ReLU, then 2x2 max-pool and dropout."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2), nn.Dropout(0.25),
    )


class EmotionCNN(nn.Module):
    """Input: 1x48x48 grayscale face. Output: scores for 7 emotions.
    48 -> 24 -> 12 -> 6 -> 3 spatial size through four conv blocks."""

    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(conv_block(1, 64), conv_block(64, 128),
                                      conv_block(128, 256), conv_block(256, 512))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512 * 3 * 3, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def detect_faces(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    return gray, faces


class EmotionModel:
    def __init__(self, weights):
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        self.classes = ckpt.get("classes", EMOTIONS)
        self.net = EmotionCNN(len(self.classes))
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.eval()

    @torch.no_grad()
    def predict_faces(self, bgr):
        """Returns a list of {box, emotion, confidence, probs} for each face, largest first."""
        gray, faces = detect_faces(bgr)
        if len(faces) == 0:
            return []
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        crops = [cv2.resize(gray[y:y + h, x:x + w], (IMG_SIZE, IMG_SIZE),
                            interpolation=cv2.INTER_AREA) for x, y, w, h in faces]
        batch = torch.from_numpy(np.stack(crops)).float().div(255).sub(0.5).div(0.5).unsqueeze(1)
        probs = torch.softmax(self.net(batch), dim=1).numpy()
        out = []
        for (x, y, w, h), p in zip(faces, probs):
            i = int(p.argmax())
            out.append({"box": (int(x), int(y), int(w), int(h)), "emotion": self.classes[i],
                        "confidence": float(p[i]),
                        "probs": {c: float(v) for c, v in zip(self.classes, p)}})
        return out


def draw_results(bgr, results):
    for n, r in enumerate(results, 1):
        x, y, w, h = r["box"]
        color = COLORS.get(r["emotion"], (255, 255, 255))
        text = f"#{n} {r['emotion'].capitalize()} {r['confidence']:.0%}"
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(bgr, (x, max(0, y - th - 10)), (x + tw + 8, y), color, -1)
        cv2.putText(bgr, text, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 2, cv2.LINE_AA)
    return bgr


def default_weights():
    return Path(__file__).resolve().parent / "models" / "emotion_cnn.pt"
