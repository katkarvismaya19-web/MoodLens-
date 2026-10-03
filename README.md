# 🎭 MoodLens: Face Emotion Recognition System

A deep learning web app that recognises human emotions from faces in photos and live webcam
video. Users log in, analyse photos, record live mood sessions, and track their emotional
patterns on a personal dashboard.

**Tech:** Python, PyTorch (custom CNN), OpenCV, Streamlit, SQLite, Pandas
**Dataset:** FER-2013 (Kaggle: msambare/fer2013), 35,887 grayscale 48×48 face images, 7 emotions

## Features

- 🔐 Login and signup with salted PBKDF2-SHA256 password hashing (SQLite)
- 📊 Dashboard: detections, live sessions, top mood, positivity score, mood charts over time
- 📷 Analyze Photo: upload or camera, multiple faces, per-face emotion probabilities
- 🎥 Live Session: real-time webcam recognition that logs your mood every second, with timeline
- 📜 History: filter by source or emotion, photo gallery, CSV export, delete records
- 🧠 About Model: test accuracy, per-emotion accuracy, training curves, confusion matrix

## Emotions recognised

😠 Angry · 🤢 Disgust · 😨 Fear · 😄 Happy · 😐 Neutral · 😢 Sad · 😲 Surprise

## How it works

```
Image / Webcam frame
        │
        ▼
OpenCV Haar Cascade  ──►  finds face locations
        │
        ▼
Crop → Grayscale → Resize 48×48 → Normalise
        │
        ▼
Custom CNN (PyTorch)  ──►  Softmax probabilities for 7 emotions
        │
        ▼
Draw boxes + labels, save to SQLite, show on dashboard
```

### CNN architecture (built from scratch, about 5.8M parameters)

```
Input 1×48×48
→ Conv block 1:  2×(Conv3×3, 64)  + BatchNorm + ReLU → MaxPool → Dropout 0.25   (24×24)
→ Conv block 2:  2×(Conv3×3, 128) + BatchNorm + ReLU → MaxPool → Dropout 0.25   (12×12)
→ Conv block 3:  2×(Conv3×3, 256) + BatchNorm + ReLU → MaxPool → Dropout 0.25   (6×6)
→ Conv block 4:  2×(Conv3×3, 512) + BatchNorm + ReLU → MaxPool → Dropout 0.25   (3×3)
→ Flatten → Dense 256 → BatchNorm → ReLU → Dropout 0.5 → Dense 7 → Softmax
```

### Training setup

| Setting | Value |
|---|---|
| Loss | Cross-entropy with label smoothing 0.1 |
| Optimizer | AdamW (weight decay 1e-4) |
| LR schedule | One-Cycle, max LR 3e-3 |
| Epochs / batch | 40 / 128 |
| Augmentation | Random crop, horizontal flip, ±10° rotation, brightness/contrast jitter |
| Speed-up | Mixed precision (AMP) on GPU |

Expected test accuracy is about **65–70%**. Human agreement on FER-2013 labels is
only about 65%, so this is a strong result for this dataset.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit app (login, dashboard, analyze, live, history, about) |
| `model.py` | CNN architecture, face detection, inference, drawing |
| `train.py` | Dataset download, training loop, evaluation, graphs |
| `detect_live.py` | Real-time webcam window with mood logging |
| `db.py` | SQLite users and detections |
| `emotions.py` | Emotion labels, emojis, colours, positivity score |
| `models/` | `emotion_cnn.pt`, `metrics.json`, `history.csv`, `graphs/` |

## 1. Train (Google Colab, about 15–25 min)

Upload `MoodLens_Build.ipynb` to colab.research.google.com, choose
**Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**, and upload
`kaggle.json` when asked. It downloads `moodlens.zip` with the trained model inside.

## 2. Run on your laptop (Windows cmd)

```
cd "path\to\moodlens"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, sign up, and log in.
Webcam only, without the dashboard: `python detect_live.py` (press Q to quit).

## Report outline

1. Introduction: affective computing, applications (e-learning engagement, driver
   monitoring, customer feedback, mental-wellbeing apps)
2. Literature survey: hand-crafted features (LBP, HOG + SVM) vs CNNs on FER-2013
3. Dataset: FER-2013 classes, image counts, class imbalance (disgust has very few images)
4. Methodology: Haar cascade face detection, CNN layers, BatchNorm, Dropout, augmentation,
   label smoothing, One-Cycle LR
5. System design: architecture diagram, database schema, login flow
6. Results: accuracy, per-class accuracy, training curves, confusion matrix, screenshots
7. Conclusion and future work: transfer learning (ResNet/EfficientNet), deep face detector,
   video-based temporal models, more diverse datasets

## Likely viva questions

- **Why build a CNN instead of using a pretrained model?** To design and understand each
  layer; FER images are small 48×48 grayscale, so a compact custom CNN works well.
- **What does BatchNorm do?** Normalises each layer's outputs so training is faster and stabler.
- **What does Dropout do?** Randomly switches off neurons during training to prevent overfitting.
- **Why is 'disgust' accuracy low?** It has only about 440 training images versus about 7,000 for
  'happy' (class imbalance). Fear vs surprise and sad vs neutral are also visually similar.
- **What is label smoothing?** Uses targets like 0.9 instead of 1.0 so the model is less
  over-confident, which helps with noisy labels.
- **How is face detection different from emotion recognition?** The Haar cascade only finds
  *where* faces are; the CNN decides *which emotion* each face shows.
- **Positivity score?** Happy + surprise as a share of all non-neutral readings.

## Ethics note

Facial expressions don't always reflect how someone actually feels, and accuracy varies with
lighting, pose, age and ethnicity. MoodLens is an educational project, not a medical or
psychological tool, and should only be used on people who have agreed to it.
