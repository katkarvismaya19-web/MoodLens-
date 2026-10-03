"""Emotion labels, emojis, colours and simple mood scoring (no deep learning imports)."""

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOJI = {"angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
         "neutral": "😐", "sad": "😢", "surprise": "😲"}

# BGR colours for OpenCV drawing
COLORS = {"angry": (40, 40, 220), "disgust": (40, 140, 60), "fear": (160, 60, 160),
          "happy": (0, 200, 255), "neutral": (180, 180, 180), "sad": (200, 120, 40),
          "surprise": (0, 140, 255)}

POSITIVE = {"happy", "surprise"}
NEGATIVE = {"angry", "disgust", "fear", "sad"}


def label(emotion: str) -> str:
    return f"{EMOJI.get(emotion, '')} {emotion.capitalize()}"


def positivity(emotions) -> float:
    """Share of positive moods among non-neutral detections (0-1). None if no data."""
    pos = sum(e in POSITIVE for e in emotions)
    neg = sum(e in NEGATIVE for e in emotions)
    return pos / (pos + neg) if pos + neg else None
