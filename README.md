# MoodLens 🎭

MoodLens is a machine learning project that detects and analyzes mood/emotion. It includes a training pipeline, a saved model, and evaluation metrics.

## Features

- Trains a mood/emotion classification model
- Saves the trained model in the `models/` folder
- Records evaluation metrics in `models/metrics.json`
- Simple setup with all dependencies listed in `requirements.txt`

## Project Structure

```
moodlens/
├── models/
│   └── metrics.json      # Model evaluation results
├── train.py              # Script to train the model
├── requirements.txt      # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

1. Clone the repository:
```bash
   git clone https://github.com/katkarvismaya19-web/MoodLens-.git
   cd MoodLens-
```

2. (Optional) Create and activate a virtual environment:
```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

## Usage

To train the model:

```bash
python train.py
```

After training, the model and its metrics will be saved in the `models/` folder.

## Results

Model performance metrics (such as accuracy) are stored in `models/metrics.json`.

## Tech Stack

- Python
- Machine learning libraries listed in `requirements.txt`

## Future Improvements

- Add a user interface for real-time mood detection
- Improve model accuracy with more training data
- Deploy as a web app

## Author

**Vismaya Katkar**
GitHub: [@katkarvismaya19-web](https://github.com/katkarvismaya19-web)

## License

This project is open source and available for learning and personal use.
