# Emotion Detection Using Speech & Face – Senior Seminar Project

This is my **Senior Seminar project** — and my **first and only attempt at building an AI model** from scratch. It's not perfect, it's kind of messy, but it's mine.

## 🎯 Goal

I wanted to make something that could **watch YouTube videos** and try to figure out **what emotion the person was feeling**, using both:

- **Facial expressions** (from frames of the video)
- **Voice tone** (from the audio)

It loads preprocessed `.png` image frames and `.wav` audio files, trains two separate models (one for face and one for speech), and then tries to predict emotions using either source.

## 🧠 How It Works

### 1. **Data Loading**
- Loads face data from `.png` images using OpenCV.
- Loads speech data from `.wav` audio files using `librosa` and extracts MFCC features.

### 2. **Model Training**
- **Face model**: Uses `VGG16` with added dense layers.
- **Speech model**: Uses stacked `LSTM` layers with batch normalization.
- Both models are trained with `EarlyStopping`, `ModelCheckpoint`, and learning rate scheduling.

### 3. **Prediction**
- After training, the script predicts emotions from any image, audio, or video in a test folder.
- It handles `.png`, `.wav`, and `.mp4` files differently and tries to return the best guess.

## 🛠 Technologies

- Python
- TensorFlow / Keras
- OpenCV
- librosa
- scikit-learn
- VGG16 (transfer learning)

## 📁 Folder Setup

You’ll need these folders:

emotionDetectionFace/ # Folder with .png face frames (named like happy1.png)
emotionDetectionSpeech/ # Folder with .wav speech clips (named like angry3.wav)
emotionDetectionAll/ # Folder with files to predict on (.png, .wav, .mp4)

## ⚠️ Notes

- This was my **first time working with deep learning, computer vision, and audio processing**.
- Some parts of the code could definitely be refactored, optimized, or cleaned up.
- Paths are **hardcoded** (e.g., `C:\\Users\\gtorr\\...`) so it won’t run on your machine without changes.
- The script expects certain filename formats (emotion name at the beginning).

## 🙃 Final Thoughts

I didn’t expect this to work as well as it did. It’s far from production-ready, but it was **a great learning experience**, and I’m proud of getting it this far.

---

> Built with frustration, too much coffee, and a dash of trial-and-error
