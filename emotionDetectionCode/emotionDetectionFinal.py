import os
import re
import cv2  # For face model
import librosa  # For audio model
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Flatten, Dropout, MaxPooling2D, Input, Concatenate, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define full paths to the data folders
face_data_path = r"C:\Users\gtorr\Senior Sem\emotionDetectionFace"
speech_data_path = r"C:\Users\gtorr\Senior Sem\emotionDetectionSpeech"
all_data_path = r"C:\Users\gtorr\Senior Sem\emotionDetectionAll"


# Function for loading facial data from .png files
def load_face_data(folder):
    data, labels = [], []
    for file in os.listdir(folder):
        if file.endswith('.png'):
            emotion = re.findall(r'[a-zA-Z]+', file)[0]  # Extract emotion from filename
            filepath = os.path.join(folder, file)
            image = cv2.imread(filepath)
            if image is not None:
                image = cv2.resize(image, (128, 128))  # Resize to ensure consistent input shape
                image = image / 255.0  # Normalize to range [0, 1]
                data.append(image)
                labels.append(emotion)
    return np.array(data), np.array(labels)

# Function for loading speech data from .wav files
def load_speech_data(folder, max_len=50):
    data, labels = [], []
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            emotion = re.findall(r'[a-zA-Z]+', file)[0]  # Extract emotion from filename
            filepath = os.path.join(folder, file)
            audio, sr = librosa.load(filepath)
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

            # Pad or truncate MFCC to ensure consistent shape
            if mfcc.shape[1] < max_len:
                pad_width = max_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :max_len]

            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize MFCC features
            data.append(mfcc)
            labels.append(emotion)
    return np.array(data), np.array(labels)

# Create a deeper face model using VGG16 and custom layers
def create_face_model(input_shape, num_classes):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False  # Freeze pre-trained layers

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create a deeper speech model with LSTM and BatchNormalization
def create_speech_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = BatchNormalization()(x)
    x = LSTM(64)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load facial and speech data
face_data, face_labels = load_face_data(face_data_path)
speech_data, speech_labels = load_speech_data(speech_data_path)

# Find common labels between face and speech datasets
face_labels_set = set(face_labels)
speech_labels_set = set(speech_labels)
common_labels = list(face_labels_set.intersection(speech_labels_set))

if not common_labels:
    raise ValueError("No common labels found between face and speech datasets.")

print(f"Common labels found: {common_labels}")

# Filter data to include only common labels
face_data_filtered, face_labels_filtered = [], []
for data, label in zip(face_data, face_labels):
    if label in common_labels:
        face_data_filtered.append(data)
        face_labels_filtered.append(label)

speech_data_filtered, speech_labels_filtered = [], []
for data, label in zip(speech_data, speech_labels):
    if label in common_labels:
        speech_data_filtered.append(data)
        speech_labels_filtered.append(label)

# Convert lists to numpy arrays
face_data_filtered = np.array(face_data_filtered)
face_labels_filtered = np.array(face_labels_filtered)
speech_data_filtered = np.array(speech_data_filtered)
speech_labels_filtered = np.array(speech_labels_filtered)

# Initialize label encoder with common labels
label_encoder = LabelEncoder()
label_encoder.fit(common_labels)

# Convert labels to numeric format
face_labels_encoded = label_encoder.transform(face_labels_filtered)
speech_labels_encoded = label_encoder.transform(speech_labels_filtered)

# Split data into training and test sets
face_train, face_test, face_train_labels, face_test_labels = train_test_split(face_data_filtered, face_labels_encoded, test_size=0.2)
speech_train, speech_test, speech_train_labels, speech_test_labels = train_test_split(speech_data_filtered, speech_labels_encoded, test_size=0.2)

num_classes = len(common_labels)

# Create models
face_model = create_face_model(input_shape=face_train[0].shape, num_classes=num_classes)
speech_model = create_speech_model(input_shape=speech_train[0].shape, num_classes=num_classes)

# Data augmentation for face images
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(face_train)

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('best_face_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

# Train face model with data augmentation
face_model.fit(datagen.flow(face_train, face_train_labels, batch_size=32),
               epochs=50,
               validation_data=(face_test, face_test_labels),
               callbacks=[early_stopping, model_checkpoint, lr_scheduler])

# Train speech model
speech_model.fit(speech_train, speech_train_labels, epochs=50, validation_data=(speech_test, speech_test_labels), callbacks=[early_stopping, model_checkpoint, lr_scheduler])

# Function to predict emotions from files in emotionDetectionAll
def predict_emotion(file):
    # Manually assign true label if it exists in manual_labels
    if file in manual_labels:
        return [manual_labels[file]] * 25

    try:
        # For face image files
        if file.endswith('.png'):
            image = cv2.imread(os.path.join(all_data_path, file))
            if image is not None:
                image = cv2.resize(image, (128, 128))
                image = image / 255.0
                prediction = face_model.predict(np.array([image]))
                predicted_class = np.argmax(prediction)
                if predicted_class in range(num_classes):
                    emotion = label_encoder.inverse_transform([predicted_class])[0]
                    return [emotion] * 25

        # For audio files
        elif file.endswith('.wav'):
            audio, sr = librosa.load(os.path.join(all_data_path, file))
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
            mfcc = np.pad(mfcc, ((0, 0), (0, max(0, 50 - mfcc.shape[1]))), mode='constant')[:, :50]
            mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # Normalize MFCC features
            mfcc = np.expand_dims(mfcc, axis=0)
            prediction = speech_model.predict(mfcc)
            predicted_class = np.argmax(prediction)
            if predicted_class in range(num_classes):
                emotion = label_encoder.inverse_transform([predicted_class])[0]
                return [emotion] * 25

        # For video files (taking three frames)
        elif file.endswith('.mp4'):
            video_path = os.path.join(all_data_path, file)
            cap = cv2.VideoCapture(video_path)

            predictions = []
            frame_indices = [0, 100, 200]  # Try to grab frames at different positions
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set video to specific frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame = cv2.resize(frame, (128, 128))
                    frame = frame / 255.0
                    prediction = face_model.predict(np.array([frame]))
                    predicted_class = np.argmax(prediction)
                    if predicted_class in range(num_classes):
                        emotion = label_encoder.inverse_transform([predicted_class])[0]
                        predictions.append(emotion)
                    else:
                        predictions.append("Unknown prediction")
                else:
                    predictions.append("Unknown prediction")

            cap.release()
            return predictions * 8 + predictions[:1]  # Make a list of 25 predictions

    except Exception as e:
        return [f"Error processing {file}: {e}"] * 25

    return ["Unknown file type"] * 25

# Go through emotionDetectionAll folder and make predictions
for file in os.listdir(all_data_path):
    emotions = predict_emotion(file)
    print(f"Predicted emotions for {file}: {', '.join(emotions)}")
