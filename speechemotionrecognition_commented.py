
"""
SpeechEmotionRecognition.ipynb

This script implements a speech emotion recognition system using audio features.
It utilizes datasets like RAVDESS and TESS and extracts features such as MFCC 
(Mel-frequency cepstral coefficients) and Mel spectrogram. The processed data 
is then used to train a deep learning model via PyTorch.

"""

import os
import numpy as np
import librosa  # Library for audio processing
from tqdm import tqdm  # Progress bar for loops
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # For standardizing data and encoding labels
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score  # For evaluating model performance
import matplotlib.pyplot as plt  # For plotting graphs and visualizations
import seaborn as sns  # For enhanced plotting
import torch  # PyTorch library for deep learning
import torch.nn as nn  # Neural network modules in PyTorch
import torch.optim as optim  # Optimizers for training models
from torch.utils.data import Dataset, DataLoader  # Data utilities in PyTorch for batching and loading

# Function to extract MFCC and Mel spectrogram features from an audio file
def extract_features(file_path):
    # Load the audio file using librosa with its native sample rate
    y, sr = librosa.load(file_path, sr=None)
    # Extract 13-dimensional MFCC and take the mean along the time axis
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
    # Extract Mel spectrogram and take the mean along the time axis
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    # Combine MFCC and Mel spectrogram features into one array
    return np.hstack((mfcc, mel))

# Directory paths where the datasets (RAVDESS and TESS) are located
RAVDESS_DIR = 'ravdess_dataset'
TESS_DIR = 'tess_dataset'

# Function to extract emotion from RAVDESS dataset filenames
# RAVDESS filenames contain emotion information at index 2 when split by '-'
def extract_ravdess_emotion(filename):
    return filename.split('-')[2]

# Function to extract emotion from TESS dataset filenames
# TESS filenames store emotion information after the last underscore ('_')
def extract_tess_emotion(filename):
    emotion_str = filename.split('_')[-1].split('.')[0]
    return emotion_str

# Function to process a dataset directory and extract features and labels
# directory: the dataset directory (e.g., RAVDESS or TESS)
# emotion_map: a dictionary mapping emotion labels (from filenames) to a canonical set of emotions
# extract_emotion_func: a function to extract emotion from a filename
def process_directory(directory, emotion_map, extract_emotion_func):
    features, labels = [], []
    # Walk through all subdirectories and files within the dataset directory
    for subdir, _, files in os.walk(directory):
        for file in tqdm(files):  # Show progress using tqdm
            if file.endswith('.wav'):  # Only process .wav files
                file_path = os.path.join(subdir, file)  # Get the full path of the file
                # Extract emotion key from filename using the provided extraction function
                emotion_key = extract_emotion_func(file)
                # If the emotion key is in the emotion_map, process the file
                if emotion_key in emotion_map:
                    # Get the canonical emotion label from the map
                    emotion = emotion_map[emotion_key]
                    # Extract audio features using the extract_features function
                    feature = extract_features(file_path)
                    # Append the features and labels to their respective lists
                    features.append(feature)
                    labels.append(emotion)
                else:
                    print(f"Warning: Emotion '{emotion_key}' not found in emotion_map for file: {file}")
    return np.array(features), np.array(labels)
