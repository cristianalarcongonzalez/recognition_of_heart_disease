import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Definir el directorio donde se encuentran los datos
data_dir = "ruta/a/tu/directorio/con/los/datos"

# Función para cargar archivos de audio y sus etiquetas
def load_data(data_dir):
    audio_files = []
    labels = []
    for file in os.listdir(data_dir):
        if file.endswith(".wav"):
            audio_path = os.path.join(data_dir, file)
            label_path = os.path.join(data_dir, file.replace(".wav", ".hea"))
            audio, sr = librosa.load(audio_path, sr=None)
            with open(label_path, 'r') as f:
                label = f.readlines()[0].strip()
            audio_files.append(audio)
            labels.append(label)
    return audio_files, labels

audio_files, labels = load_data(data_dir)

# Exploración básica
print(f"Número de archivos de audio: {len(audio_files)}")
print(f"Etiquetas disponibles: {set(labels)}")

# Preprocesamiento de Audio
def extract_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

features = [extract_features(audio) for audio in audio_files]

# Conversión de etiquetas a numéricas
label_dict = {label: idx for idx, label in enumerate(set(labels))}
numeric_labels = [label_dict[label] for label in labels]

# División del conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, numeric_labels, test_size=0.2, random_state=42)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Verificación de los tamaños de los conjuntos de datos
print(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
print(f"Tamaño del conjunto de prueba: {len(X_test)}")
