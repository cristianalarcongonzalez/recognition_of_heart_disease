import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, SimpleRNN
import flet as ft

# Definir el directorio donde se encuentran los datos
data_dir = "training/training-a"  # Cambia esto a la ruta correcta

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

# Cargar los datos
audio_files, labels = load_data(data_dir)

# Exploración básica
unique_labels = sorted(set(labels))  # Ordenamos las etiquetas
label_counts = {label: labels.count(label) for label in unique_labels}

# Preprocesamiento de Audio
def extract_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

features = [extract_features(audio) for audio in audio_files]

# Conversión de etiquetas a numéricas
label_dict = {label: idx for idx, label in enumerate(unique_labels)}
numeric_labels = [label_dict[label] for label in labels]

# División del conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, numeric_labels, test_size=0.2, random_state=42)
num_classes = len(unique_labels)
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Verificar las dimensiones de las etiquetas
print(f"Dimensiones de y_train: {y_train.shape}")
print(f"Dimensiones de y_test: {y_test.shape}")

# Definir el modelo CNN
model_cnn = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(13, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Asegurarse de que num_classes sea correcto
])

# Compilar el modelo CNN
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo CNN
history_cnn = model_cnn.fit(np.array(X_train).reshape(-1, 13, 1), y_train, epochs=30, batch_size=32, validation_data=(np.array(X_test).reshape(-1, 13, 1), y_test))

# Evaluación del modelo CNN
score_cnn = model_cnn.evaluate(np.array(X_test).reshape(-1, 13, 1), y_test)

# Definir el modelo RNN
model_rnn = Sequential([
    SimpleRNN(100, input_shape=(13, 1)),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Asegurarse de que num_classes sea correcto
])

# Compilar el modelo RNN
model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo RNN
history_rnn = model_rnn.fit(np.array(X_train).reshape(-1, 13, 1), y_train, epochs=30, batch_size=32, validation_data=(np.array(X_test).reshape(-1, 13, 1), y_test))

# Evaluación del modelo RNN
score_rnn = model_rnn.evaluate(np.array(X_test).reshape(-1, 13, 1), y_test)

# Comparativa de modelos
comparison_result = "El modelo CNN tiene mejor rendimiento." if score_cnn[1] > score_rnn[1] else "El modelo RNN tiene mejor rendimiento."

# Función para predecir la enfermedad cardíaca a partir de un nuevo archivo de audio
def predict_heart_disease(file_path, model, label_dict):
    audio, sr = librosa.load(file_path, sr=None)
    features = extract_features(audio).reshape(1, -1, 1)
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)
    predicted_label_name = list(label_dict.keys())[list(label_dict.values()).index(predicted_label)]
    prediction_percentage = np.max(prediction) * 100  # Obtener el porcentaje de la predicción
    return predicted_label_name, prediction_percentage

# Generar gráfica comparativa de precisión de los modelos
plt.figure(figsize=(10, 5))
plt.plot(history_cnn.history['accuracy'], label='Precisión de entrenamiento CNN')
plt.plot(history_cnn.history['val_accuracy'], label='Precisión de validación CNN')
plt.plot(history_rnn.history['accuracy'], label='Precisión de entrenamiento RNN')
plt.plot(history_rnn.history['val_accuracy'], label='Precisión de validación RNN')
plt.title('Comparativa de precisión de los modelos')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend(loc='upper left')
plt.savefig("model_comparison_accuracy.png")
plt.close()


# Crear gráficos para mostrar los resultados
def create_plots():
    # Conteo de etiquetas
    plt.figure(figsize=(10, 6))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.title("Conteo de Etiquetas")
    plt.xlabel("Etiquetas")
    plt.ylabel("Conteo")
    plt.savefig("label_counts.png")

    # Pérdida y precisión del entrenamiento CNN
    plt.figure(figsize=(10, 6))
    plt.plot(history_cnn.history['loss'], label='Entrenamiento - Pérdida')
    plt.plot(history_cnn.history['val_loss'], label='Validación - Pérdida')
    plt.title('Modelo CNN - Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.savefig('cnn_loss.png')

    plt.figure(figsize=(10, 6))
    plt.plot(history_cnn.history['accuracy'], label='Entrenamiento - Precisión')
    plt.plot(history_cnn.history['val_accuracy'], label='Validación - Precisión')
    plt.title('Modelo CNN - Precisión durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.savefig('cnn_accuracy.png')

    plt.figure(figsize=(10, 6))
    plt.plot(history_rnn.history['loss'], label='Entrenamiento - Pérdida')
    plt.plot(history_rnn.history['val_loss'], label='Validación - Pérdida')
    plt.title('Modelo RNN - Pérdida durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.savefig('rnn_loss.png')

    plt.figure(figsize=(10, 6))
    plt.plot(history_rnn.history['accuracy'], label='Entrenamiento - Precisión')
    plt.plot(history_rnn.history['val_accuracy'], label='Validación - Precisión')
    plt.title('Modelo RNN - Precisión durante el entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.savefig('rnn_accuracy.png')

create_plots()
result_text = ft.Text()
# Interfaz gráfica con flet
def main(page: ft.Page):
    page.title = "Reconocimiento de Enfermedades Cardíacas"
    
    def on_file_upload(e: ft.FilePickerResultEvent):
        if e.files:
            uploaded_file_path = e.files[0].path
            prediction_cnn, percentage_cnn = predict_heart_disease(uploaded_file_path, model_cnn, label_dict)
            prediction_rnn, percentage_rnn = predict_heart_disease(uploaded_file_path, model_rnn, label_dict)
            result_text.value = (
                f"Predicción del modelo CNN: {prediction_cnn} ({percentage_cnn:.2f}%)\n"
                f"Predicción del modelo RNN: {prediction_rnn} ({percentage_rnn:.2f}%)"
            )
            page.update()

    file_picker = ft.FilePicker(on_result=on_file_upload)
    page.overlay.append(file_picker)  # Agregar file_picker a la página

    page.add(
        ft.Column([
            ft.Text("Sube un archivo de audio para predecir la enfermedad cardíaca:"),
            ft.ElevatedButton("Seleccionar archivo", on_click=lambda _: file_picker.pick_files(allow_multiple=False)),
            result_text,  
            ft.Image(src="model_comparison_accuracy.png"),
            ft.Image(src="cnn_loss.png"),
            ft.Image(src="cnn_accuracy.png"),
            ft.Image(src="rnn_loss.png"),
            ft.Image(src="rnn_accuracy.png")
        ])
        
    )

ft.app(target=main)

