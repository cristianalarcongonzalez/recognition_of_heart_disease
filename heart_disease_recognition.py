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

# Función para definir el modelo CNN con parámetros personalizables
def create_cnn_model(conv1_filters, conv2_filters, dense_units):
    model = Sequential([
        Conv1D(conv1_filters, kernel_size=3, activation='relu', input_shape=(13, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(conv2_filters, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Asegurarse de que num_classes sea correcto
    ])
    return model

# Función para definir el modelo RNN con parámetros personalizables
def create_rnn_model(rnn_units, dense_units):
    model = Sequential([
        SimpleRNN(rnn_units, input_shape=(13, 1)),
        Dense(dense_units, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Asegurarse de que num_classes sea correcto
    ])
    return model

# Definir el modelo CNN inicial
conv1_filters = 32
conv2_filters = 64
dense_units = 128
model_cnn = create_cnn_model(conv1_filters, conv2_filters, dense_units)

# Compilar el modelo CNN
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Definir el modelo RNN inicial
rnn_units = 100
model_rnn = create_rnn_model(rnn_units, dense_units)

# Compilar el modelo RNN
model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Crear gráficos para mostrar los resultados
def create_plots(history_cnn, history_rnn):
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

    # Pérdida y precisión del entrenamiento RNN
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

# Función para actualizar el modelo CNN con nuevos parámetros desde la interfaz
def update_cnn_model(conv1_filters, conv2_filters, dense_units):
    global model_cnn
    model_cnn = create_cnn_model(conv1_filters, conv2_filters, dense_units)
    model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Función para actualizar el modelo RNN con nuevos parámetros desde la interfaz
def update_rnn_model(rnn_units, dense_units):
    global model_rnn
    model_rnn = create_rnn_model(rnn_units, dense_units)
    model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Interfaz gráfica con flet
def main(page: ft.Page):
    page.title = "Reconocimiento de Enfermedades Cardíacas"

    selected_model = ft.Dropdown(
        options=[
            ft.dropdown.Option(key="CNN", text="CNN"),
            ft.dropdown.Option(key="RNN", text="RNN")
        ],
        label="Selecciona el modelo",
        value="CNN",
        width=200
    )

    conv1_input = ft.TextField(label="Filtros de Conv1D (1ª capa)", value="32", width=200)
    conv2_input = ft.TextField(label="Filtros de Conv1D (2ª capa)", value="64", width=200)
    cnn_dense_input = ft.TextField(label="Unidades de Dense (CNN)", value="128", width=200)

    rnn_units_input = ft.TextField(label="Unidades de RNN", value="100", width=200)
    rnn_dense_input = ft.TextField(label="Unidades de Dense (RNN)", value="128", width=200)

    epochs_input = ft.TextField(label="Número de épocas", value="30", width=200)

    result_text = ft.Text()

    def update_models(e):
        conv1_filters = int(conv1_input.value)
        conv2_filters = int(conv2_input.value)
        cnn_dense_units = int(cnn_dense_input.value)

        rnn_units = int(rnn_units_input.value)
        rnn_dense_units = int(rnn_dense_input.value)

        epochs = int(epochs_input.value)

        global model_cnn, model_rnn

        update_cnn_model(conv1_filters, conv2_filters, cnn_dense_units)
        update_rnn_model(rnn_units, rnn_dense_units)

        history_cnn = model_cnn.fit(
            np.array(X_train).reshape(-1, 13, 1), y_train,
            epochs=epochs, batch_size=32,
            validation_data=(np.array(X_test).reshape(-1, 13, 1), y_test)
        )
        history_rnn = model_rnn.fit(
            np.array(X_train).reshape(-1, 13, 1), y_train,
            epochs=epochs, batch_size=32,
            validation_data=(np.array(X_test).reshape(-1, 13, 1), y_test)
        )
        create_plots(history_cnn, history_rnn)
        result_text.value = "Modelos actualizados y entrenados con los nuevos parámetros."
        page.update()
  

    update_button = ft.ElevatedButton("Actualizar y entrenar modelos", on_click=update_models)
    def on_file_upload(e: ft.FilePickerResultEvent):
        if e.files:
            uploaded_file_path = e.files[0].path
            model = model_cnn if selected_model.value == "CNN" else model_rnn
            prediction, percentage = predict_heart_disease(uploaded_file_path, model, label_dict)
            result_text.value = (
                f"Predicción del modelo {selected_model.value}: {prediction} ({percentage:.2f}%)"
            )
            page.update()

    file_picker = ft.FilePicker(on_result=on_file_upload)
    page.overlay.append(file_picker)
    page.add(
        ft.Column([
            ft.Text("Sube un archivo de audio para predecir la enfermedad cardíaca:"),
            selected_model,
            ft.Row([conv1_input, conv2_input, cnn_dense_input]),
            ft.Row([rnn_units_input, rnn_dense_input]),
            epochs_input,
            update_button,
            ft.ElevatedButton("Seleccionar archivo", on_click=lambda _: file_picker.pick_files(allow_multiple=False)),
            result_text,
            ft.Row([
                ft.Image(src="model_comparison_accuracy.png", width=400, height=300),
                ft.Image(src="cnn_loss.png", width=400, height=300),
                ft.Image(src="cnn_accuracy.png", width=400, height=300),
            ]),
            ft.Row([
                ft.Image(src="rnn_loss.png", width=400, height=300),
                ft.Image(src="rnn_accuracy.png", width=400, height=300)
            ])
        ])
    )

ft.app(target=main)
