import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout

# Definir el modelo CNN
model_cnn = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(13, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(set(labels)), activation='softmax')
])

# Compilar el modelo
model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model_cnn.fit(np.array(X_train).reshape(-1, 13, 1), y_train, epochs=30, batch_size=32, validation_data=(np.array(X_test).reshape(-1, 13, 1), y_test))

# Evaluación del modelo
score = model_cnn.evaluate(np.array(X_test).reshape(-1, 13, 1), y_test)
print(f"Precisión del modelo CNN: {score[1]}")
