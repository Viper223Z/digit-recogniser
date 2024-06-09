import tensorflow as tf
import cv2
import numpy as np

def load_and_predict(image_path):
    # Wczytaj model z pliku
    loaded_model = tf.keras.models.load_model('my_model.keras')

    # Wczytaj obraz, na którym chcesz dokonać predykcji
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Wczytaj obraz w skali szarości

    if image is None:
        print("Error: Could not load image.")
        return None

    # Zmień rozmiar obrazu na 28x28, aby pasował do modelu
    resized_image = cv2.resize(image, (28, 28))

    # Zastosuj binaryzację (próg) do obrazu - konwersja na obraz binarny (czarno-biały)
    _, thresholded_image = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY_INV)

    # Normalizuj obraz do zakresu [0, 1]
    normalized_image = thresholded_image / 255.0

    # Rozszerz wymiary obrazu, aby pasowały do wymagań modelu
    input_image = np.expand_dims(normalized_image, axis=0)
    input_image = np.expand_dims(input_image, axis=-1)

    # Debugowanie: Wyświetl przetworzony obraz
    print("Processed image shape:", input_image.shape)

    # Wykonaj predykcję za pomocą wczytanego modelu
    predictions = loaded_model.predict(input_image)

    # Debugowanie: Wyświetl wartości predykcji
    print("Predictions:", predictions)

    # Wypisz wyniki predykcji
    predicted_number = np.argmax(predictions)
    return predicted_number
