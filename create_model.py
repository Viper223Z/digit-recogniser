import threading
import time
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Input
import numpy as np

class ModelGenerator(threading.Thread):
    def __init__(self, progress_callback):
        super().__init__()
        self.progress_callback = progress_callback

    def run(self):
        # Załaduj zbiór danych MNIST z lokalnego pliku
        path = 'mnist.npz'
        with np.load(path, allow_pickle=True) as f:
            train_images, train_labels = f['x_train'], f['y_train']
            test_images, test_labels = f['x_test'], f['y_test']

        # Normalizuj obrazy do zakresu [0, 1]
        train_images = train_images / 255.0
        test_images = test_images / 255.0

        # Zbuduj model
        model = models.Sequential([
            Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        # Kompilacja modelu
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Trenuj model
        epochs = 10
        for epoch in range(epochs):
            model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels), verbose=2)
            progress = (epoch + 1) / epochs * 100
            self.progress_callback(progress)

        # Zapisz model do pliku
        model.save('my_model.keras')

def create_and_save_model(progress_callback):
    generator = ModelGenerator(progress_callback)
    generator.start()
