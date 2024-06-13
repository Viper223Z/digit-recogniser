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
        try:
            path = 'mnist.npz'
            with np.load(path, allow_pickle=True) as f:
                train_images, train_labels = f['x_train'], f['y_train']
                test_images, test_labels = f['x_test'], f['y_test']

            train_images = train_images / 255.0
            test_images = test_images / 255.0

            model = models.Sequential([
                Input(shape=(28, 28)),
                layers.Flatten(),
                layers.Dense(128, activation='relu'),
                layers.Dense(10, activation='softmax')
            ])

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            epochs = 10
            for epoch in range(epochs):
                model.fit(train_images, train_labels, epochs=1, validation_data=(test_images, test_labels), verbose=2)
                progress = (epoch + 1) / epochs * 100
                self.progress_callback(progress)

            model.save('my_model.keras')
        except Exception as e:
            self.progress_callback(0)
            print(f"Error during model creation: {str(e)}")

def create_and_save_model(progress_callback):
    generator = ModelGenerator(progress_callback)
    generator.start()
