import os
import numpy as np
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CNNTrainer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.model = None

    def load_processed_data(self):
        try:
            X = np.load(os.path.join(self.data_dir, 'processed_data.npy'))
            y = np.load(os.path.join(self.data_dir, 'labels.npy'))
            return X, y
        except Exception as e:
            logging.error(f"Error in loading processed data: {e}")
            raise

    def preprocess_labels(self, y):
        try:
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            y_categorical = to_categorical(y_encoded)
            return y_categorical, label_encoder
        except Exception as e:
            logging.error(f"Error in preprocessing labels: {e}")
            raise

    def build_model(self, input_shape, num_classes):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def plot_history(self, history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')
        
        plt.show()

    def train_cnn(self):
        try:
            X, y = self.load_processed_data()
            y, label_encoder = self.preprocess_labels(y)
            input_shape = X.shape[1:]
            num_classes = y.shape[1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.model = self.build_model(input_shape, num_classes)
            history = self.model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
            
            self.plot_history(history)
            
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test)
            logging.info(f'Test accuracy: {test_accuracy}')
            
            model_path = os.path.join(self.data_dir, 'cnn_model.h5')
            self.model.save(model_path)
            logging.info(f"Model saved to {model_path}")
        except Exception as e:
            logging.error(f"Error in training CNN: {e}")
            raise

if __name__ == "__main__":
    data_dir = "SaveData"
    trainer = CNNTrainer(data_dir)
    trainer.train_cnn()
